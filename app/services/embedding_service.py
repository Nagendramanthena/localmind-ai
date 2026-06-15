"""
ONNX-powered embedding service.

Loads a Sentence-Transformer model exported to ONNX and runs inference
using ONNX Runtime.  On Windows the DirectML execution provider is used
for GPU acceleration; everywhere else it falls back to the CPU provider.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_instance: OnnxEmbeddingService | None = None


def get_embedding_service() -> OnnxEmbeddingService:
    """Return (or lazily create) the singleton embedding service."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = OnnxEmbeddingService(settings.embedding_model_dir)
    return _instance


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class OnnxEmbeddingService:
    """
    Embed text using an ONNX model with DirectML / CPU.

    The service handles tokenization, inference, mean-pooling and
    L2-normalisation so callers get unit-length vectors ready for
    cosine-similarity search (or inner-product on FAISS ``IndexFlatIP``).
    """

    def __init__(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)

        # Locate the ONNX file
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx file found in {model_dir}. "
                "Run `python scripts/export_model.py` first."
            )

        onnx_path = str(onnx_files[0])
        providers = settings.onnx_provider_list

        log.info(
            "loading_onnx_model",
            path=onnx_path,
            requested_providers=providers,
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Cache input names for fast lookup
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        # Determine embedding dimension from a probe run
        self._dim = self._probe_dimension()

        log.info(
            "onnx_model_ready",
            active_providers=self._session.get_providers(),
            embedding_dim=self._dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality (e.g. 384 for MiniLM)."""
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Parameters
        ----------
        texts : list[str]
            One or more strings to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), dimension)`` of L2-normalised float32 vectors.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )

        # Build feed dict (only keys the model expects)
        feed = {k: v for k, v in encoded.items() if k in self._input_names}

        # Inference → (batch, seq_len, hidden_dim)
        token_embeddings = self._session.run(None, feed)[0]

        # Mean-pooling
        mask = encoded["attention_mask"]
        mask_expanded = np.expand_dims(mask, axis=-1).astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embeddings = summed / counts

        # L2 normalise
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        sentence_embeddings = sentence_embeddings / norms

        return sentence_embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _probe_dimension(self) -> int:
        """Run a single dummy inference to discover the hidden dimension."""
        dummy = self.embed(["dim probe"])
        return dummy.shape[1]
