"""
FAISS-based semantic retrieval service.

Manages a flat inner-product index (``IndexFlatIP``) over L2-normalised
embeddings, giving cosine-similarity ranking.  A JSON sidecar stores
the original text chunks and any metadata.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from app.services.embedding_service import OnnxEmbeddingService, get_embedding_service
from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single chunk returned by a retrieval search."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_instance: FAISSRetriever | None = None


def get_retriever() -> FAISSRetriever:
    """Return (or lazily create) the singleton retriever."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FAISSRetriever()
    return _instance


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class FAISSRetriever:
    """
    Wraps a FAISS inner-product index with text/metadata storage.

    Thread safety: reads (``search``) are safe for concurrent access.
    Writes (``add_documents``) acquire an internal lock.
    """

    def __init__(
        self,
        embedder: OnnxEmbeddingService | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self._embedder = embedder or get_embedding_service()
        self._index_dir = Path(index_dir or settings.faiss_index_dir)
        self._write_lock = threading.Lock()

        # Try to load an existing index; otherwise create an empty one.
        if self._index_path.exists() and self._meta_path.exists():
            self._load()
        else:
            dim = self._embedder.dimension
            self._index = faiss.IndexFlatIP(dim)
            self._chunks: list[str] = []
            self._metadata: list[dict] = []
            log.info("created_empty_index", dim=dim)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self._index_dir / "faiss.index"

    @property
    def _meta_path(self) -> Path:
        return self._index_dir / "metadata.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    def add_documents(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> int:
        """
        Embed and add document chunks to the index.

        Returns the new total number of indexed vectors.
        """
        if not texts:
            return self.size

        metadata = metadata or [{} for _ in texts]
        if len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")

        embeddings = self._embedder.embed(texts)

        with self._write_lock:
            self._index.add(embeddings)
            self._chunks.extend(texts)
            self._metadata.extend(metadata)

        log.info("added_documents", count=len(texts), total=self.size)
        return self.size

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Embed *query* and return the top-K most similar chunks.

        Parameters
        ----------
        query : str
            Natural language query.
        top_k : int, optional
            Number of results to return (default from settings).

        Returns
        -------
        list[RetrievedChunk]
            Ranked list, highest score first.
        """
        if self.size == 0:
            return []

        top_k = min(top_k or settings.top_k, self.size)
        query_vec = self._embedder.embed([query])

        scores, indices = self._index.search(query_vec, top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS sentinel for "no result"
            results.append(
                RetrievedChunk(
                    text=self._chunks[idx],
                    score=float(score),
                    metadata=self._metadata[idx],
                )
            )
        return results

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps(
                {"chunks": self._chunks, "metadata": self._metadata},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("index_saved", path=str(self._index_dir), vectors=self.size)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load a previously saved index + metadata."""
        self._index = faiss.read_index(str(self._index_path))

        raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self._chunks = raw["chunks"]
        self._metadata = raw["metadata"]

        log.info("index_loaded", path=str(self._index_dir), vectors=self.size)
