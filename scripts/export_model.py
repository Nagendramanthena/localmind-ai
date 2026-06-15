#!/usr/bin/env python3
"""
Export a Sentence-Transformer model to ONNX format.

Uses Hugging Face Optimum for a clean, validated export that includes
the tokenizer config alongside the ONNX graph.

Usage:
    python scripts/export_model.py                         # default export
    python scripts/export_model.py --quantize              # + INT8 dynamic quantization
    python scripts/export_model.py --model bge-small-en-v1.5  # custom model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export embedding model to ONNX")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model ID (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output",
        default="./models/onnx",
        help="Output directory for the ONNX model (default: ./models/onnx)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization after export",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading model: {args.model}")

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    # ── Step 1: Export to ONNX ────────────────────────────────
    model = ORTModelForFeatureExtraction.from_pretrained(
        args.model,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[2/4] Saving ONNX model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Step 2: Optional quantization ─────────────────────────
    if args.quantize:
        print("[3/4] Applying INT8 dynamic quantization...")
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(output_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
        )
        quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
        print("   ✓ Quantized model saved")
    else:
        print("[3/4] Skipping quantization (pass --quantize to enable)")

    # ── Step 3: Validate the export ───────────────────────────
    print("[4/4] Validating exported model...")
    _validate_export(output_dir, tokenizer)

    print("\n✅ Export complete!")
    print(f"   Model directory: {output_dir}")
    print(f"   Files: {[f.name for f in output_dir.iterdir() if f.is_file()]}")


def _validate_export(model_dir: Path, tokenizer) -> None:
    """Run a quick inference to ensure the ONNX model produces valid embeddings."""
    import onnxruntime as ort

    # Find the ONNX file
    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        print("   ✗ No .onnx file found in output directory!")
        sys.exit(1)

    onnx_path = str(onnx_files[0])
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Tokenize a test sentence
    test_text = "This is a validation sentence for the embedding model."
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )

    # Run inference
    input_feed = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
    outputs = session.run(None, input_feed)

    # Validate shape — expect (1, seq_len, hidden_dim)
    token_embeddings = outputs[0]
    if token_embeddings.ndim != 3:
        print(f"   ✗ Unexpected output shape: {token_embeddings.shape}")
        sys.exit(1)

    hidden_dim = token_embeddings.shape[2]

    # Mean pooling
    attention_mask = inputs["attention_mask"]
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    sentence_embedding = (summed / counts).flatten()

    print(f"   ✓ Output dimension: {hidden_dim}")
    print(f"   ✓ Sentence embedding shape: {sentence_embedding.shape}")
    print(f"   ✓ Embedding norm: {np.linalg.norm(sentence_embedding):.4f}")


if __name__ == "__main__":
    main()
