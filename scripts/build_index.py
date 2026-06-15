#!/usr/bin/env python3
"""
Ingest documents from ``data/documents/`` into the FAISS index.

Reads all ``.md`` and ``.txt`` files, splits them into overlapping chunks,
embeds via the ONNX embedding service, and persists the FAISS index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --docs ./my_docs --chunk-size 256
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import `app.*`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.embedding_service import get_embedding_service
from app.services.retrieval_service import FAISSRetriever


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    Split *text* into overlapping chunks by paragraph/sentence boundaries.

    Falls back to character-level splitting when paragraphs are too large.
    """
    # Normalise whitespace
    text = text.strip()
    if not text:
        return []

    # Split into paragraphs first
    paragraphs = re.split(r"\n{2,}", text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # If a single paragraph exceeds chunk_size, split it further
        if para_len > chunk_size:
            # Flush current buffer
            if current:
                chunks.append("\n\n".join(current))
                # Keep overlap by retaining the tail
                overlap_text = "\n\n".join(current)
                current = [overlap_text[-chunk_overlap:]] if chunk_overlap else []
                current_len = len(current[0]) if current else 0

            # Split the big paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len + len(sent) + 1 > chunk_size and current:
                    chunks.append(" ".join(current))
                    tail = " ".join(current)
                    current = [tail[-chunk_overlap:]] if chunk_overlap else []
                    current_len = len(current[0]) if current else 0
                current.append(sent)
                current_len += len(sent) + 1
            continue

        if current_len + para_len + 2 > chunk_size and current:
            chunks.append("\n\n".join(current))
            tail = "\n\n".join(current)
            current = [tail[-chunk_overlap:]] if chunk_overlap else []
            current_len = len(current[0]) if current else 0

        current.append(para)
        current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current))

    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from documents")
    parser.add_argument(
        "--docs",
        default=None,
        help="Document directory (default: from .env / settings)",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    docs_dir = Path(args.docs) if args.docs else settings.documents_dir
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap

    if not docs_dir.exists():
        print(f"✗ Document directory not found: {docs_dir}")
        sys.exit(1)

    # Collect all text files
    files = sorted(
        f for f in docs_dir.rglob("*") if f.suffix.lower() in {".md", ".txt"}
    )
    if not files:
        print(f"✗ No .md or .txt files found in {docs_dir}")
        sys.exit(1)

    print(f"[1/3] Found {len(files)} document(s) in {docs_dir}")

    # Read and chunk
    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    for fpath in files:
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "source": str(fpath.relative_to(docs_dir)),
                    "chunk_index": i,
                }
            )
        print(f"   • {fpath.name}: {len(chunks)} chunk(s)")

    print(f"\n[2/3] Embedding {len(all_chunks)} chunk(s)...")

    # Build index
    embedder = get_embedding_service()
    retriever = FAISSRetriever(embedder=embedder)
    retriever.add_documents(all_chunks, all_metadata)

    # Save
    retriever.save()
    print(f"\n[3/3] Index saved to {settings.faiss_index_dir}")
    print(f"   Total vectors: {retriever.size}")
    print(f"   Dimension:     {retriever.dimension}")
    print("\n✅ Index build complete!")


if __name__ == "__main__":
    main()
