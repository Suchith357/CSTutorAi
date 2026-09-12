"""OSTutorAI knowledge-base statistics (reads the persisted index; no LLM).

Run from the repository root:
    .venv/Scripts/python.exe scripts/kb_stats.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings  # noqa: E402
from backend.app.rag.ingestion import load_chunk_records, load_manifest  # noqa: E402


def main() -> int:
    index_dir = settings.index_path
    manifest = load_manifest(index_dir) or {}
    chunks = load_chunk_records(index_dir)

    documents = sorted({c.metadata.title for c in chunks})
    topics = Counter(c.metadata.topic for c in chunks)
    sources = Counter(c.metadata.source for c in chunks)

    print("=" * 60)
    print("OSTutorAI Knowledge Base")
    print("=" * 60)
    print(f"Documents (by title):     {len(documents)}")
    print(f"Source categories:        {len(sources)}")
    print(f"Topics:                   {len(topics)}")
    print(f"Chunks:                   {len(chunks)}")
    print(f"Indexed vectors:          {manifest.get('num_chunks', len(chunks))}")
    print(f"Embedding dimension:      {manifest.get('dimension', '?')}")
    print(f"Index type:               FAISS IndexFlatIP (exact inner product)")
    print(f"Embedding model:          {manifest.get('embedding_model', '?')}")
    print(f"Chunk size / overlap:     {manifest.get('chunk_size', '?')} / "
          f"{manifest.get('chunk_overlap', '?')}")
    print(f"Index directory:          {index_dir}")
    print()
    print("Chunks per topic:")
    for topic, count in sorted(topics.items()):
        print(f"  {topic:<22} {count}")
    print()
    print("Source categories:")
    for source, count in sorted(sources.items()):
        print(f"  {source:<40} {count} chunks")
    print()
    print("Document list (title | topic | chunks | license):")
    per_doc = Counter((c.metadata.title, c.metadata.topic) for c in chunks)
    licenses = {}
    for c in chunks:
        licenses.setdefault((c.metadata.title, c.metadata.topic), c.metadata.license)
    for (title, topic), count in sorted(per_doc.items()):
        print(f"  {title[:44]:<46} {topic:<22} {count}  {licenses[(title, topic)] or '-'}")
    print()
    avg = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
    print(f"Average chunk length:     {avg:.0f} characters")
    print(f"Manifest created:         {manifest.get('created_at', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
