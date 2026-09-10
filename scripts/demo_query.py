"""Manual demo of the complete RAG + LLM pipeline.

Run from the repository root (build the index first if needed):
    python -m backend.app.rag.ingestion
    python scripts/demo_query.py "What is deadlock?"

First run downloads the model (~1 GB) from the HuggingFace hub; later runs
load it from the local cache.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from backend.app.core.config import settings
    from backend.app.llm.client import create_llm_client
    from backend.app.llm.prompts import build_grounded_prompt
    from backend.app.rag.retriever import Retriever

    question = " ".join(sys.argv[1:]).strip() or "What is deadlock?"

    print("=" * 60)
    print("CSTutorAI - full pipeline demo")
    print("=" * 60)

    retriever = Retriever.from_prebuilt(settings.index_path)
    retrieved = retriever.retrieve(question)

    llm = create_llm_client()
    system, user = build_grounded_prompt(question, retrieved)
    print("\nGenerating answer with local model "
          f"{settings.llm_model_name} (first run downloads ~1 GB)...")

    answer = llm.generate(system, user)

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer or "(empty model response)")

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)
    if not retrieved:
        print("(no sources retrieved)")
    for i, chunk in enumerate(retrieved, start=1):
        print(f"[{i}] {chunk.metadata.title} (source: {chunk.metadata.source}) "
              f"score={chunk.score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
