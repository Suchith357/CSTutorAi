"""Ingestion pipeline: documents on disk -> chunks with metadata -> embeddings -> persistent index.

Pipeline stages:

    data/raw -> loader -> chunker -> metadata -> embeddings -> FAISS -> persisted index

Run from the repository root:

    python -m backend.app.rag.ingestion

Artifacts written to the index directory:
    index.faiss      FAISS inner-product index over normalized chunk embeddings
    metadata.jsonl   one DocumentChunk record (text + provenance) per indexed vector
    manifest.json    build configuration (model, dimension, chunking params, counts)
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Document

from backend.app.core.config import settings
from backend.app.rag.chunker import chunk_documents
from backend.app.rag.embeddings import EmbeddingModel
from backend.app.rag.loader import load_documents
from backend.app.rag.schemas import DocumentChunk, DocumentMetadata
from backend.app.rag.vector_store import VectorStore

METADATA_FILE = "metadata.jsonl"
MANIFEST_FILE = "manifest.json"


def derive_metadata(document: Document) -> DocumentMetadata:
    """Derive provenance metadata for a loaded document."""
    file_path = (
        document.metadata.get("file_path")
        or document.metadata.get("file_name")
        or "unknown"
    )
    stem = Path(file_path).stem
    display_name = stem.replace("_", " ").replace("-", " ").strip() or file_path

    page_raw = document.metadata.get("page") or document.metadata.get("page_label")
    try:
        page = int(page_raw) if page_raw is not None else None
    except (TypeError, ValueError):
        page = None

    return DocumentMetadata(
        title=document.metadata.get("title") or display_name,
        source=document.metadata.get("source") or Path(file_path).name,
        topic=document.metadata.get("topic") or display_name,
        url=document.metadata.get("url"),
        license=document.metadata.get("license"),
        page=page,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def document_to_chunks(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Split a loaded document into chunks and attach derived metadata."""
    nodes = chunk_documents(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    base_metadata = derive_metadata(document)

    chunks: list[DocumentChunk] = []
    for node in nodes:
        text = node.get_content().strip()
        if not text:
            continue
        chunks.append(
            DocumentChunk(text=text, metadata=base_metadata.model_copy())
        )
    return chunks


def save_index(
    index_dir: str | Path,
    chunks: list[DocumentChunk],
    embedding_model: EmbeddingModel,
    num_documents: int,
    chunk_size: int,
    chunk_overlap: int,
):
    """Embed chunks and persist the FAISS index, chunk records, and manifest."""
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)

    embeddings = embedding_model.encode_documents([chunk.text for chunk in chunks])

    store = VectorStore(embeddings.shape[1])
    store.add(embeddings)

    # Write chunk records first and the index last: a missing index.faiss
    # signals an incomplete build instead of a silently mismatched one.
    metadata_path = directory / METADATA_FILE
    with metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": embedding_model.model_name,
        "dimension": int(embeddings.shape[1]),
        "num_documents": num_documents,
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    store.save(directory)


def load_chunk_records(index_dir: str | Path) -> list[DocumentChunk]:
    """Load persisted chunk records (text + provenance) from an index directory."""
    path = Path(index_dir) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No chunk records found at: {path}. Run ingestion first: "
            "python -m backend.app.rag.ingestion"
        )

    chunks: list[DocumentChunk] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            chunks.append(DocumentChunk.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid chunk record at line {line_number}: {exc}") from exc
    return chunks


def load_manifest(index_dir: str | Path) -> dict | None:
    """Load the build manifest if present (older indexes may not have one)."""
    path = Path(index_dir) / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ingest(
    raw_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """Run the full ingestion pipeline and persist the knowledge index."""
    raw_dir = Path(raw_dir or settings.raw_data_path)
    index_dir = Path(index_dir or settings.index_path)
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    documents = load_documents(str(raw_dir))
    if not documents:
        raise ValueError(f"No documents found to ingest in: {raw_dir}")

    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(
            document_to_chunks(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    if not chunks:
        raise ValueError(f"Ingestion produced no chunks from documents in: {raw_dir}")

    embedding_model = EmbeddingModel()
    save_index(
        index_dir=index_dir,
        chunks=chunks,
        embedding_model=embedding_model,
        num_documents=len(documents),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return {
        "raw_dir": str(raw_dir),
        "index_dir": str(index_dir),
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": embedding_model.model_name,
        "dimension": embedding_model.dimension,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build the CSTutorAI knowledge index from documents in data/raw."
    )
    parser.add_argument("--raw-dir", default=None, help="Defaults to RAW_DATA_DIR from settings")
    parser.add_argument("--index-dir", default=None, help="Defaults to INDEX_DIR from settings")
    args = parser.parse_args()

    report = ingest(raw_dir=args.raw_dir, index_dir=args.index_dir)
    print("Ingestion complete:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
