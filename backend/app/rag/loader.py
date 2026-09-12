from pathlib import Path

from llama_index.core import SimpleDirectoryReader

# Files/directories under raw_data_dir that must NEVER be ingested as knowledge:
# - README.md files are repository usage notes, not course content.
# - fixtures/ holds deterministic test corpora (e.g., the old toy corpus).
_INGESTION_EXCLUDES = ["README*"]


def load_documents(directory: str, exclude_fixtures: bool = True):
    """
    Load supported knowledge documents from a directory.

    Test fixtures (data/raw/fixtures) and README files are excluded so the
    knowledge base contains only real course material.

    Args:
        directory: directory containing knowledge-base documents.
        exclude_fixtures: skip the 'fixtures' subdirectory (leave True for
            ingestion; tests use a dedicated temp directory instead).

    Returns:
        A list of LlamaIndex document objects.
    """
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    excludes = list(_INGESTION_EXCLUDES)
    if exclude_fixtures:
        excludes.append("fixtures/*")

    reader = SimpleDirectoryReader(
        input_dir=str(path),
        recursive=True,
        exclude=excludes,
    )

    return reader.load_data()
