from pathlib import Path

from llama_index.core import SimpleDirectoryReader


def load_documents(directory: str):
    """
    Load supported documents from a directory.

    Returns:
        A list of LlamaIndex document objects.
    """
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    reader = SimpleDirectoryReader(
        input_dir=str(path),
        recursive=True,
    )

    return reader.load_data()