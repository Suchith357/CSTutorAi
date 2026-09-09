from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """
    Split documents into smaller chunks for retrieval.
    """

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.get_nodes_from_documents(documents)