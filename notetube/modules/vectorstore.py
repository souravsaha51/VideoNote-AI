"""
modules/vectorstore.py
FAISS vector database build and retrieval for NoteTube.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_vectorstore(
    docs: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> FAISS:
    """
    Build a FAISS vector store from a list of Documents.

    Args:
        docs: Chunked transcript Documents (with optional timestamp metadata).
        embeddings: GoogleGenerativeAIEmbeddings instance.

    Returns:
        In-memory FAISS vector store.

    Raises:
        ValueError: If docs list is empty.
    """
    if not docs:
        raise ValueError("Cannot build vector store from empty document list.")

    return FAISS.from_documents(docs, embeddings)


def get_retriever(vectorstore: FAISS, k: int = 4):
    """
    Create a retriever from a FAISS vector store.

    Args:
        vectorstore: Populated FAISS index.
        k: Number of top similar chunks to retrieve per query.

    Returns:
        LangChain retriever object.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def similarity_search(
    vectorstore: FAISS,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Perform a direct similarity search and return matching Documents.

    Args:
        vectorstore: Populated FAISS index.
        query: Search query string.
        k: Number of results to return.

    Returns:
        List of most similar Documents.
    """
    return vectorstore.similarity_search(query, k=k)
