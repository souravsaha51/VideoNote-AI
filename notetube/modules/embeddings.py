"""
modules/embeddings.py
Google Generative AI Embeddings singleton for NoteTube.
"""

import os
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings


EMBEDDING_MODEL = "models/gemini-embedding-001"


@lru_cache(maxsize=1)
def get_embeddings(api_key: str = "") -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached GoogleGenerativeAIEmbeddings instance.

    Using lru_cache avoids re-creating the client on every pipeline run,
    which is wasteful and can hit rate limits faster.

    Args:
        api_key: Gemini API key. Reads from GEMINI_API_KEY env var if empty.

    Returns:
        Configured GoogleGenerativeAIEmbeddings instance.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=key,
    )
