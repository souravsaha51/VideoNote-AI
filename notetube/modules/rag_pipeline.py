"""
modules/rag_pipeline.py
Full RAG pipeline orchestrator for NoteTube.
"""

import os
from typing import Dict, Tuple, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from modules.transcript import get_transcript, TranscriptError
from modules.chunking import chunk_with_timestamps
from modules.embeddings import get_embeddings
from modules.vectorstore import build_vectorstore, get_retriever
from modules.summarizer import (
    generate_short_summary,
    generate_detailed_summary,
    generate_insights,
    generate_timestamp_highlights,
)


LLM_MODEL = "gemini-2.5-flash-lite"   # confirmed working in rag_project.ipynb
LLM_FALLBACK = "gemini-2.0-flash"


def get_llm(api_key: str = "", model: str = LLM_MODEL) -> ChatGoogleGenerativeAI:
    """
    Create a ChatGoogleGenerativeAI instance.

    Args:
        api_key: Gemini API key (reads from env if empty).
        model: Model identifier string.

    Returns:
        Configured LLM instance.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.3,
    )


def build_pipeline(
    video_id: str,
    language: str = "en",
    api_key: str = "",
    progress_callback=None,
) -> Tuple[FAISS, str, List[Dict]]:
    """
    Build the full RAG pipeline for a YouTube video.

    Workflow:
        transcript extraction → cleaning → chunking →
        embedding generation → FAISS indexing

    Args:
        video_id: 11-character YouTube video ID.
        language: Preferred transcript language.
        api_key: Gemini API key.
        progress_callback: Optional callable(step: int, total: int, msg: str).

    Returns:
        Tuple of (vectorstore, clean_transcript_text, segments).

    Raises:
        TranscriptError: If transcript cannot be fetched.
        Exception: On embedding or indexing failures.
    """
    total_steps = 4

    def _progress(step: int, msg: str):
        if progress_callback:
            progress_callback(step, total_steps, msg)

    # Step 1: Fetch transcript
    _progress(1, "📥 Fetching transcript from YouTube…")
    transcript_text, segments = get_transcript(video_id, language=language)

    # Step 2: Chunk with timestamps
    _progress(2, "✂️ Splitting transcript into chunks…")
    docs = chunk_with_timestamps(segments)

    # Step 3: Generate embeddings
    _progress(3, "🧠 Generating semantic embeddings…")
    embeddings = get_embeddings(api_key=api_key)

    # Step 4: Build FAISS index
    _progress(4, "🗄️ Building vector database…")
    vectorstore = build_vectorstore(docs, embeddings)

    return vectorstore, transcript_text, segments


def run_full_analysis(
    video_id: str,
    language: str = "en",
    api_key: str = "",
    summary_type: str = "short",
    progress_callback=None,
) -> Dict:
    """
    Run the complete analysis pipeline: build RAG + generate all summaries.

    Args:
        video_id: YouTube video ID.
        language: Transcript language.
        api_key: Gemini API key.
        summary_type: 'short' or 'detailed'.
        progress_callback: Optional callable(step, total, message).

    Returns:
        Dict with keys:
            vectorstore, transcript, segments,
            summary_short, summary_detailed, insights, timestamps
    """
    # Build RAG pipeline
    vectorstore, transcript, segments = build_pipeline(
        video_id=video_id,
        language=language,
        api_key=api_key,
        progress_callback=progress_callback,
    )

    # Create LLM — try preview model, fall back if needed
    try:
        llm = get_llm(api_key=api_key, model=LLM_MODEL)
        # Quick validation ping
        llm.invoke("hi")
    except Exception:
        llm = get_llm(api_key=api_key, model=LLM_FALLBACK)

    results = {
        "vectorstore": vectorstore,
        "transcript": transcript,
        "segments": segments,
        "llm": llm,
    }

    # Generate summaries
    if summary_type == "detailed":
        results["summary_short"] = generate_short_summary(transcript, llm)
        results["summary_detailed"] = generate_detailed_summary(transcript, llm)
    else:
        results["summary_short"] = generate_short_summary(transcript, llm)
        results["summary_detailed"] = None

    results["insights"] = generate_insights(transcript, llm)
    results["timestamps"] = generate_timestamp_highlights(transcript, llm)

    return results
