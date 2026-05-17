"""
modules/summarizer.py
AI summary, insight, and timestamp generation for NoteTube.
"""

from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.prompts import (
    SUMMARY_SHORT_PROMPT,
    SUMMARY_DETAILED_PROMPT,
    INSIGHTS_PROMPT,
    TIMESTAMPS_PROMPT,
    NOTES_PROMPT,
)


def _run_chain(prompt_template, llm: ChatGoogleGenerativeAI, **kwargs) -> str:
    """Helper: build a chain from prompt + llm and invoke it."""
    chain = prompt_template | llm
    result = chain.invoke(kwargs)
    return result.content.strip()


def generate_short_summary(transcript: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Generate a concise 3–5 sentence summary of a video transcript.

    Args:
        transcript: Cleaned transcript text.
        llm: ChatGoogleGenerativeAI instance.

    Returns:
        Short summary string.
    """
    # Truncate very long transcripts to avoid token limits
    truncated = transcript[:12000] if len(transcript) > 12000 else transcript
    return _run_chain(SUMMARY_SHORT_PROMPT, llm, transcript=truncated)


def generate_detailed_summary(transcript: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Generate a structured, detailed summary with sections.

    Args:
        transcript: Cleaned transcript text.
        llm: ChatGoogleGenerativeAI instance.

    Returns:
        Markdown-formatted detailed summary.
    """
    truncated = transcript[:15000] if len(transcript) > 15000 else transcript
    return _run_chain(SUMMARY_DETAILED_PROMPT, llm, transcript=truncated)


def generate_insights(
    transcript: str,
    llm: ChatGoogleGenerativeAI,
) -> List[str]:
    """
    Extract 5–8 key insights from a transcript.

    Args:
        transcript: Cleaned transcript text.
        llm: ChatGoogleGenerativeAI instance.

    Returns:
        List of insight strings (one per bullet point).
    """
    truncated = transcript[:12000] if len(transcript) > 12000 else transcript
    raw = _run_chain(INSIGHTS_PROMPT, llm, transcript=truncated)

    # Parse bullet lines; filter out empty lines
    lines = [
        line.strip().lstrip("•-*").strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return [l for l in lines if l]


def generate_timestamp_highlights(
    transcript: str,
    llm: ChatGoogleGenerativeAI,
) -> List[Dict]:
    """
    Identify important moments in the video from the transcript.

    Args:
        transcript: Cleaned transcript text.
        llm: ChatGoogleGenerativeAI instance.

    Returns:
        List of {'label': str, 'position': str, 'reason': str} dicts.
    """
    truncated = transcript[:12000] if len(transcript) > 12000 else transcript
    raw = _run_chain(TIMESTAMPS_PROMPT, llm, transcript=truncated)

    highlights = []
    current: Dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            if current:
                highlights.append(current)
                current = {}
            continue
        if line.startswith("-") and "label" not in current:
            current["label"] = line.lstrip("-•* ").strip()
        elif "early" in line.lower() or "middle" in line.lower() or "late" in line.lower():
            for pos in ["early", "middle", "late"]:
                if pos in line.lower():
                    current["position"] = pos
                    break
        elif line and "label" in current and "reason" not in current:
            current["reason"] = line

    if current:
        highlights.append(current)

    # Ensure each highlight has all keys
    return [
        {
            "label": h.get("label", "Key moment"),
            "position": h.get("position", "middle"),
            "reason": h.get("reason", ""),
        }
        for h in highlights
        if "label" in h
    ]


def generate_learning_notes(transcript: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Generate structured learning notes from a transcript.

    Args:
        transcript: Cleaned transcript text.
        llm: ChatGoogleGenerativeAI instance.

    Returns:
        Markdown-formatted learning notes.
    """
    truncated = transcript[:12000] if len(transcript) > 12000 else transcript
    return _run_chain(NOTES_PROMPT, llm, transcript=truncated)
