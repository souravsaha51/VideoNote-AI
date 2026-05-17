"""
utils/helpers.py
Formatting, text cleaning, and general helper utilities for NoteTube.
"""

import re
from typing import List, Dict


# Noise tokens commonly found in auto-generated captions
_NOISE_PATTERNS = [
    r"\[Music\]", r"\[Applause\]", r"\[Laughter\]", r"\[Noise\]",
    r"\[Intro\]", r"\[Outro\]", r"\[music\]", r"\[applause\]",
    r"\[MUSIC\]", r"\[APPLAUSE\]", r"♪+", r"&amp;",
]


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to a human-readable timestamp string.

    Args:
        seconds: Time in seconds (float or int).

    Returns:
        'MM:SS' for durations under 1 hour, 'HH:MM:SS' otherwise.
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_youtube_url(video_id: str, seconds: float = 0) -> str:
    """
    Build a YouTube URL that deep-links to a specific timestamp.

    Args:
        video_id: 11-character YouTube video ID.
        seconds: Start time in seconds.

    Returns:
        Full YouTube watch URL with &t=<seconds>s parameter.
    """
    t = int(seconds)
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"


def clean_transcript_text(raw: str) -> str:
    """
    Remove noise tokens from raw transcript text.

    Args:
        raw: Raw transcript string.

    Returns:
        Cleaned transcript with noise tokens removed.
    """
    text = raw
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Collapse multiple whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_len: int = 500, suffix: str = "…") -> str:
    """
    Safely truncate text to a maximum character length.

    Args:
        text: Input text.
        max_len: Maximum number of characters.
        suffix: String appended when truncated.

    Returns:
        Truncated string with suffix if over max_len.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + suffix


def segments_to_plain_text(segments: List) -> str:
    """
    Convert a list of transcript segments to a single plain-text string.
    Supports both dict {'text': ...} and FetchedTranscriptSnippet objects.

    Matches notebook pattern:
        " ".join(snippet.text for snippet in fetched)
    """
    parts = []
    for seg in segments:
        if isinstance(seg, dict):
            parts.append(seg.get("text", ""))
        else:
            parts.append(getattr(seg, "text", ""))
    return clean_transcript_text(" ".join(parts))


def word_count(text: str) -> int:
    """Return the number of words in a text string."""
    return len(text.split())


def reading_time_minutes(text: str, wpm: int = 200) -> int:
    """
    Estimate reading time in minutes.

    Args:
        text: Input text.
        wpm: Average words per minute (default 200).

    Returns:
        Estimated reading time in whole minutes (minimum 1).
    """
    return max(1, word_count(text) // wpm)
