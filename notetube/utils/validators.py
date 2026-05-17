"""
utils/validators.py
URL validation and input sanitization for NoteTube.
"""

import re
from typing import Tuple, Optional


# Regex patterns for all common YouTube URL formats
_YT_PATTERNS = [
    r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",          # watch?v=
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",                         # youtu.be/
    r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",                # embed/
    r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",               # shorts/
    r"(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})",                    # /v/
    r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})",                 # live/
    r"^([a-zA-Z0-9_-]{11})$",                                     # bare video ID
]


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract an 11-character YouTube video ID from any supported URL format.

    Args:
        url: Raw user input (URL or bare video ID).

    Returns:
        11-character video ID string, or None if not found.
    """
    if not url:
        return None

    url = url.strip()

    for pattern in _YT_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def validate_url(url: str) -> Tuple[bool, Optional[str], str]:
    """
    Validate a YouTube URL and extract the video ID.

    Args:
        url: Raw user input.

    Returns:
        Tuple of (is_valid, video_id, error_message).
        - is_valid: True if the URL is a valid YouTube URL.
        - video_id: The extracted video ID, or None if invalid.
        - error_message: Human-readable error string (empty if valid).
    """
    if not url or not url.strip():
        return False, None, "Please enter a YouTube URL."

    video_id = extract_video_id(url.strip())

    if video_id is None:
        return (
            False,
            None,
            "Invalid YouTube URL. Supported formats: watch?v=, youtu.be/, /shorts/, or bare video ID.",
        )

    return True, video_id, ""
