"""
modules/transcript.py
YouTube transcript extraction with timestamps for NoteTube.
Uses the exact API pattern confirmed in rag_project.ipynb:
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=["en"])
    text = " ".join(snippet.text for snippet in fetched)
"""

from typing import Tuple, List, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from utils.helpers import clean_transcript_text


class TranscriptError(Exception):
    """Raised when transcript extraction fails for any reason."""
    pass


def get_transcript(
    video_id: str,
    language: str = "en",
) -> Tuple[str, List]:
    """
    Fetch the transcript for a YouTube video.

    Mirrors rag_project.ipynb pattern:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en"])
        transcript = " ".join(snippet.text for snippet in fetched)

    Args:
        video_id: 11-character YouTube video ID.
        language: Preferred transcript language code (e.g. 'en', 'hi').

    Returns:
        Tuple of (clean_text: str, segments: list of FetchedTranscriptSnippet).

    Raises:
        TranscriptError: On any failure with a user-friendly message.
    """
    try:
        api = YouTubeTranscriptApi()

        # Try preferred language first, then fall back to any available
        try:
            segments = api.fetch(video_id, languages=[language])
        except NoTranscriptFound:
            # Fallback: list all available and pick any
            transcript_list = api.list(video_id)
            try:
                transcript = transcript_list.find_transcript([language, "en"])
            except Exception:
                transcript = next(iter(transcript_list))
            segments = transcript.fetch()

        # Flatten to plain text — same as notebook: " ".join(snippet.text for snippet in fetched)
        raw_text = " ".join(snippet.text for snippet in segments)
        clean_text = clean_transcript_text(raw_text)

        return clean_text, segments

    except TranscriptsDisabled:
        raise TranscriptError(
            "Transcripts are disabled for this video. "
            "The creator has turned off captions."
        )
    except VideoUnavailable:
        raise TranscriptError(
            "This video is unavailable. It may be private, age-restricted, "
            "or deleted."
        )
    except NoTranscriptFound:
        raise TranscriptError(
            f"No transcript found for language '{language}'. "
            "Try selecting a different language in the sidebar."
        )
    except TranscriptError:
        raise
    except Exception as e:
        raise TranscriptError(
            f"Failed to fetch transcript: {str(e)}"
        )


def get_available_languages(video_id: str) -> List[Dict[str, str]]:
    """
    List all available transcript languages for a video.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        List of {'code': str, 'name': str, 'is_generated': bool} dicts.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        languages = []
        for t in transcript_list:
            languages.append({
                "code": t.language_code,
                "name": t.language,
                "is_generated": t.is_generated,
            })
        return languages
    except Exception:
        return [{"code": "en", "name": "English", "is_generated": True}]
