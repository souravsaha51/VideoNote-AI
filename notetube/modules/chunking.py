"""
modules/chunking.py
Text splitting with timestamp metadata preservation for NoteTube.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Return a configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_transcript(text: str) -> List[Document]:
    """
    Split a plain transcript string into overlapping Document chunks.

    Args:
        text: Clean transcript string.

    Returns:
        List of LangChain Document objects.
    """
    splitter = _get_splitter()
    return splitter.create_documents([text])


def chunk_with_timestamps(segments: List) -> List[Document]:
    """
    Split transcript segments into chunks while preserving start timestamps.

    Each resulting Document carries a 'start' metadata key (seconds, float)
    which can be used to build deep-links into the video.

    Args:
        segments: List of transcript segments.

    Returns:
        List of Documents with metadata={'start': <seconds>}.
    """
    splitter = _get_splitter()
    docs: List[Document] = []
    current_text = ""
    current_start: float = 0.0
    char_count = 0

    for seg in segments:
        seg_text = (seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip()
        seg_start = float(seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0))

        if not seg_text:
            continue

        # Track the start time of the first segment in each chunk
        if char_count == 0:
            current_start = seg_start

        current_text += " " + seg_text
        char_count += len(seg_text)

        # When we've accumulated enough text, create a Document
        if char_count >= CHUNK_SIZE:
            doc = Document(
                page_content=current_text.strip(),
                metadata={"start": current_start},
            )
            docs.append(doc)

            # Keep overlapping tail for context continuity
            overlap_text = current_text[-CHUNK_OVERLAP:]
            current_text = overlap_text
            char_count = len(overlap_text)
            current_start = seg_start  # Rough overlap start

    # Flush remaining text as a final chunk
    if current_text.strip():
        docs.append(
            Document(
                page_content=current_text.strip(),
                metadata={"start": current_start},
            )
        )

    return docs
