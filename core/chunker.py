"""Timestamp-preserving transcript chunker with configurable overlap."""

from __future__ import annotations

import re
from typing import NamedTuple

from core.settings import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS


class TranscriptChunk(NamedTuple):
    """A chunk of transcript text with timestamp metadata."""

    text: str  # Full text with [mm:ss] timestamps prefixed
    start: float  # Start time in seconds (timestamp of first entry)
    end: float  # End time in seconds (timestamp + duration of last entry)
    word_count: int  # Number of words in this chunk


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to [mm:ss] or [hh:mm:ss] format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def _count_words(text: str) -> int:
    """Count words in text by splitting on whitespace."""
    return len(text.split())


def _build_timestamped_text(
    transcript_entries: list[dict],
) -> list[tuple[str, float, float]]:
    """Build list of (timestamped_text, start_time, end_time) tuples.

    Each transcript entry is prefixed with its timestamp.

    Args:
        transcript_entries: List of dicts with 'text', 'start', 'duration'.

    Returns:
        List of (timestamped_text, start_time, end_time) tuples.
    """
    result = []
    for entry in transcript_entries:
        timestamp = _seconds_to_timestamp(entry["start"])
        prefixed_text = f"{timestamp} {entry['text']}"
        start_time = float(entry["start"])
        end_time = start_time + float(entry["duration"])
        result.append((prefixed_text, start_time, end_time))

    return result


def chunk_transcript(
    transcript: list[dict],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TranscriptChunk]:
    """Split transcript into overlapping chunks with timestamp preservation.

    Each chunk contains text prefixed with [mm:ss] timestamps. Chunks overlap
    by the specified number of words to maintain context.

    Args:
        transcript: List of transcript dicts with 'text', 'start', 'duration'.
        chunk_words: Target number of words per chunk.
        overlap_words: Number of words to overlap between consecutive chunks.

    Returns:
        List of TranscriptChunk objects.
    """
    if not transcript:
        return []

    # Build timestamped text entries
    timestamped = _build_timestamped_text(transcript)

    # Flatten all timestamped text into one continuous string
    all_text = " ".join(t[0] for t in timestamped)
    all_words = all_text.split()

    chunks = []
    chunk_start = 0

    # Map word indices to (start_time, end_time) based on timestamped entries
    word_to_time = {}
    word_idx = 0
    for timestamped_text, start_time, end_time in timestamped:
        words_in_entry = timestamped_text.split()
        for _ in words_in_entry:
            word_to_time[word_idx] = (start_time, end_time)
            word_idx += 1

    while chunk_start < len(all_words):
        chunk_end = min(chunk_start + chunk_words, len(all_words))

        # Extract chunk words
        chunk_words_list = all_words[chunk_start:chunk_end]
        chunk_text = " ".join(chunk_words_list)

        # Determine chunk's time span from first and last word
        chunk_start_time = word_to_time.get(chunk_start, (0.0, 0.0))[0]
        chunk_end_time = word_to_time.get(chunk_end - 1, (0.0, 0.0))[1]

        chunk = TranscriptChunk(
            text=chunk_text,
            start=chunk_start_time,
            end=chunk_end_time,
            word_count=len(chunk_words_list),
        )
        chunks.append(chunk)

        # Move start for next iteration (with overlap). Ensure forward progress
        next_start = max(0, chunk_end - overlap_words)
        if next_start <= chunk_start:
            # cannot make forward progress (bad overlap or reached end); stop
            break
        chunk_start = next_start

    return chunks


if __name__ == "__main__":
    # quick transcript extraction test
    from transcript import (
        extract_video_id,
        fetch_youtube_transcript,
        load_transcript_cache,
        save_transcript_cache,
        validate_transcript_duration,
    )

    url = "https://www.youtube.com/watch?v=TrvLEgPpV8s"  # Productivity Tips From Tim Ferriss, <7min
    video_id = extract_video_id(url)
    print(f"{video_id=}")
    transcript_data = fetch_youtube_transcript(url)
    # print(f"{transcript_data=}")
    chunks = chunk_transcript(transcript_data)
    print(f"\n{chunks=}")
