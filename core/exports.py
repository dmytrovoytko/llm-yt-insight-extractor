"""Utility functions for exporting post-processed results."""

import re

TIMESTAMP_PATTERN = re.compile(r"\[(\d{2}):(\d{2})(?::(\d{2}))?\]")


def timestamp_to_seconds(timestamp: str) -> int:
    """Convert a [mm:ss] or [hh:mm:ss] timestamp to total seconds."""
    match = TIMESTAMP_PATTERN.fullmatch(timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    hour_or_minute = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)

    if match.group(3) is None:
        return (
            minute * 60 + second
            if hour_or_minute == 0
            else hour_or_minute * 60 + second
        )
    return hour_or_minute * 3600 + minute * 60 + second


def convert_timestamps_to_youtube_links(text: str, video_id: str) -> str:
    """Replace bracketed timestamps with clickable YouTube links."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    if not video_id or not isinstance(video_id, str):
        raise ValueError("Video ID must be a non-empty string.")

    def _replace(match: re.Match[str]) -> str:
        original = match.group(0)
        seconds = timestamp_to_seconds(original)
        url = f"https://youtu.be/{video_id}?t={seconds}"
        return f"[{original[1:-1]}]({url})"

    return TIMESTAMP_PATTERN.sub(_replace, text)
