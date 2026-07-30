from __future__ import annotations

from typing import Iterable
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

VALIDATE_DURATION = True
DURATION_TRESHOLD = 60*60 # 60 minutes

class TranscriptError(RuntimeError):
    pass


def extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url.strip())
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/")

    if parsed.netloc in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        query_params = parse_qs(parsed.query)
        if "v" in query_params and query_params["v"]:
            return query_params["v"][0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[-1]
        if parsed.path.startswith("/v/"):
            return parsed.path.split("/v/")[-1]

    raise ValueError(
        "Could not extract YouTube video ID from the provided URL."
    )


def validate_transcript_duration(transcript: list[dict]) -> float:
    if not transcript:
        raise TranscriptError(
            "Could not retrieve transcript for this video. It may not have subtitles available."
        )

    first_start = float(transcript[0]["start"])
    last_entry = transcript[-1]
    total_duration = (
        float(last_entry["start"]) + float(last_entry["duration"]) - first_start
    )

    if total_duration > DURATION_TRESHOLD:
        raise TranscriptError(f"For the MVP, videos must be under {DURATION_TRESHOLD//60} minutes.")

    return total_duration


def fetch_youtube_transcript(
    youtube_url: str, languages: Iterable[str] | None = None
) -> list[dict]:
    video_id = extract_video_id(youtube_url)
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(
            video_id, languages=list(languages) if languages else ["en"]
        )
        if not isinstance(transcript_data, list):
            transcript_data = transcript_data.to_raw_data()
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        raise TranscriptError(
            "Could not retrieve transcript for this video. It may not have subtitles available."
        )
    except Exception as exc:
        raise TranscriptError(
            "Could not retrieve transcript for this video. It may not have subtitles available."
        ) from exc

    formatted = [
        {
            "text": entry["text"].strip(),
            "start": float(entry["start"]),
            "duration": float(entry["duration"]),
        }
        for entry in transcript_data
    ]

    if VALIDATE_DURATION:
        validate_transcript_duration(formatted)

    return formatted

if __name__ == "__main__":
    # quick test
    url = "https://www.youtube.com/watch?v=TrvLEgPpV8s"  # Productivity Tips From Tim Ferriss, <7min
    video_id = extract_video_id(url)
    print(f"{video_id=}")
    transcript_data = fetch_youtube_transcript(url)
    print(f"{transcript_data=}")
