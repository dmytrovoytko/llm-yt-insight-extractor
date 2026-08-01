from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from pytubefix import YouTube

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

VALIDATE_DURATION = True
DURATION_TRESHOLD = 60 * 60  # 60 minutes

USE_TRANSCRIPT_CACHE = True
TRANSCRIPT_CACHE_DIR = Path("data") / ".transcript_cache"
TRANSCRIPT_CACHE_EXTENSION = ".txt"


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

def extract_video_title(youtube_url: str) -> str:
    video_id = extract_video_id(youtube_url) # if it errors, no sense for title
    try:
        yt = YouTube(youtube_url) # , use_po_token=True # try when Youtube detects requests as "bot"
        return yt.title
    except Exception as e:
        print("! extract_video_title error:", e)

    return video_id

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
        raise TranscriptError(
            f"For the MVP, videos must be under {DURATION_TRESHOLD//60} minutes."
        )

    return total_duration


def _get_transcript_cache_path(
    video_id: str, cache_dir: Path | str = TRANSCRIPT_CACHE_DIR
) -> Path:
    return Path(cache_dir) / f"{video_id}{TRANSCRIPT_CACHE_EXTENSION}"


def load_transcript_cache(
    video_id: str, cache_dir: Path | str = TRANSCRIPT_CACHE_DIR
) -> list[dict] | None:
    cache_path = _get_transcript_cache_path(video_id, cache_dir)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, list):
        return None

    formatted = []
    for entry in data:
        formatted.append(
            {
                "text": str(entry["text"]).strip(),
                "start": float(entry["start"]),
                "duration": float(entry["duration"]),
            }
        )

    return formatted


def save_transcript_cache(
    video_id: str,
    transcript: list[dict],
    cache_dir: Path | str = TRANSCRIPT_CACHE_DIR,
) -> Path:
    cache_path = _get_transcript_cache_path(video_id, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(transcript, fp, indent=2)

    return cache_path


def fetch_youtube_transcript(
    youtube_url: str,
    languages: Iterable[str] | None = None,
    use_cache: bool = USE_TRANSCRIPT_CACHE,
    cache_dir: Path | str = TRANSCRIPT_CACHE_DIR,
) -> list[dict]:
    video_id = extract_video_id(youtube_url)

    if use_cache:
        cached = load_transcript_cache(video_id, cache_dir)
        if cached is not None:
            if VALIDATE_DURATION:
                validate_transcript_duration(cached)
            return cached

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

    if use_cache:
        save_transcript_cache(video_id, formatted, cache_dir)

    return formatted


if __name__ == "__main__":
    # quick transcript extraction test
    url = "https://www.youtube.com/watch?v=TrvLEgPpV8s"  # Productivity Tips From Tim Ferriss, <7min
    video_id = extract_video_id(url)
    print(f"{video_id=}")
    transcript_data = fetch_youtube_transcript(url)
    print(f"{transcript_data=}")
