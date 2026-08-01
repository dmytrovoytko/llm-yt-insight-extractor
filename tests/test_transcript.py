import tempfile
import unittest
from unittest.mock import MagicMock, patch

from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from core.transcript import (
    TranscriptError,
    extract_video_title,
    fetch_youtube_transcript,
    load_transcript_cache,
    save_transcript_cache,
    validate_transcript_duration,
)


class TestTranscriptExtraction(unittest.TestCase):
    url_no_transcript_error = "https://www.youtube.com/watch?v=ru4hdcMmlwQ"  # 10 Minute Meditation Music • Pure Waves
    url_duration_rejects_over_60_minutes = "https://www.youtube.com/watch?v=8KPLs-ZFuPo"  # How to Embrace Slow Productivity, Achieve Mastery, and Defend Your Time — Cal Newport & Tim Ferriss
    video_id_valid = "TrvLEgPpV8s"  # Productivity Tips From Tim Ferriss, <7min
    url_duration_passes_within_limit = (
        f"https://www.youtube.com/watch?v={video_id_valid}"
    )

    def test_fetch_youtube_transcript_returns_formatted_entries(self):
        sample_transcript = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "This is a test.", "start": 2.0, "duration": 3.0},
        ]

        with patch(
            "core.transcript.YouTubeTranscriptApi.fetch",
            return_value=sample_transcript,
        ):
            result = fetch_youtube_transcript(
                self.url_duration_passes_within_limit, use_cache=False
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Hello world")
        self.assertEqual(result[1]["start"], 2.0)
        self.assertEqual(result[1]["duration"], 3.0)

    def test_fetch_youtube_transcript_no_transcript_error(self):
        for error in (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            with self.subTest(error=error):
                with patch(
                    "core.transcript.YouTubeTranscriptApi.fetch",
                    side_effect=error,
                ):
                    with self.assertRaises(TranscriptError) as context:
                        fetch_youtube_transcript(self.url_no_transcript_error)

                self.assertEqual(
                    str(context.exception),
                    "Could not retrieve transcript for this video. It may not have subtitles available.",
                )

    def test_extract_video_title_returns_title(self):
        mock_yt = MagicMock()
        mock_yt.title = "Mock Video Title"

        with patch("core.transcript.YouTube", return_value=mock_yt):
            title = extract_video_title(self.url_duration_passes_within_limit)

        self.assertEqual(title, "Mock Video Title")

    def test_extract_video_title_falls_back_to_video_id(self):
        with patch(
            "core.transcript.YouTube",
            side_effect=Exception("No title available"),
        ):
            title = extract_video_title(self.url_duration_passes_within_limit)

        self.assertEqual(title, self.video_id_valid)

    def test_load_and_save_transcript_cache_roundtrip(self):
        sample_transcript = [
            {"text": "One", "start": 0.0, "duration": 1.0},
            {"text": "Two", "start": 1.0, "duration": 2.0},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            save_transcript_cache(
                "video123", sample_transcript, cache_dir=temp_dir
            )
            loaded = load_transcript_cache("video123", cache_dir=temp_dir)

        self.assertEqual(loaded, sample_transcript)

    def test_fetch_youtube_transcript_uses_cache_if_available(self):
        sample_transcript = [
            {"text": "Cached text", "start": 0.0, "duration": 1.5}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            save_transcript_cache(
                self.video_id_valid, sample_transcript, cache_dir=temp_dir
            )
            with patch(
                "core.transcript.YouTubeTranscriptApi.fetch",
                side_effect=AssertionError(
                    "Remote API should not be called when cache exists"
                ),
            ):
                result = fetch_youtube_transcript(
                    self.url_duration_passes_within_limit,
                    cache_dir=temp_dir,
                )

        self.assertEqual(result, sample_transcript)

    def test_validate_transcript_duration_rejects_over_60_minutes(self):
        transcript = [
            {"text": "Start", "start": 0.0, "duration": 1.0},
            {"text": "End", "start": 3601.0, "duration": 1.0},
        ]

        with self.assertRaises(TranscriptError) as context:
            validate_transcript_duration(transcript)

        self.assertEqual(
            str(context.exception),
            "For the MVP, videos must be under 60 minutes.",
        )

    def test_validate_transcript_duration_passes_within_limit(self):
        transcript = [
            {"text": "Start", "start": 0.0, "duration": 1.0},
            {"text": "End", "start": 3598.0, "duration": 1.5},
        ]

        duration = validate_transcript_duration(transcript)
        self.assertAlmostEqual(duration, 3599.5)


if __name__ == "__main__":
    unittest.main()
