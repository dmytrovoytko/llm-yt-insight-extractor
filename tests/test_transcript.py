import unittest
from unittest.mock import patch

from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from core.transcript import (
    TranscriptError,
    fetch_youtube_transcript,
    validate_transcript_duration,
)


class TestTranscriptExtraction(unittest.TestCase):
    url_no_transcript_error = "https://www.youtube.com/watch?v=ru4hdcMmlwQ"  # 10 Minute Meditation Music • Pure Waves
    url_duration_rejects_over_60_minutes = "https://www.youtube.com/watch?v=8KPLs-ZFuPo"  # How to Embrace Slow Productivity, Achieve Mastery, and Defend Your Time — Cal Newport & Tim Ferriss
    url_duration_passes_within_limit = "https://www.youtube.com/watch?v=TrvLEgPpV8s"  # Productivity Tips From Tim Ferriss

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
                self.url_duration_passes_within_limit
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
