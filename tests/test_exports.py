"""Tests for clickable YouTube timestamp post-processing."""

import json
import unittest

from core.exports import (
    convert_timestamps_to_youtube_links,
    export_actionable_ideas_to_markdown,
    export_subtopics_to_markdown,
    export_to_json,
    export_to_markdown,
    timestamp_to_seconds,
)
from core.prompts import (
    ActionableIdea,
    ActionableIdeas,
    Subtopic,
    Subtopics,
)

TEST_VIDEO_ID = "TrvLEgPpV8s"


class TestExportsTimestampConversion(unittest.TestCase):
    def test_convert_mmss_timestamp_to_youtube_link(self):
        text = "This happens at [01:23] in the video."
        output = convert_timestamps_to_youtube_links(text, TEST_VIDEO_ID)
        self.assertIn(f"[01:23](https://youtu.be/{TEST_VIDEO_ID}?t=83)", output)

    def test_convert_hhmmss_timestamp_to_youtube_link(self):
        text = "A longer segment starts at [01:02:03]."
        output = convert_timestamps_to_youtube_links(text, TEST_VIDEO_ID)
        self.assertIn(
            f"[01:02:03](https://youtu.be/{TEST_VIDEO_ID}?t=3723)", output
        )

    def test_convert_zero_timestamp_to_zero_seconds(self):
        text = "The video opens at [00:00]."
        output = convert_timestamps_to_youtube_links(text, TEST_VIDEO_ID)
        self.assertIn(f"[00:00](https://youtu.be/{TEST_VIDEO_ID}?t=0)", output)

    def test_multiple_timestamps_are_converted(self):
        text = "Start [00:10], then [00:20], and finish at [00:30]."
        output = convert_timestamps_to_youtube_links(text, TEST_VIDEO_ID)
        self.assertEqual(
            output.count(f"https://youtu.be/{TEST_VIDEO_ID}?t="), 3
        )

    def test_missing_brackets_not_converted(self):
        text = "Watch minute 01:23 for the key point."
        output = convert_timestamps_to_youtube_links(text, TEST_VIDEO_ID)
        self.assertEqual(output, text)

    def test_timestamp_to_seconds_parses_mmss(self):
        self.assertEqual(timestamp_to_seconds("[58:12]"), 3492)

    def test_timestamp_to_seconds_parses_hhmmss(self):
        self.assertEqual(timestamp_to_seconds("[01:00:00]"), 3600)

    def test_convert_raises_on_empty_video_id(self):
        with self.assertRaises(ValueError):
            convert_timestamps_to_youtube_links("[00:05] example", "")

    def test_convert_raises_on_non_string_text(self):
        with self.assertRaises(TypeError):
            convert_timestamps_to_youtube_links(None, TEST_VIDEO_ID)


class TestExportsMarkdownGeneration(unittest.TestCase):
    """Test markdown export functions."""

    def setUp(self):
        """Create sample subtopics and ideas for testing."""
        self.subtopics = Subtopics(
            subtopics=[
                Subtopic(
                    title="Introduction",
                    timestamp="[00:30]",
                    summary="Opening segment.",
                ),
                Subtopic(
                    title="Key Points",
                    timestamp="[05:15]",
                    summary="Important takeaways.",
                ),
            ]
        )

        self.ideas = ActionableIdeas(
            ideas=[
                ActionableIdea(
                    title="Start meditation",
                    description="Daily practice.",
                    timestamp="[12:45]",
                ),
                ActionableIdea(
                    title="Build routine",
                    description="Consistent schedule.",
                    timestamp="[18:30]",
                ),
                ActionableIdea(
                    title="Join group",
                    description="Community support.",
                    timestamp="[24:10]",
                ),
                ActionableIdea(
                    title="Track progress",
                    description="Weekly review.",
                    timestamp="[31:05]",
                ),
                ActionableIdea(
                    title="Find partner",
                    description="Accountability.",
                    timestamp="[37:20]",
                ),
            ]
        )

    def test_export_subtopics_to_markdown(self):
        """Test exporting subtopics to markdown format."""
        md = export_subtopics_to_markdown(self.subtopics, TEST_VIDEO_ID)

        self.assertIn("# Subtopics Summary", md)
        self.assertIn("## Introduction", md)
        self.assertIn("## Key Points", md)
        self.assertIn("Opening segment.", md)
        self.assertIn("Important takeaways.", md)
        # Check for clickable links
        self.assertIn(f"https://youtu.be/{TEST_VIDEO_ID}?t=", md)

    def test_export_actionable_ideas_to_markdown(self):
        """Test exporting actionable ideas to markdown format."""
        md = export_actionable_ideas_to_markdown(self.ideas, TEST_VIDEO_ID)

        self.assertIn("# Top 5 Actionable Ideas", md)
        self.assertIn("## 1. Start meditation", md)
        self.assertIn("## 5. Find partner", md)
        self.assertIn("Daily practice.", md)
        # Check for clickable links
        self.assertIn(f"https://youtu.be/{TEST_VIDEO_ID}?t=", md)
        # Should have 5 links
        self.assertEqual(md.count(f"https://youtu.be/{TEST_VIDEO_ID}?t="), 5)

    def test_export_to_markdown_combined(self):
        """Test combined markdown export."""
        md = export_to_markdown(
            self.subtopics,
            self.ideas,
            TEST_VIDEO_ID,
            video_url="https://youtu.be/test",
            area_of_life="Career",
        )

        # Check for metadata
        self.assertIn("# YT Insight Extractor - Analysis Results", md)
        self.assertIn("Generated on", md)
        self.assertIn(f"https://youtu.be/test", md)
        self.assertIn("Career", md)

        # Check for both sections
        self.assertIn("# Subtopics Summary", md)
        self.assertIn("# Top 5 Actionable Ideas", md)
        self.assertIn("## Introduction", md)
        self.assertIn("## 1. Start meditation", md)


class TestExportsJsonGeneration(unittest.TestCase):
    """Test JSON export functions."""

    def setUp(self):
        """Create sample subtopics and ideas for testing."""
        self.subtopics = Subtopics(
            subtopics=[
                Subtopic(
                    title="Introduction",
                    timestamp="[00:30]",
                    summary="Opening segment.",
                ),
            ]
        )

        self.ideas = ActionableIdeas(
            ideas=[
                ActionableIdea(
                    title="Start meditation",
                    description="Daily practice.",
                    timestamp="[12:45]",
                ),
                ActionableIdea(
                    title="Build routine",
                    description="Consistent schedule.",
                    timestamp="[18:30]",
                ),
                ActionableIdea(
                    title="Join group",
                    description="Community support.",
                    timestamp="[24:10]",
                ),
                ActionableIdea(
                    title="Track progress",
                    description="Weekly review.",
                    timestamp="[31:05]",
                ),
                ActionableIdea(
                    title="Find partner",
                    description="Accountability.",
                    timestamp="[37:20]",
                ),
            ]
        )

    def test_export_to_json_pretty(self):
        """Test pretty JSON export."""
        json_str = export_to_json(
            self.subtopics,
            self.ideas,
            video_url="https://youtu.be/test",
            area_of_life="Career",
            goal="Improve focus",
            pretty=True,
        )

        # Verify it's valid JSON
        data = json.loads(json_str)

        # Check structure
        self.assertIn("metadata", data)
        self.assertIn("subtopics", data)
        self.assertIn("actionable_ideas", data)

        # Check metadata
        self.assertEqual(data["metadata"]["video_url"], "https://youtu.be/test")
        self.assertEqual(data["metadata"]["area_of_life"], "Career")
        self.assertEqual(data["metadata"]["goal"], "Improve focus")

        # Check content
        self.assertEqual(len(data["subtopics"]["subtopics"]), 1)
        self.assertEqual(len(data["actionable_ideas"]["ideas"]), 5)

    def test_export_to_json_compact(self):
        """Test compact JSON export."""
        json_str = export_to_json(
            self.subtopics,
            self.ideas,
            pretty=False,
        )

        # Verify it's valid JSON
        data = json.loads(json_str)
        self.assertIn("metadata", data)

        # Verify it's compact (no pretty printing)
        # Compact JSON shouldn't have much whitespace
        self.assertNotIn("\n  ", json_str)

    def test_export_to_json_preserves_data(self):
        """Test that JSON export preserves all data correctly."""
        json_str = export_to_json(
            self.subtopics,
            self.ideas,
            video_url="https://youtu.be/test",
        )

        data = json.loads(json_str)

        # Verify subtopic content
        subtopic = data["subtopics"]["subtopics"][0]
        self.assertEqual(subtopic["title"], "Introduction")
        self.assertEqual(subtopic["timestamp"], "[00:30]")
        self.assertEqual(subtopic["summary"], "Opening segment.")

        # Verify ideas
        ideas = data["actionable_ideas"]["ideas"]
        self.assertEqual(ideas[0]["title"], "Start meditation")
        self.assertEqual(len(ideas), 5)


if __name__ == "__main__":
    unittest.main()
