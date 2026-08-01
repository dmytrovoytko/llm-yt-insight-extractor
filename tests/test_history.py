"""Tests for local history persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from core.history import (
    HistoryEntry,
    HistoryStore,
    load_history,
    save_to_history,
)
from core.prompts import ActionableIdea, ActionableIdeas, Subtopic, Subtopics

TEST_VIDEO_ID = "TrvLEgPpV8s"


class BaseHistoryTestCase(unittest.TestCase):
    """Base test case with common setup/teardown and helper methods."""

    def setUp(self):
        """Create temporary history file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files."""
        if self.temp_path.exists():
            self.temp_path.unlink()

    def get_sample_subtopics(self) -> Subtopics:
        """Create sample subtopics for testing."""
        return Subtopics(
            subtopics=[
                Subtopic(
                    title="Introduction to the Topic",
                    timestamp="[00:30]",
                    summary="This is the opening segment discussing the main topic.",
                ),
                Subtopic(
                    title="Key Insights",
                    timestamp="[05:15]",
                    summary="A discussion of important takeaways and insights.",
                ),
            ]
        )

    def get_sample_actionable_ideas(self) -> ActionableIdeas:
        """Create sample actionable ideas for testing."""
        return ActionableIdeas(
            ideas=[
                ActionableIdea(
                    title="Start daily meditation practice",
                    description="Allocate 10 minutes every morning for meditation. Begin with guided videos. Track consistency.",
                    timestamp="[12:45]",
                ),
                ActionableIdea(
                    title="Create a structured routine",
                    description="Establish consistent wake and sleep times. Plan daily activities. Use a calendar app.",
                    timestamp="[18:30]",
                ),
                ActionableIdea(
                    title="Join a community group",
                    description="Find local or online groups with shared interests. Attend weekly meetings. Build relationships.",
                    timestamp="[24:10]",
                ),
                ActionableIdea(
                    title="Track progress weekly",
                    description="Keep a journal of accomplishments and challenges. Review every Sunday. Adjust as needed.",
                    timestamp="[31:05]",
                ),
                ActionableIdea(
                    title="Find an accountability partner",
                    description="Identify someone with similar goals. Check in regularly. Share your progress.",
                    timestamp="[37:20]",
                ),
            ]
        )


class TestHistoryStoreBasics(BaseHistoryTestCase):
    """Test basic HistoryStore creation and operations."""

    def test_history_store_creation(self):
        """Test creating a history store."""
        store = HistoryStore(self.temp_path)
        self.assertEqual(store.history_file, self.temp_path)

    def test_save_single_entry(self):
        """Test saving a single analysis result."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        store.save_entry(
            video_url=f"https://youtu.be/{TEST_VIDEO_ID}",
            video_title="Test video title",
            area_of_life="Career",
            goal="Improve productivity",
            subtopics=subtopics,
            actionable_ideas=ideas,
            timestamp="2026-07-31T10:30:00",
        )

        # Verify file was created
        self.assertTrue(self.temp_path.exists())

        # Verify content
        with open(self.temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(
            data[0]["video_url"], f"https://youtu.be/{TEST_VIDEO_ID}"
        )
        self.assertEqual(data[0]["area_of_life"], "Career")
        self.assertEqual(data[0]["goal"], "Improve productivity")
        self.assertEqual(data[0]["timestamp"], "2026-07-31T10:30:00")

    def test_load_empty_history(self):
        """Test loading history when file doesn't exist."""
        store = HistoryStore(self.temp_path)
        history = store.load_history()
        self.assertEqual(history, [])

    def test_load_existing_history(self):
        """Test loading existing history entries."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        # Save multiple entries
        store.save_entry(
            video_url="https://youtu.be/video1",
            video_title="Video 1 title",
            area_of_life="Fitness",
            goal="Get stronger",
            subtopics=subtopics,
            actionable_ideas=ideas,
        )
        store.save_entry(
            video_url="https://youtu.be/video2",
            video_title="Video 2 title",
            area_of_life="Finance",
            goal="Save more money",
            subtopics=subtopics,
            actionable_ideas=ideas,
        )

        # Load history
        history = store.load_history()
        self.assertEqual(len(history), 2)
        self.assertTrue(
            all(isinstance(entry, HistoryEntry) for entry in history)
        )
        self.assertEqual(history[0].video_url, "https://youtu.be/video1")
        self.assertEqual(history[1].video_url, "https://youtu.be/video2")


class TestHistoryAppend(BaseHistoryTestCase):
    """Test appending to existing history."""

    def test_append_to_existing_history(self):
        """Test appending to existing history without overwriting."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        # Save first entry
        store.save_entry(
            video_url="https://youtu.be/video1",
            video_title="Video 1 title",
            area_of_life="Career",
            goal="Goal 1",
            subtopics=subtopics,
            actionable_ideas=ideas,
        )

        # Verify first entry
        history = store.load_history()
        self.assertEqual(len(history), 1)

        # Save second entry
        store.save_entry(
            video_url="https://youtu.be/video2",
            video_title="Video 2 title",
            area_of_life="Health",
            goal="Goal 2",
            subtopics=subtopics,
            actionable_ideas=ideas,
        )

        # Verify both entries exist
        history = store.load_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].video_url, "https://youtu.be/video1")
        self.assertEqual(history[1].video_url, "https://youtu.be/video2")


class TestHistoryValidation(BaseHistoryTestCase):
    """Test input validation in history saving."""

    def test_save_entry_invalid_url_empty(self):
        """Test saving with empty video URL."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        with self.assertRaisesRegex(
            ValueError, "video_url must be a non-empty string"
        ):
            store.save_entry(
                video_url="",
                video_title="",
                area_of_life="Career",
                goal="Goal",
                subtopics=subtopics,
                actionable_ideas=ideas,
            )

    def test_save_entry_invalid_url_none(self):
        """Test saving with None video URL."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        with self.assertRaisesRegex(
            ValueError, "video_url must be a non-empty string"
        ):
            store.save_entry(
                video_url=None,
                video_title="",
                area_of_life="Career",
                goal="Goal",
                subtopics=subtopics,
                actionable_ideas=ideas,
            )

    def test_save_entry_invalid_area_of_life_empty(self):
        """Test saving with empty area of life."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        with self.assertRaisesRegex(
            ValueError, "area_of_life must be a non-empty string"
        ):
            store.save_entry(
                video_url="https://youtu.be/test",
                video_title="",
                area_of_life="",
                goal="Goal",
                subtopics=subtopics,
                actionable_ideas=ideas,
            )

    def test_save_entry_invalid_area_of_life_none(self):
        """Test saving with None area of life."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        with self.assertRaisesRegex(
            ValueError, "area_of_life must be a non-empty string"
        ):
            store.save_entry(
                video_url="https://youtu.be/test",
                video_title="",
                area_of_life=None,
                goal="Goal",
                subtopics=subtopics,
                actionable_ideas=ideas,
            )

    def test_save_entry_invalid_goal_none(self):
        """Test saving with None goal."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        with self.assertRaisesRegex(ValueError, "goal must be a string"):
            store.save_entry(
                video_url="https://youtu.be/test",
                video_title="",
                area_of_life="Career",
                goal=None,
                subtopics=subtopics,
                actionable_ideas=ideas,
            )


class TestHistoryClear(BaseHistoryTestCase):
    """Test clearing history."""

    def test_clear_history(self):
        """Test clearing history."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        # Save entries
        store.save_entry(
            video_url="https://youtu.be/video1",
            video_title="Video 1 title",
            area_of_life="Career",
            goal="Goal",
            subtopics=subtopics,
            actionable_ideas=ideas,
        )
        self.assertTrue(self.temp_path.exists())

        # Clear history
        store.clear_history()
        self.assertFalse(self.temp_path.exists())

        # Verify history is empty
        history = store.load_history()
        self.assertEqual(history, [])


class TestConvenienceFunctions(BaseHistoryTestCase):
    """Test module-level convenience functions."""

    def test_convenience_functions(self):
        """Test convenience functions for default store."""
        # Use the temporary file as the default
        from core import history as history_module

        history_module._default_store = None
        history_module.get_history_store(self.temp_path)

        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        # Save using convenience function
        save_to_history(
            video_url="https://youtu.be/test",
            video_title="Test video title",
            area_of_life="Career",
            goal="Goal",
            subtopics=subtopics,
            actionable_ideas=ideas,
        )

        # Load using convenience function
        loaded = load_history()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].video_url, "https://youtu.be/test")


class TestHistoryEntry(BaseHistoryTestCase):
    """Test HistoryEntry Pydantic model."""

    def test_history_entry_pydantic_model(self):
        """Test HistoryEntry Pydantic model."""
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        entry = HistoryEntry(
            video_url="https://youtu.be/test",
            area_of_life="Career",
            goal="Goal",
            subtopics=subtopics,
            actionable_ideas=ideas,
            timestamp="2026-07-31T10:30:00",
        )

        # Test model_dump
        data = entry.model_dump()
        self.assertEqual(data["video_url"], "https://youtu.be/test")
        self.assertEqual(data["area_of_life"], "Career")

        # Test model_dump_json
        json_str = entry.model_dump_json()
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["video_url"], "https://youtu.be/test")

    def test_history_with_empty_goal(self):
        """Test saving entry with empty goal string."""
        store = HistoryStore(self.temp_path)
        subtopics = self.get_sample_subtopics()
        ideas = self.get_sample_actionable_ideas()

        # Empty goal should be allowed
        store.save_entry(
            video_url="https://youtu.be/test",
            video_title="Test video title",
            area_of_life="Career",
            goal="",  # Empty is OK for optional goals
            subtopics=subtopics,
            actionable_ideas=ideas,
        )

        history = store.load_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].goal, "")


if __name__ == "__main__":
    unittest.main()
