"""Local history persistence for analysis results."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from core.prompts import ActionableIdeas, Subtopics


class HistoryEntry(BaseModel):
    """A single entry in the analysis history."""

    video_url: str = Field(..., description="YouTube video URL")
    video_title: str = Field(
        "",
        description="Title of the YouTube video",
    )
    area_of_life: str = Field(..., description="User's selected area of life")
    goal: str = Field(..., description="User's specific goal (optional)")
    subtopics: Subtopics = Field(..., description="Extracted subtopics")
    actionable_ideas: ActionableIdeas = Field(
        ..., description="Extracted actionable ideas"
    )
    timestamp: str = Field(
        ..., description="ISO format timestamp of when analysis was completed"
    )
    llm_info: str = Field(
        ..., description="LLM provider: model"
    )

class HistoryStore:
    """Manages persistence of analysis history to a local JSON file."""

    def __init__(self, history_file: Optional[Path] = None):
        """Initialize the history store.

        Args:
            history_file: Path to the history.json file. Defaults to data/history.json.
        """
        if history_file is None:
            history_file = (
                Path(__file__).parent.parent / "data" / "history.json"
            )
        self.history_file = history_file
        # print(f"{history_file=}") # debug

    def save_entry(
        self,
        video_url: str,
        video_title: str,
        area_of_life: str,
        goal: str,
        subtopics: Subtopics,
        actionable_ideas: ActionableIdeas,
        timestamp: Optional[str] = None,
        llm_info: str = "",
    ) -> None:
        """Save an analysis result to history.

        Args:
            video_url: YouTube video URL
            area_of_life: User's selected area of life
            goal: User's specific goal
            subtopics: Extracted subtopics
            actionable_ideas: Extracted actionable ideas
            timestamp: ISO format timestamp (default: current time)

        Raises:
            ValueError: If inputs are invalid
        """
        if not video_url or not isinstance(video_url, str):
            raise ValueError("video_url must be a non-empty string")
        if not area_of_life or not isinstance(area_of_life, str):
            raise ValueError("area_of_life must be a non-empty string")
        if not isinstance(goal, str):
            raise ValueError("goal must be a string")

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entry = HistoryEntry(
            video_url=video_url,
            video_title=video_title,
            area_of_life=area_of_life,
            goal=goal,
            subtopics=subtopics,
            actionable_ideas=actionable_ideas,
            timestamp=timestamp,
            llm_info=llm_info,
        )

        # Ensure directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history or initialize empty list
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
        except Exception as e:
            print(f"! History {self.history_file} load error:", e)
            history = []

        # Append new entry
        history.append(json.loads(entry.model_dump_json()))

        # Write back to file
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def load_history(self) -> List[HistoryEntry]:
        """Load all entries from history.

        Returns:
            List of HistoryEntry objects. Empty list if file doesn't exist.

        Raises:
            json.JSONDecodeError: If history.json is malformed
        """
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"! History {self.history_file} load error:", e)
            data = []        

        if not isinstance(data, list):
            raise ValueError("History file must contain a JSON array")

        return [HistoryEntry.model_validate(entry) for entry in data]

    def clear_history(self) -> None:
        """Clear all history entries (useful for testing)."""
        if self.history_file.exists():
            self.history_file.unlink()


# Convenience functions for default store
_default_store: Optional[HistoryStore] = None


def get_history_store(history_file: Optional[Path] = None) -> HistoryStore:
    """Get or create the default history store.

    Args:
        history_file: Path to custom history.json file (optional)

    Returns:
        HistoryStore instance
    """
    global _default_store
    if _default_store is None or history_file is not None:
        _default_store = HistoryStore(history_file)
    return _default_store


def save_to_history(
    video_url: str,
    video_title: str,
    area_of_life: str,
    goal: str,
    subtopics: Subtopics,
    actionable_ideas: ActionableIdeas,
    timestamp: Optional[str] = None,
    llm_info: str = "Ollama",
) -> None:
    """Convenience function to save an analysis result using the default store.

    Args:
        video_url: YouTube video URL
        area_of_life: User's selected area of life
        goal: User's specific goal
        subtopics: Extracted subtopics
        actionable_ideas: Extracted actionable ideas
        timestamp: ISO format timestamp (default: current time)
    """
    store = get_history_store()
    store.save_entry(
        video_url,
        video_title,
        area_of_life,
        goal,
        subtopics,
        actionable_ideas,
        timestamp,
        llm_info,
    )


def load_history() -> List[HistoryEntry]:
    """Convenience function to load history using the default store.

    Returns:
        List of HistoryEntry objects
    """
    store = get_history_store()
    return store.load_history()
