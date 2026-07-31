"""Utility functions for exporting post-processed results."""

import json
import re
from datetime import datetime
from typing import Optional

from core.prompts import ActionableIdeas, Subtopics

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


def export_subtopics_to_markdown(
    subtopics: Subtopics, video_id: str
) -> str:
    """Convert Subtopics model to formatted markdown with clickable timestamps.

    Args:
        subtopics: Subtopics Pydantic model
        video_id: YouTube video ID for creating timestamp links

    Returns:
        Formatted markdown string
    """
    md_lines = ["# Subtopics Summary\n"]

    for subtopic in subtopics.subtopics:
        # Convert timestamp to clickable link
        timestamp_with_link = convert_timestamps_to_youtube_links(
            subtopic.timestamp, video_id
        )

        md_lines.append(f"## {subtopic.title}\n")
        md_lines.append(f"**Timestamp:** {timestamp_with_link}\n")
        md_lines.append(f"{subtopic.summary}\n")

    return "\n".join(md_lines)


def export_actionable_ideas_to_markdown(
    ideas: ActionableIdeas, video_id: str
) -> str:
    """Convert ActionableIdeas model to formatted markdown with clickable timestamps.

    Args:
        ideas: ActionableIdeas Pydantic model
        video_id: YouTube video ID for creating timestamp links

    Returns:
        Formatted markdown string
    """
    md_lines = ["# Top 5 Actionable Ideas\n"]

    for idx, idea in enumerate(ideas.ideas, start=1):
        # Convert timestamp to clickable link
        timestamp_with_link = convert_timestamps_to_youtube_links(
            idea.timestamp, video_id
        )

        md_lines.append(f"## {idx}. {idea.title}\n")
        md_lines.append(f"**Description:** {idea.description}\n")
        md_lines.append(f"**Timestamp:** {timestamp_with_link}\n")

    return "\n".join(md_lines)


def export_to_markdown(
    subtopics: Subtopics,
    ideas: ActionableIdeas,
    video_id: str,
    video_url: str = "",
    area_of_life: str = "",
) -> str:
    """Export both subtopics and actionable ideas to a single markdown document.

    Args:
        subtopics: Subtopics Pydantic model
        ideas: ActionableIdeas Pydantic model
        video_id: YouTube video ID for creating timestamp links
        video_url: Original YouTube URL (optional, for reference)
        area_of_life: User's selected area of life (optional, for context)

    Returns:
        Complete markdown document as string
    """
    md_lines = [
        "# YT Insight Extractor - Analysis Results\n",
        f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    ]

    if video_url:
        md_lines.append(f"**Video:** [{video_url}]({video_url})\n")

    if area_of_life:
        md_lines.append(f"**Area of Life:** {area_of_life}\n")

    md_lines.append("\n---\n")

    # Add subtopics section
    md_lines.append(export_subtopics_to_markdown(subtopics, video_id))
    md_lines.append("\n---\n")

    # Add actionable ideas section
    md_lines.append(export_actionable_ideas_to_markdown(ideas, video_id))

    return "\n".join(md_lines)


def export_to_json(
    subtopics: Subtopics,
    ideas: ActionableIdeas,
    video_url: str = "",
    area_of_life: str = "",
    goal: str = "",
    pretty: bool = True,
) -> str:
    """Export subtopics and actionable ideas to JSON format.

    Args:
        subtopics: Subtopics Pydantic model
        ideas: ActionableIdeas Pydantic model
        video_url: Original YouTube URL (optional)
        area_of_life: User's selected area of life (optional)
        goal: User's specific goal (optional)
        pretty: Whether to pretty-print JSON with indentation

    Returns:
        JSON string
    """
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "video_url": video_url,
            "area_of_life": area_of_life,
            "goal": goal,
        },
        "subtopics": json.loads(subtopics.model_dump_json()),
        "actionable_ideas": json.loads(ideas.model_dump_json()),
    }

    if pretty:
        return json.dumps(output, indent=2, ensure_ascii=False)
    else:
        return json.dumps(output, ensure_ascii=False)
