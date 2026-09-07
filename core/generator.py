"""LLM generation pipeline for structured output (subtopics and actionable ideas)."""

from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from core.llm_config import BaseLLMConfig, LLMConfigError, create_llm
from core.prompts import (
    ActionableIdea,
    ActionableIdeas,
    Subtopic,
    Subtopics,
    build_actionable_ideas_prompt,
    build_subtopics_prompt,
)
from core.settings import GENERATOR_TEST_DEBUG, DEBUG

class GenerationError(RuntimeError):
    """Exception raised when LLM generation fails."""

    pass


FIELD_MAX_LENGTHS = {
    "title": 50,
    "summary": 300,
    "description": 300,
    "timestamp": 20,
}


def _truncate_string(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3].rstrip() + "..."


def _sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        sanitized = {}
        for key, value in payload.items():
            if isinstance(value, str) and key in FIELD_MAX_LENGTHS:
                sanitized[key] = _truncate_string(
                    value.strip(), FIELD_MAX_LENGTHS[key]
                )
            else:
                sanitized[key] = _sanitize_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [_sanitize_payload(item) for item in payload]
    return payload


def _find_json_block(text: str) -> str:
    start_chars = ["{", "["]
    for start_char in start_chars:
        start_idx = text.find(start_char)
        while start_idx != -1:
            depth = 0
            escape = False
            for idx in range(start_idx, len(text)):
                char = text[idx]
                if char == "\\" and not escape:
                    escape = True
                    continue
                if not escape:
                    if char == start_char:
                        depth += 1
                    elif char == ("}" if start_char == "{" else "]"):
                        depth -= 1
                        if depth == 0:
                            return text[start_idx : idx + 1]
                escape = False
            start_idx = text.find(start_char, start_idx + 1)
    raise ValueError("No JSON object or array found in LLM output.")


def parse_structured_output(
    raw_output: str, model_cls: Type[BaseModel]
) -> BaseModel:
    """Extract JSON from raw LLM output and parse it into a Pydantic model."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise GenerationError("LLM output is empty or not a valid string.")

    json_text = raw_output.strip()
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            json_text = _find_json_block(raw_output)
            payload = json.loads(json_text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise GenerationError(
                f"Unable to parse JSON from LLM output: {str(exc)}"
            )

    try:
        return model_cls.model_validate(payload)
    except ValidationError:
        sanitized = _sanitize_payload(payload)
        try:
            return model_cls.model_validate(sanitized)
        except ValidationError as exc:
            raise GenerationError(
                f"LLM output could not be validated against schema: {exc}"
            )


def parse_subtopics_output(raw_output: str) -> Subtopics:
    return parse_structured_output(raw_output, Subtopics)


def parse_actionable_ideas_output(raw_output: str) -> ActionableIdeas:
    return parse_structured_output(raw_output, ActionableIdeas)


def format_context_for_llm(retrieved_chunks: list[dict]) -> str:
    """Format retrieved RAG chunks into a coherent context string for the LLM.

    Args:
        retrieved_chunks: List of dicts with 'text', 'start_time', 'end_time', 'score'.

    Returns:
        Formatted context string.
    """
    if not retrieved_chunks:
        return ""

    formatted_parts = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        score = chunk.get("score", 0.0)
        formatted_parts.append(f"[Relevance: {score:.2f}]\n{text}")

    return "\n\n---\n\n".join(formatted_parts)


def generate_subtopics(
    area_of_life: str,
    specific_goal: str,
    retrieved_chunks: list[dict],
    llm_config: BaseLLMConfig | None = None,
) -> Subtopics:
    """Generate structured subtopics using the Ollama LLM.

    Args:
        area_of_life: The selected area of life.
        specific_goal: The optional specific goal.
        retrieved_chunks: List of retrieved transcript chunks.
        llm_config: Optional LLMConfig instance (creates new one if not provided).

    Returns:
        Subtopics model with generated subtopics.

    Raises:
        GenerationError: If generation fails.
    """
    if llm_config is None:
        try:
            llm_config = create_llm()
        except LLMConfigError as e:
            raise GenerationError(f"Failed to initialize LLM: {str(e)}")

    # Format context for the prompt
    context_str = format_context_for_llm(retrieved_chunks)

    if not context_str.strip():
        raise GenerationError("No context available for subtopic generation.")

    # Build prompt
    prompt = build_subtopics_prompt(area_of_life, specific_goal, context_str)

    try:
        # Generate structured output
        subtopics = llm_config.structured_complete(prompt, Subtopics)
        if DEBUG:
            print("+structured_complete:", subtopics)
        return subtopics
    except LLMConfigError as e:
        if DEBUG:
            print("!! llm_config.structured_complete subtopics... prompt:", len(prompt), prompt, Subtopics)
            print(" error:", e)
        raise GenerationError(f"LLM failed to generate subtopics: {str(e)}")
    except Exception as e:
        raise GenerationError(
            f"Unexpected error during subtopic generation: {str(e)}"
        )


def generate_actionable_ideas(
    area_of_life: str,
    specific_goal: str,
    retrieved_chunks: list[dict],
    llm_config: BaseLLMConfig | None = None,
) -> ActionableIdeas:
    """Generate structured actionable ideas using the Ollama LLM.

    Args:
        area_of_life: The selected area of life.
        specific_goal: The optional specific goal.
        retrieved_chunks: List of retrieved transcript chunks.
        llm_config: Optional LLMConfig instance (creates new one if not provided).

    Returns:
        ActionableIdeas model with generated ideas.

    Raises:
        GenerationError: If generation fails.
    """
    if llm_config is None:
        try:
            llm_config = create_llm()
        except LLMConfigError as e:
            raise GenerationError(f"Failed to initialize LLM: {str(e)}")

    # Format context for the prompt
    context_str = format_context_for_llm(retrieved_chunks)

    if not context_str.strip():
        raise GenerationError("No context available for ideas generation.")

    # Build prompt
    prompt = build_actionable_ideas_prompt(
        area_of_life, specific_goal, context_str
    )

    try:
        # Generate structured output
        # !!! model must support structured_complete
        ideas = llm_config.structured_complete(prompt, ActionableIdeas)
        if DEBUG:
            print("+structured_complete:", ideas)
        return ideas
    except LLMConfigError as e:
        if DEBUG:
            print("!! llm_config.structured_complete ideas... prompt:", len(prompt), prompt, ActionableIdeas)
            print(" error:", e)
        raise GenerationError(
            f"LLM failed to generate actionable ideas: {str(e)}"
        )
    except Exception as e:
        raise GenerationError(
            f"Unexpected error during ideas generation: {str(e)}"
        )


def _create_debug_outputs(
    area_of_life: str, specific_goal: str
) -> tuple[Subtopics, ActionableIdeas]:
    """Create fixed debug outputs for UI rendering tests."""
    subtopics = Subtopics(
        subtopics=[
            Subtopic(
                title=f"Explore {area_of_life} goals",
                timestamp="[00:05]",
                summary=f"Use the transcript to identify key {area_of_life.lower()} themes.",
            ),
            Subtopic(
                title=f"Clarify {specific_goal}",
                timestamp="[01:20]",
                summary="Define a clear, actionable objective from the discussion.",
            ),
            Subtopic(
                title="Turn insights into priorities",
                timestamp="[02:45]",
                summary="Organize the most important points into a short, prioritized list.",
            ),
        ]
    )

    ideas = ActionableIdeas(
        ideas=[
            ActionableIdea(
                title="Summarize main takeaways",
                description="Write down the top three actionable ideas from the transcript.",
                timestamp="[00:10]",
            ),
            ActionableIdea(
                title="Create a follow-up task",
                description="Turn one of the subtopics into a concrete next step.",
                timestamp="[00:30]",
            ),
            ActionableIdea(
                title="Set a quick deadline",
                description="Assign a near-term deadline to the most important action.",
                timestamp="[01:00]",
            ),
            ActionableIdea(
                title="Share with a peer",
                description="Review the idea with someone who can help keep you accountable.",
                timestamp="[01:50]",
            ),
            ActionableIdea(
                title="Track one success metric",
                description="Choose one measurable metric to evaluate progress.",
                timestamp="[02:20]",
            ),
        ]
    )

    return subtopics, ideas


def generate_all_outputs(
    area_of_life: str,
    specific_goal: str,
    retrieved_chunks: list[dict],
    llm_config: BaseLLMConfig | None = None,
    content: str = "all",
    debug_outputs: bool = GENERATOR_TEST_DEBUG,
) -> tuple[Subtopics, ActionableIdeas]:
    """Generate both subtopics and actionable ideas in sequence.

    Args:
        area_of_life: The selected area of life.
        specific_goal: The optional specific goal.
        retrieved_chunks: List of retrieved transcript chunks.
        llm_config: Optional LLMConfig instance.
        content: Optional, "all"/"subtopics"/"ideas",

    Returns:
        Tuple of (Subtopics, ActionableIdeas).

    Raises:
        GenerationError: If either generation step fails.
    """
    if debug_outputs:
        # 
        return _create_debug_outputs(area_of_life, specific_goal)

    if llm_config is None:
        try:
            llm_config = create_llm()
        except LLMConfigError as e:
            raise GenerationError(f"Failed to initialize LLM: {str(e)}")

    if content in ["all", "subtopics"]:
        if DEBUG:
            print("\n...generate_subtopics():", retrieved_chunks)
        subtopics = generate_subtopics(
            area_of_life, specific_goal, retrieved_chunks, llm_config
        )
        # handle edge case: area_of_life or specific_goal have no correlation with the video
        # so we return an artificial entry explaining that
        if subtopics.subtopics==[]:
            subtopics = Subtopics(
                subtopics=[
                    Subtopic(
                        title="Nothing found",
                        timestamp="[00:00]",
                        summary="No relevant ideas found related to the chosen area of life/goal",
                    ),
                ]
            )
    else:
        subtopics = None

    if content in ["all", "ideas"]:
        try:
            ideas = generate_actionable_ideas(
                area_of_life, specific_goal, retrieved_chunks, llm_config
            )
            # handle edge case: area_of_life or specific_goal have no correlation with the video
            # so we return an artificial entry explaining that
            if ideas.ideas==[]:
                ideas = ActionableIdeas(
                    ideas=[
                        ActionableIdea(
                            title="Nothing found",
                            description="No relevant ideas found related to the chosen area of life/goal",
                            timestamp="[00:00]",
                        ),
                    ]
                )
        except Exception as e:
            # FIXME for testing - returning the error as idea - to show at least topics
            if DEBUG:
                print("!! generate_actionable_ideas() error:", e)
            ideas = ActionableIdeas(
                ideas=[
                    ActionableIdea(
                        title="Generate actionable ideas failed",
                        description=str(e),
                        timestamp="[00:00]",
                    ),
                ]
            )
    else:
        ideas = None

    return subtopics, ideas
