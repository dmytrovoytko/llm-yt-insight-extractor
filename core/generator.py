"""LLM generation pipeline for structured output (subtopics and actionable ideas)."""

from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from core.llm_config import OllamaLLMConfig, OllamaLLMError, create_ollama_llm
from core.prompts import (
    ActionableIdeas,
    Subtopics,
    build_actionable_ideas_prompt,
    build_subtopics_prompt,
)


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
    llm_config: OllamaLLMConfig | None = None,
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
            llm_config = create_ollama_llm()
        except OllamaLLMError as e:
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
        return subtopics
    except OllamaLLMError as e:
        raise GenerationError(f"LLM failed to generate subtopics: {str(e)}")
    except Exception as e:
        raise GenerationError(
            f"Unexpected error during subtopic generation: {str(e)}"
        )


def generate_actionable_ideas(
    area_of_life: str,
    specific_goal: str,
    retrieved_chunks: list[dict],
    llm_config: OllamaLLMConfig | None = None,
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
            llm_config = create_ollama_llm()
        except OllamaLLMError as e:
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
        ideas = llm_config.structured_complete(prompt, ActionableIdeas)
        return ideas
    except OllamaLLMError as e:
        raise GenerationError(
            f"LLM failed to generate actionable ideas: {str(e)}"
        )
    except Exception as e:
        raise GenerationError(
            f"Unexpected error during ideas generation: {str(e)}"
        )


def generate_all_outputs(
    area_of_life: str,
    specific_goal: str,
    retrieved_chunks: list[dict],
    llm_config: OllamaLLMConfig | None = None,
) -> tuple[Subtopics, ActionableIdeas]:
    """Generate both subtopics and actionable ideas in sequence.

    Args:
        area_of_life: The selected area of life.
        specific_goal: The optional specific goal.
        retrieved_chunks: List of retrieved transcript chunks.
        llm_config: Optional LLMConfig instance.

    Returns:
        Tuple of (Subtopics, ActionableIdeas).

    Raises:
        GenerationError: If either generation step fails.
    """
    if llm_config is None:
        try:
            llm_config = create_ollama_llm()
        except OllamaLLMError as e:
            raise GenerationError(f"Failed to initialize LLM: {str(e)}")

    subtopics = generate_subtopics(
        area_of_life, specific_goal, retrieved_chunks, llm_config
    )
    ideas = generate_actionable_ideas(
        area_of_life, specific_goal, retrieved_chunks, llm_config
    )

    return subtopics, ideas
