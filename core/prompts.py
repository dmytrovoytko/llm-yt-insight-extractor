"""Pydantic models and prompt templates for structured LLM output."""

from typing import List

from pydantic import BaseModel, Field

TOP_K = 3 # 5


class Subtopic(BaseModel):
    """A single subtopic extracted from the video transcript."""

    title: str # = Field(
    #     ...,
    #     description="Concise title of the subtopic (max 10 words)",
    #     max_length=50,
    # )
    timestamp: str # = Field(
    #     ...,
    #     description="Timestamp in [mm:ss] or [hh:mm:ss] format",
    #     pattern=r"\[\d{2}:\d{2}(:\d{2})?\]",
    # )
    summary: str # = Field(
    #     ...,
    #     description="Concise summary paragraph of the subtopic (1-3 sentences)",
    #     max_length=300,
    # )


class Subtopics(BaseModel):
    """Collection of subtopics extracted from the video."""

    subtopics: List[Subtopic] # = Field(
    #     ...,
    #     description="List of 1-5 key subtopics from the video",
    #     min_items=1,
    #     max_items=5,
    # )


class ActionableIdea(BaseModel):
    """A single actionable idea from the video."""

    title: str # = Field(
    #     ...,
    #     description="Actionable idea title (max 7 words)",
    #     max_length=50,
    # )
    description: str # = Field(
    #     ...,
    #     description="Description of the idea (max 3 sentences)",
    #     max_length=300,
    # )
    timestamp: str # = Field(
    #     ...,
    #     description="Timestamp in [mm:ss] or [hh:mm:ss] format where this idea is discussed",
    #     pattern=r"\[\d{2}:\d{2}(:\d{2})?\]",
    # )


class ActionableIdeas(BaseModel):
    """Collection of top actionable ideas from the video."""

    ideas: List[ActionableIdea] # = Field(
    #     ...,
    #     description="Top 5 actionable ideas with timestamps",
    #     min_items=1, # we strive for 5, 1 - still ok 
    #     max_items=5,
    # )


# Prompt templates for LLM generation

SUBTOPICS_PROMPT_TEMPLATE = """You are an expert content analyst. Extract 1-"""+str(TOP_K)+""" key subtopics from the following transcript context.

For each subtopic, provide:
1. A concise title (max 10 words)
2. The exact timestamp [mm:ss] where the subtopic starts
3. A brief summary (1-3 sentences)

User's focus area: {area_of_life}
User's specific goal: {specific_goal}

Retrieved transcript context:
{context}

Return valid JSON matching this schema exactly. Do NOT add any explanatory text before or after the JSON."""


ACTIONABLE_IDEAS_PROMPT_TEMPLATE = """You are an expert life coach. Extract 1-"""+str(TOP_K)+""" top actionable ideas from the following transcript context.

For each idea, provide:
1. A concise, actionable title (max 7 words, must start with a verb if possible)
2. A clear description (max 3 sentences) explaining how to implement it (max 100 words)
3. The exact timestamp [mm:ss] where this idea is discussed

User's focus area: {area_of_life}
User's specific goal: {specific_goal}

Retrieved transcript context:
{context}

Return valid JSON matching this schema exactly. Do NOT add any explanatory text before or after the JSON."""


def build_subtopics_prompt(
    area_of_life: str, specific_goal: str, context: str
) -> str:
    """Build the prompt for generating subtopics.

    Args:
        area_of_life: The selected area of life.
        specific_goal: The optional specific goal.
        context: The retrieved transcript context.

    Returns:
        Formatted prompt string.
    """
    return SUBTOPICS_PROMPT_TEMPLATE.format(
        area_of_life=area_of_life,
        specific_goal=specific_goal or "No specific goal provided",
        context=context,
    )


def build_actionable_ideas_prompt(
    area_of_life: str, specific_goal: str, context: str
) -> str:
    """Build the prompt for generating actionable ideas.

    Args:
        area_of_life: The selected area of life.
        specific_goal: The optional specific goal.
        context: The retrieved transcript context.

    Returns:
        Formatted prompt string.
    """
    return ACTIONABLE_IDEAS_PROMPT_TEMPLATE.format(
        area_of_life=area_of_life,
        specific_goal=specific_goal or "No specific goal provided",
        context=context,
    )
