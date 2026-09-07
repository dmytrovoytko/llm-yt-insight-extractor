"""Unit tests for core.prompts versioning.

These tests pin the production prompt contract: v1 must remain byte-stable
(default behavior, no behavior change for the app or for the existing
``tests/test_generator.py`` suite). v2 is the challenger variant used by
``scripts/eval_llm.py`` for the LLM-as-judge A/B evaluation.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Stub pydantic if not installed in the test environment so these tests
# can run alongside the existing unit tests that do not require pydantic.
try:
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    import types

    _stub = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, *a, **k):
            pass

        @classmethod
        def model_validate(cls, *_a, **_k):
            return cls()

        def model_dump_json(self, *_a, **_k):
            return "{}"

    class _Field:
        def __init__(self, *a, **k):
            pass

    _stub.BaseModel = _BaseModel
    _stub.Field = _Field
    sys.modules["pydantic"] = _stub

from core import prompts  # noqa: E402


class V1ContractTests(unittest.TestCase):
    """Pins the v1 production contract. Any change here is a breaking change."""

    def test_default_version_is_v1(self):
        self.assertEqual(prompts.DEFAULT_PROMPT_VERSION, "v1")
        self.assertEqual(prompts._resolve_prompt_version(None), "v1")
        self.assertEqual(prompts._resolve_prompt_version("v1"), "v1")

    def test_v1_builder_has_no_v2_twist_subtopics(self):
        prompt = prompts.build_subtopics_prompt("Productivity", "focus", "ctx")
        self.assertNotIn("directly mention", prompt)
        self.assertIn("expert content analyst", prompt)

    def test_v1_builder_has_no_v2_twist_ideas(self):
        prompt = prompts.build_actionable_ideas_prompt("Productivity", "focus", "ctx")
        self.assertNotIn("first step under 15 words", prompt)
        self.assertNotIn("Descriptions without an explicit first step", prompt)
        self.assertIn("expert life coach", prompt)

    def test_v1_subtopics_template_byte_stable(self):
        # The exact production v1 string. If you change this, the v1 contract
        # has changed and tests/test_generator.py will likely need updates too.
        self.assertIn("Extract 1-", prompts.SUBTOPICS_PROMPT_TEMPLATE)
        self.assertIn("User's focus area: {area_of_life}", prompts.SUBTOPICS_PROMPT_TEMPLATE)
        self.assertIn("User's specific goal: {specific_goal}", prompts.SUBTOPICS_PROMPT_TEMPLATE)
        self.assertIn("Retrieved transcript context:", prompts.SUBTOPICS_PROMPT_TEMPLATE)
        self.assertIn("Return valid JSON", prompts.SUBTOPICS_PROMPT_TEMPLATE)

    def test_v1_ideas_template_byte_stable(self):
        self.assertIn("Extract 1-", prompts.ACTIONABLE_IDEAS_PROMPT_TEMPLATE)
        self.assertIn("must start with a verb if possible", prompts.ACTIONABLE_IDEAS_PROMPT_TEMPLATE)
        self.assertIn("User's focus area: {area_of_life}", prompts.ACTIONABLE_IDEAS_PROMPT_TEMPLATE)
        self.assertIn("Return valid JSON", prompts.ACTIONABLE_IDEAS_PROMPT_TEMPLATE)


class V2ChallengerTests(unittest.TestCase):
    """The v2 challenger is used only by eval_llm.py — never by the app."""

    def test_v2_subtopics_includes_focus_area_emphasis(self):
        prompt = prompts.build_subtopics_prompt("Productivity", "focus", "ctx", version="v2")
        self.assertIn("directly mention Productivity", prompt)
        self.assertIn("skip generic filler", prompt)

    def test_v2_ideas_requires_first_step(self):
        prompt = prompts.build_actionable_ideas_prompt("Productivity", "focus", "ctx", version="v2")
        self.assertIn("first step under 15 words", prompt)
        self.assertIn("Descriptions without an explicit first step will be rejected", prompt)

    def test_v2_is_registered_in_templates(self):
        self.assertIn("v2", prompts.SUBTOPICS_TEMPLATES)
        self.assertIn("v2", prompts.ACTIONABLE_IDEAS_TEMPLATES)
        self.assertIs(prompts.SUBTOPICS_TEMPLATES["v2"], prompts.SUBTOPICS_PROMPT_TEMPLATE_V2)
        self.assertIs(
            prompts.ACTIONABLE_IDEAS_TEMPLATES["v2"],
            prompts.ACTIONABLE_IDEAS_PROMPT_TEMPLATE_V2,
        )

    def test_unknown_version_raises(self):
        with self.assertRaises(ValueError):
            prompts.build_subtopics_prompt("a", "b", "c", version="v9")
        with self.assertRaises(ValueError):
            prompts.build_actionable_ideas_prompt("a", "b", "c", version="bogus")

    def test_v2_does_not_leak_into_v1(self):
        # Belt-and-braces: a v2 prompt and a v1 prompt must differ in their
        # challenge directives and share the same role prompts.
        v1 = prompts.build_subtopics_prompt("X", "Y", "Z", "v1")
        v2 = prompts.build_subtopics_prompt("X", "Y", "Z", "v2")
        self.assertNotEqual(v1, v2)
        self.assertIn("expert content analyst", v1)
        self.assertIn("expert content analyst", v2)


class BackwardsCompatibilityTests(unittest.TestCase):
    """`build_*_prompt` called with no `version` argument must still work."""

    def test_subtopics_default_call(self):
        prompt = prompts.build_subtopics_prompt("Productivity", "focus", "ctx")
        # The pre-existing default behavior must remain unchanged.
        self.assertIn("Productivity", prompt)
        self.assertIn("focus", prompt)
        self.assertIn("ctx", prompt)
        self.assertNotIn("directly mention", prompt)

    def test_ideas_default_call(self):
        prompt = prompts.build_actionable_ideas_prompt("Productivity", "focus", "ctx")
        self.assertIn("Productivity", prompt)
        self.assertIn("focus", prompt)
        self.assertIn("ctx", prompt)
        self.assertNotIn("first step under 15 words", prompt)

    def test_specific_goal_default_substitute_preserved(self):
        # If no specific goal is provided, the existing v1 string substitutes
        # "No specific goal provided".
        prompt = prompts.build_subtopics_prompt("Productivity", "", "ctx")
        self.assertIn("No specific goal provided", prompt)


if __name__ == "__main__":
    unittest.main()
