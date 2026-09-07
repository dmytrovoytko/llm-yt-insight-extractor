"""Unit tests for scripts/eval_llm.py.

Pure stdlib + the script itself. No LLM, no pydantic, no onnxruntime.
The script auto-stubs pydantic if missing, so we import it directly here.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_llm  # noqa: E402  (path injection above)


# --------------------------------------------------------------------------- #
# Deterministic metrics
# --------------------------------------------------------------------------- #

class TimestampFormatTests(unittest.TestCase):
    def test_mmss_valid(self):
        self.assertTrue(eval_llm.timestamp_format_ok("[01:30]"))

    def test_hhmmss_valid(self):
        self.assertTrue(eval_llm.timestamp_format_ok("[01:01:30]"))

    def test_unbracketed_rejected(self):
        # common bug surfaced by historical data
        self.assertFalse(eval_llm.timestamp_format_ok("01:30"))
        self.assertFalse(eval_llm.timestamp_format_ok("1:30"))
        self.assertFalse(eval_llm.timestamp_format_ok("[1:30]"))

    def test_empty_rejected(self):
        self.assertFalse(eval_llm.timestamp_format_ok(""))
        self.assertFalse(eval_llm.timestamp_format_ok("[]"))


class CountInRangeTests(unittest.TestCase):
    def test_empty_rejected(self):
        self.assertFalse(eval_llm.count_in_range([]))

    def test_one_accepted(self):
        self.assertTrue(eval_llm.count_in_range([{"a": 1}]))

    def test_above_max_rejected(self):
        items = [{} for _ in range(eval_llm.TOP_K + 1)]
        self.assertFalse(eval_llm.count_in_range(items))

    def test_at_max_accepted(self):
        items = [{} for _ in range(eval_llm.TOP_K)]
        self.assertTrue(eval_llm.count_in_range(items))


class KeywordCoverageTests(unittest.TestCase):
    def test_full_coverage(self):
        items = [
            {"title": "Productivity tips", "summary": "be more productive"},
            {"title": "Productivity hacks", "summary": "productivity mindset"},
        ]
        self.assertEqual(eval_llm.keyword_coverage(items, "Productivity", ""), 1.0)

    def test_no_coverage(self):
        items = [{"title": "x", "summary": "y"}, {"title": "a", "summary": "b"}]
        self.assertEqual(eval_llm.keyword_coverage(items, "Productivity", ""), 0.0)

    def test_partial(self):
        items = [
            {"title": "Productivity", "summary": "y"},
            {"title": "Finance", "summary": "y"},
        ]
        self.assertEqual(eval_llm.keyword_coverage(items, "Productivity", ""), 0.5)

    def test_empty_items(self):
        self.assertEqual(eval_llm.keyword_coverage([], "Productivity", ""), 0.0)

    def test_no_relevant_tokens(self):
        # area="" goal="" → no relevant tokens
        self.assertEqual(eval_llm.keyword_coverage([{"title": "anything"}], "", ""), 0.0)

    def test_description_field_used(self):
        items = [{"title": "x", "description": "productivity"}]
        self.assertEqual(eval_llm.keyword_coverage(items, "Productivity", ""), 1.0)


class FirstStepPresentTests(unittest.TestCase):
    def test_imperative_verb_accepted(self):
        self.assertTrue(eval_llm.first_step_present("Take a break. Start now."))

    def test_long_sentence_rejected(self):
        self.assertFalse(
            eval_llm.first_step_present(
                "Take a break and then consider the larger question of whether you are really doing what matters most to you in your life."
            )
        )

    def test_no_imperative_rejected(self):
        self.assertFalse(eval_llm.first_step_present("This is a description. Nothing to do here."))

    def test_empty_rejected(self):
        self.assertFalse(eval_llm.first_step_present(""))
        self.assertFalse(eval_llm.first_step_present("..."))


class ScoreSubtopicsTests(unittest.TestCase):
    def test_valid_payload(self):
        payload = {
            "subtopics": [
                {"title": "Productivity tips", "timestamp": "[01:30]", "summary": "be more productive"},
            ]
        }
        s = eval_llm.score_subtopics(payload, "Productivity", "")
        self.assertEqual(s["json_valid"], 1)
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["count_in_range"], 1)
        self.assertEqual(s["timestamp_valid"], 1)
        self.assertEqual(s["keyword_coverage"], 1.0)

    def test_unbracketed_timestamps(self):
        payload = {"subtopics": [{"title": "x", "timestamp": "01:30", "summary": "y"}]}
        s = eval_llm.score_subtopics(payload, "Career", "")
        self.assertEqual(s["timestamp_valid"], 0)

    def test_empty_payload(self):
        s = eval_llm.score_subtopics({}, "x", "")
        self.assertEqual(s["json_valid"], 0)
        self.assertEqual(s["n"], 0)

    def test_missing_subtopics_key(self):
        s = eval_llm.score_subtopics(None, "x", "")
        self.assertEqual(s["json_valid"], 0)
        self.assertEqual(s["n"], 0)


class ScoreIdeasTests(unittest.TestCase):
    def test_valid_payload(self):
        payload = {
            "ideas": [
                {
                    "title": "Take a break",
                    "description": "Schedule a break. Start tomorrow.",
                    "timestamp": "[01:30]",
                }
            ]
        }
        s = eval_llm.score_ideas(payload, "Productivity", "")
        self.assertEqual(s["json_valid"], 1)
        self.assertEqual(s["timestamp_valid"], 1)
        self.assertEqual(s["first_step"], 1)

    def test_missing_first_step(self):
        payload = {
            "ideas": [
                {
                    "title": "Be aware",
                    "description": "Just think about things generally without taking action.",
                    "timestamp": "[01:30]",
                }
            ]
        }
        s = eval_llm.score_ideas(payload, "Productivity", "")
        self.assertEqual(s["first_step"], 0)

    def test_empty(self):
        s = eval_llm.score_ideas(None, "x", "")
        self.assertEqual(s["json_valid"], 0)
        self.assertEqual(s["n"], 0)


# --------------------------------------------------------------------------- #
# Sample loading + selection
# --------------------------------------------------------------------------- #

class SampleSelectionTests(unittest.TestCase):
    def _entry(self, video_id: str, area: str = "Productivity") -> dict:
        return {
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "video_title": "title",
            "area_of_life": area,
            "goal": "",
            "subtopics": {"subtopics": []},
            "actionable_ideas": {"ideas": []},
            "timestamp": "2024-01-01T00:00:00",
            "llm_info": "Ollama: llama3.2:1b",
        }

    def test_select_recent_unique(self):
        history = [
            self._entry("AAA"),
            self._entry("AAA"),  # duplicate
            self._entry("BBB"),
            self._entry("CCC"),
        ]
        picked = eval_llm.select_samples(history, n=2)
        self.assertEqual(len(picked), 2)
        # Most recent unique first
        self.assertEqual(eval_llm.video_id_from_url(picked[0]["video_url"]), "CCC")
        self.assertEqual(eval_llm.video_id_from_url(picked[1]["video_url"]), "BBB")

    def test_select_zero_returns_empty(self):
        self.assertEqual(eval_llm.select_samples([self._entry("x")], 0), [])

    def test_select_more_than_history(self):
        history = [self._entry("AAA"), self._entry("BBB")]
        picked = eval_llm.select_samples(history, n=10)
        self.assertEqual(len(picked), 2)

    def test_video_id_from_url(self):
        self.assertEqual(
            eval_llm.video_id_from_url("https://www.youtube.com/watch?v=abc123"), "abc123"
        )
        self.assertEqual(eval_llm.video_id_from_url("https://youtu.be/xyz789"), "xyz789")
        self.assertEqual(eval_llm.video_id_from_url(""), "")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

class AggregateTests(unittest.TestCase):
    def test_empty_rows(self):
        summary = eval_llm.aggregate([])
        self.assertEqual(summary["n_samples"], 0)
        self.assertEqual(summary["subtopics"]["v1"]["n"], 0)
        self.assertEqual(summary["subtopics"]["v2"]["n"], 0)

    def test_perfect_score(self):
        rows = [
            {
                "v1_subtopics": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 1.0, "first_step": 1},
                "v2_subtopics": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 1.0, "first_step": 1},
                "v1_ideas": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 1.0, "first_step": 1},
                "v2_ideas": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 1.0, "first_step": 1},
            }
        ]
        summary = eval_llm.aggregate(rows)
        self.assertEqual(summary["subtopics"]["v1"]["json_valid_rate"], 1.0)
        self.assertEqual(summary["subtopics"]["v2"]["json_valid_rate"], 1.0)
        self.assertEqual(summary["ideas"]["v1"]["first_step_rate"], 1.0)

    def test_v2_better_on_first_step(self):
        rows = [
            {
                "v1_subtopics": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 0},
                "v2_subtopics": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 1},
                "v1_ideas": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 0},
                "v2_ideas": {"json_valid": 1, "n": 4, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 1},
            }
        ]
        summary = eval_llm.aggregate(rows)
        self.assertEqual(summary["ideas"]["v1"]["first_step_rate"], 0.0)
        self.assertEqual(summary["ideas"]["v2"]["first_step_rate"], 1.0)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

class MarkdownRenderTests(unittest.TestCase):
    def _row(self, sub_v1=None, sub_v2=None, idea_v1=None, idea_v2=None):
        sub_v1 = sub_v1 or {"json_valid": 1, "n": 3, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 0}
        sub_v2 = sub_v2 or {"json_valid": 1, "n": 3, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 0}
        idea_v1 = idea_v1 or {"json_valid": 1, "n": 3, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 0}
        idea_v2 = idea_v2 or {"json_valid": 1, "n": 3, "count_in_range": 1, "timestamp_valid": 1, "keyword_coverage": 0.5, "first_step": 0}
        return {
            "video_id": "abc123",
            "area": "Productivity",
            "goal": "",
            "llm_info": "Ollama: llama3.2:1b",
            "v1_subtopics": sub_v1,
            "v2_subtopics": sub_v2,
            "v1_ideas": idea_v1,
            "v2_ideas": idea_v2,
        }

    def test_render_contains_required_sections(self):
        rows = [self._row()]
        summary = eval_llm.aggregate(rows)
        report = eval_llm.render_markdown(
            samples=rows,
            rows=rows,
            summary=summary,
            prompt_versions={"v1": "v1", "v2": "v2"},
            live=False,
            use_judge=False,
        )
        for section in [
            "# LLM evaluation",
            "## Prompts under evaluation",
            "## Mode",
            "## Aggregate metrics",
            "## Per-sample results",
            "## Findings",
            "## Decision",
            "## How to re-run",
        ]:
            self.assertIn(section, report)

    def test_render_offline_mode_message(self):
        rows = [self._row()]
        summary = eval_llm.aggregate(rows)
        report = eval_llm.render_markdown(
            samples=rows, rows=rows, summary=summary,
            prompt_versions={"v1": "v1", "v2": "v2"},
            live=False, use_judge=False,
        )
        self.assertIn("Offline", report)
        self.assertIn("re-score historical v1 outputs", report)

    def test_render_live_mode_message(self):
        rows = [self._row()]
        summary = eval_llm.aggregate(rows)
        report = eval_llm.render_markdown(
            samples=rows, rows=rows, summary=summary,
            prompt_versions={"v1": "v1", "v2": "v2"},
            live=True, use_judge=False,
        )
        self.assertIn("Live", report)
        self.assertIn("re-ran each sample", report)

    def test_render_includes_delta_column(self):
        rows = [self._row()]
        summary = eval_llm.aggregate(rows)
        report = eval_llm.render_markdown(
            samples=rows, rows=rows, summary=summary,
            prompt_versions={"v1": "v1", "v2": "v2"},
            live=False, use_judge=False,
        )
        # Δ column header
        self.assertIn("Δ", report)


# --------------------------------------------------------------------------- #
# Findings (timestamp-validity note)
# --------------------------------------------------------------------------- #

class FindingsTests(unittest.TestCase):
    def _sample(self, sub_ts="[01:30]", idea_ts="[02:00]"):
        return {
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "area_of_life": "Productivity",
            "goal": "",
            "subtopics": {
                "subtopics": [
                    {"title": "Focus tips", "timestamp": sub_ts, "summary": "Productivity focus."}
                ]
            },
            "actionable_ideas": {
                "ideas": [
                    {
                        "title": "Take a break",
                        "description": "Schedule a break. Start tomorrow.",
                        "timestamp": idea_ts,
                    }
                ]
            },
        }

    def _rows_for(self, samples):
        return eval_llm.evaluate_offline(samples)

    def test_unbracketed_timestamps_produce_note(self):
        samples = [self._sample(sub_ts="01:30", idea_ts="02:00")]
        rows = self._rows_for(samples)
        notes = eval_llm._findings(samples, rows)
        self.assertTrue(notes, "expected a findings note for unbracketed timestamps")
        text = " ".join(notes)
        self.assertIn("timestamp_valid_rate", text)
        self.assertIn("bracket", text)
        # Must mention the display/export fallback so reviewers don't read
        # 0.000 as broken output.
        self.assertIn("core/exports.py", text)

    def test_bracketed_timestamps_produce_no_timestamp_note(self):
        samples = [self._sample()]
        rows = self._rows_for(samples)
        notes = eval_llm._findings(samples, rows)
        self.assertFalse(
            [n for n in notes if "timestamp_valid_rate" in n],
            f"unexpected timestamp note: {notes}",
        )

    def test_findings_reads_raw_samples_not_rows(self):
        # Rows hold aggregates only (no item lists). The note must still fire
        # because the raw samples carry the unbracketed timestamps.
        samples = [self._sample(sub_ts="01:30")]
        rows = self._rows_for(samples)
        self.assertNotIn("subtopics", rows[0]["v1_subtopics"])
        notes = eval_llm._findings(samples, rows)
        self.assertTrue(any("timestamp_valid_rate" in n for n in notes))

    def test_render_includes_timestamp_note(self):
        samples = [self._sample(sub_ts="01:30")]
        rows = self._rows_for(samples)
        summary = eval_llm.aggregate(rows)
        report = eval_llm.render_markdown(
            samples=samples, rows=rows, summary=summary,
            prompt_versions={"v1": "v1", "v2": "v2"},
            live=False, use_judge=False,
        )
        self.assertIn("timestamp_valid_rate", report)
        self.assertNotIn("No notable findings", report)


# --------------------------------------------------------------------------- #
# Script smoke (offline)
# --------------------------------------------------------------------------- #

class ScriptSmokeTests(unittest.TestCase):
    def test_script_runs_offline(self):
        with mock.patch.object(
            sys, "argv", ["eval_llm.py", "--samples", "3", "--out", str(REPO_ROOT / "docs" / "llm_eval.md")]
        ):
            rc = eval_llm.main()
        self.assertEqual(rc, 0)
        report_path = REPO_ROOT / "docs" / "llm_eval.md"
        self.assertTrue(report_path.exists())
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("# LLM evaluation", text)
        self.assertIn("v1", text)
        self.assertIn("v2", text)

    def test_print_only(self):
        import io

        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["eval_llm.py", "--print-only", "--samples", "2"]):
            with mock.patch.object(sys, "stdout", buf):
                rc = eval_llm.main()
        self.assertEqual(rc, 0)
        self.assertIn("# LLM evaluation", buf.getvalue())

    def test_no_history_returns_error(self):
        with mock.patch.object(eval_llm, "load_history", return_value=[]):
            with mock.patch.object(sys, "argv", ["eval_llm.py", "--samples", "1"]):
                rc = eval_llm.main()
        self.assertEqual(rc, 1)


# --------------------------------------------------------------------------- #
# Offline evaluation of historical data (real end-to-end)
# --------------------------------------------------------------------------- #

class OfflineEvalTests(unittest.TestCase):
    def test_evaluate_offline_preserves_v1_scores(self):
        history = eval_llm.load_history()
        samples = eval_llm.select_samples(history, 3)
        if not samples:
            self.skipTest("no history available")
        rows = eval_llm.evaluate_offline(samples)
        self.assertEqual(len(rows), len(samples))
        for r in rows:
            self.assertIn("v1_subtopics", r)
            self.assertIn("v2_subtopics", r)
            self.assertIn("v1_ideas", r)
            self.assertIn("v2_ideas", r)
            # v1 and v2 are identical in offline mode
            for key in ("json_valid", "n", "count_in_range", "timestamp_valid", "keyword_coverage"):
                self.assertEqual(
                    r["v1_subtopics"].get(key, 0),
                    r["v2_subtopics"].get(key, 0),
                    f"v1 != v2 for {key} in offline mode",
                )


if __name__ == "__main__":
    unittest.main()
