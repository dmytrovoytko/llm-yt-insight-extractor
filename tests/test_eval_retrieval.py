"""Unit tests for scripts/eval_retrieval.py.

Pure stdlib + the script itself. Does NOT load onnxruntime / chromadb.
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

import eval_retrieval  # noqa: E402  (path injection above)


def _make_chunk(text: str, start: float = 0.0, end: float = 60.0, cid: int = 0) -> dict:
    return {
        "text": text,
        "start_time": start,
        "end_time": end,
        "chunk_id": cid,
        "word_count": len(text.split()),
    }


class TokenizationTests(unittest.TestCase):
    def test_tokens_lowercases_and_strips_punctuation(self):
        tokens = eval_retrieval._tokens("Hello, World! 42 things.")
        self.assertEqual(tokens, ["hello", "world", "42"])

    def test_tokens_drops_stopwords(self):
        tokens = eval_retrieval._tokens("the quick brown fox is a fox")
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("fox", tokens)


class CosineAndOverlapTests(unittest.TestCase):
    def test_cosine_identical_vectors_is_one(self):
        a = eval_retrieval._doc_vector("productivity morning routine")
        self.assertAlmostEqual(eval_retrieval._cosine(a, a), 1.0, places=6)

    def test_cosine_disjoint_is_zero(self):
        a = eval_retrieval._doc_vector("productivity routine")
        b = eval_retrieval._doc_vector("fitness workout")
        self.assertEqual(eval_retrieval._cosine(a, b), 0.0)

    def test_cosine_handles_empty_inputs(self):
        empty = eval_retrieval._doc_vector("")
        full = eval_retrieval._doc_vector("anything")
        self.assertEqual(eval_retrieval._cosine(empty, full), 0.0)
        self.assertEqual(eval_retrieval._cosine(empty, empty), 0.0)

    def test_keyword_overlap_full_hit(self):
        score = eval_retrieval._keyword_overlap_score(
            ["productivity"], eval_retrieval._doc_vector("productivity tips")
        )
        self.assertEqual(score, 1.0)

    def test_keyword_overlap_partial(self):
        score = eval_retrieval._keyword_overlap_score(
            ["productivity", "focus"],
            eval_retrieval._doc_vector("productivity tips and tricks"),
        )
        self.assertAlmostEqual(score, 0.5, places=6)

    def test_keyword_overlap_empty_query(self):
        self.assertEqual(
            eval_retrieval._keyword_overlap_score([], eval_retrieval._doc_vector("x")),
            0.0,
        )


class MetricsTests(unittest.TestCase):
    def test_keyword_hit_rate_all_match(self):
        results = [
            _make_chunk("Productivity tips for [00:00]"),
            _make_chunk("More productivity here [01:00]"),
        ]
        self.assertEqual(eval_retrieval.keyword_hit_rate(results, "Productivity"), 1.0)

    def test_keyword_hit_rate_mixed(self):
        results = [
            _make_chunk("Productivity tips [00:00]"),
            _make_chunk("Fitness is good [01:00]"),
        ]
        self.assertEqual(eval_retrieval.keyword_hit_rate(results, "Productivity"), 0.5)

    def test_keyword_hit_rate_empty_results(self):
        self.assertEqual(eval_retrieval.keyword_hit_rate([], "anything"), 0.0)

    def test_timestamp_valid_rate(self):
        results = [
            _make_chunk("[00:10] Productivity matters"),
            _make_chunk("No timestamp here"),
        ]
        self.assertEqual(eval_retrieval.timestamp_valid_rate(results), 0.5)

    def test_overlap_diversity_distinct_chunks(self):
        results = [
            _make_chunk("alpha beta gamma delta epsilon"),
            _make_chunk("lorem ipsum dolor sit amet"),
        ]
        self.assertGreater(eval_retrieval.overlap_diversity(results), 0.9)

    def test_overlap_diversity_duplicate_chunks(self):
        results = [
            _make_chunk("productivity morning routine productivity"),
            _make_chunk("productivity morning routine productivity"),
        ]
        self.assertLess(eval_retrieval.overlap_diversity(results), 0.1)


class LocalRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            _make_chunk("[00:00] Productivity morning routine for focus", cid=0),
            _make_chunk("[01:00] Productivity tips and habits", cid=1),
            _make_chunk("[02:00] Fitness workout suggestions", cid=2),
            _make_chunk("[03:00] Career advice for promotion", cid=3),
        ]
        self.retriever = eval_retrieval.LocalRetriever(self.chunks)

    def test_vector_mode_returns_top_k(self):
        results, _ = self.retriever.retrieve(
            "Productivity morning", mandatory_keyword="", use_reranking=False, top_k=2
        )
        self.assertEqual(len(results), 2)
        self.assertIn("[00:00]", results[0]["text"])

    def test_keyword_filter_drops_irrelevant(self):
        results, _ = self.retriever.retrieve(
            "Productivity morning",
            mandatory_keyword="Productivity",
            use_reranking=True,
            top_k=5,
        )
        self.assertTrue(results)
        for r in results:
            self.assertIn("productivity", r["text"].lower())

    def test_keyword_mode_prefers_lexical_match(self):
        # Use goal that appears verbatim in chunk 1
        results, _ = self.retriever.retrieve(
            "tips habits",
            mandatory_keyword="",
            use_reranking=False,
            top_k=1,
            mode=eval_retrieval.LocalRetriever.KEYWORD,
        )
        self.assertEqual(len(results), 1)
        self.assertIn("tips", results[0]["text"].lower())

    def test_hybrid_mode_returns_results(self):
        results, latency = self.retriever.retrieve(
            "Productivity",
            mandatory_keyword="",
            use_reranking=False,
            top_k=2,
            mode=eval_retrieval.LocalRetriever.HYBRID,
        )
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(latency, 0.0)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("x", mode="bogus")

    def test_top_k_one_always_returns_one(self):
        results, _ = self.retriever.retrieve(
            "anything", mandatory_keyword="", use_reranking=False, top_k=1
        )
        self.assertEqual(len(results), 1)

    def test_hybrid_mode_uses_production_rrf(self):
        """The offline hybrid path must use the same RRF math as production.

        Cross-checks ``LocalRetriever.retrieve(mode='hybrid')`` ordering
        against a direct call to ``core.hybrid.rrf_fuse`` on the same
        candidate orderings. Prevents drift between task (1) and task (3).
        """
        from core.hybrid import rrf_fuse  # noqa: WPS433

        results, _ = self.retriever.retrieve(
            "Productivity morning",
            mandatory_keyword="",
            use_reranking=False,
            top_k=4,
            mode=eval_retrieval.LocalRetriever.HYBRID,
        )
        # Recompute the expected fused order from raw scores.
        scored = self.retriever.score("Productivity morning")
        by_vec = sorted(scored, key=lambda x: x[1], reverse=True)
        by_kw = sorted(scored, key=lambda x: x[2], reverse=True)
        vec_order = [e[3] for e in by_vec]
        kw_order = [e[3] for e in by_kw]
        expected = rrf_fuse(vec_order, kw_order)[:4]
        actual = [r["chunk_id"] for r in results]
        self.assertEqual(actual, expected)


class AggregateAndRenderTests(unittest.TestCase):
    def _row(self, **overrides):
        base = {
            "query": "Productivity",
            "area": "Productivity",
            "goal": "",
            "mode": "vector",
            "rerank": False,
            "filter": False,
            "keyword_for_filter": "",
            "expected_token": "Productivity",
            "n_returned": 4,
            "keyword_hit_rate": 0.75,
            "timestamp_valid_rate": 1.0,
            "overlap_diversity": 0.6,
            "latency_s": 0.01,
        }
        base.update(overrides)
        return base

    def test_aggregate_groups_by_config(self):
        rows = [
            self._row(rerank=False, filter=False),
            self._row(rerank=False, filter=False),
            self._row(rerank=True, filter=True),
        ]
        summary = eval_retrieval.aggregate(rows)
        labels = [s["config"] for s in summary]
        self.assertIn("A: vector-only", labels)
        self.assertIn("C: vector + rerank + kw filter", labels)
        # means over 2 rows of 0.75 = 0.75
        a = next(s for s in summary if s["config"] == "A: vector-only")
        self.assertAlmostEqual(a["avg_keyword_hit_rate"], 0.75, places=3)
        self.assertEqual(a["n_queries"], 2)

    def test_aggregate_groups_hybrid_configs(self):
        rows = [
            self._row(mode="keyword", rerank=False, filter=True),
            self._row(mode="hybrid", rerank=False, filter=True),
            self._row(mode="hybrid", rerank=True, filter=True),
        ]
        summary = eval_retrieval.aggregate(rows)
        labels = [s["config"] for s in summary]
        self.assertIn("D: keyword + kw filter", labels)
        self.assertIn("E: hybrid + kw filter", labels)
        self.assertIn("F: hybrid + rerank + kw filter", labels)

    def test_config_label_disambiguates_modes(self):
        # Same (rerank, filter) but different mode -> different label
        a_row = self._row(mode="vector", rerank=True, filter=True)
        b_row = self._row(mode="hybrid", rerank=True, filter=True)
        c_row = self._row(mode="keyword", rerank=True, filter=True)
        labels = {
            eval_retrieval._config_label(r) for r in (a_row, b_row, c_row)
        }
        self.assertEqual(len(labels), 3)

    def test_render_markdown_contains_aggregate_table(self):
        rows = [self._row()]
        summary = eval_retrieval.aggregate(rows)
        report = eval_retrieval.render_markdown("TrvLEgPpV8s", rows, summary)
        self.assertIn("# Retrieval evaluation", report)
        self.assertIn("`TrvLEgPpV8s`", report)
        self.assertIn("| Config | Queries |", report)
        self.assertIn("## Decision", report)

    def test_decision_text_picks_winner(self):
        rows = [
            self._row(rerank=False, filter=False, keyword_hit_rate=0.5),
            self._row(rerank=True, filter=False, keyword_hit_rate=0.7),
            self._row(rerank=True, filter=True, keyword_hit_rate=0.9),
        ]
        summary = eval_retrieval.aggregate(rows)
        decision = eval_retrieval._decision_text(summary)
        # C wins
        self.assertIn("C", decision)


class CachedIoTests(unittest.TestCase):
    def test_list_cached_ids_reads_directory(self):
        ids = eval_retrieval.list_cached_ids()
        # test cache directory should have at least the default video
        self.assertIn("TrvLEgPpV8s", ids)

    def test_load_cached_transcript_returns_entries(self):
        transcript = eval_retrieval.load_cached_transcript("TrvLEgPpV8s")
        self.assertGreater(len(transcript), 10)
        for entry in transcript:
            self.assertIn("text", entry)
            self.assertIn("start", entry)
            self.assertIn("duration", entry)

    def test_load_cached_transcript_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            eval_retrieval.load_cached_transcript("__nonexistent__")


class ScriptSmokeTests(unittest.TestCase):
    """End-to-end: the script should run offline and produce a report."""

    def test_script_runs_and_writes_report(self):
        with mock.patch.object(
            sys, "argv", ["eval_retrieval.py", "--cache-id", "TrvLEgPpV8s"]
        ):
            rc = eval_retrieval.main()
        self.assertEqual(rc, 0)
        report_path = REPO_ROOT / "docs" / "retrieval_eval.md"
        self.assertTrue(report_path.exists())
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("TrvLEgPpV8s", text)
        # Legacy vector-path labels
        self.assertIn("A: vector-only", text)
        self.assertIn("B: vector + rerank", text)
        self.assertIn("C: vector + rerank + kw filter", text)
        # New hybrid-search labels (task 3)
        self.assertIn("D: keyword + kw filter", text)
        self.assertIn("E: hybrid + kw filter", text)
        self.assertIn("F: hybrid + rerank + kw filter", text)
        # Cross-reference to the new helper module
        self.assertIn("core/hybrid.py", text)

    def test_print_only_writes_to_stdout(self):
        import io
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["eval_retrieval.py", "--print-only"]):
            with mock.patch.object(sys, "stdout", buf):
                rc = eval_retrieval.main()
        self.assertEqual(rc, 0)
        self.assertIn("Retrieval evaluation", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
