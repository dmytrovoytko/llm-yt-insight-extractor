"""Unit tests for core/hybrid.py.

Pure-function tests, no ONNX, no chromadb, no pydantic.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import hybrid  # noqa: E402


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(
            hybrid.tokenize("Hello, World! 42 things."),
            ["hello", "world", "42"],
        )

    def test_drops_stopwords(self):
        toks = hybrid.tokenize("the quick brown fox is a fox")
        self.assertNotIn("the", toks)
        self.assertNotIn("is", toks)
        self.assertNotIn("a", toks)
        self.assertEqual(toks.count("fox"), 2)

    def test_empty(self):
        self.assertEqual(hybrid.tokenize(""), [])
        self.assertEqual(hybrid.tokenize("  , . "), [])


class KeywordScoreTests(unittest.TestCase):
    def test_full_overlap(self):
        # All query tokens present in document
        self.assertEqual(
            hybrid.keyword_score("productivity tips", "here are productivity tips for you"),
            1.0,
        )

    def test_no_overlap(self):
        self.assertEqual(
            hybrid.keyword_score("productivity", "fitness workout suggestions"),
            0.0,
        )

    def test_partial_overlap(self):
        # Two query tokens, one in document -> 0.5
        self.assertEqual(
            hybrid.keyword_score("productivity focus", "productivity tips"),
            0.5,
        )

    def test_empty_query_returns_zero(self):
        self.assertEqual(hybrid.keyword_score("", "anything"), 0.0)
        self.assertEqual(hybrid.keyword_score("the a an", "anything"), 0.0)

    def test_empty_doc_returns_zero(self):
        self.assertEqual(hybrid.keyword_score("productivity", ""), 0.0)
        self.assertEqual(hybrid.keyword_score("productivity", "the a an"), 0.0)

    def test_score_in_unit_interval(self):
        score = hybrid.keyword_score("a b c d", "a b")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class RrfFuseTests(unittest.TestCase):
    def test_both_agree_on_top(self):
        # Identical orderings -> fused order is the same
        order = [1, 2, 3, 4, 5]
        self.assertEqual(hybrid.rrf_fuse(order, order), order)

    def test_disjoint_rankings_are_dominant_in_their_order(self):
        # a ranks 1 first, b ranks 10 first
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        fused = hybrid.rrf_fuse(a, b)
        # First items are the "head" of both orderings; #1 should be in
        # the top 2 (it was #1 in a, last in b -> low b contribution).
        self.assertIn(1, fused[:2])
        self.assertIn(10, fused[:2])

    def test_includes_items_only_in_one_ordering(self):
        a = [1, 2, 3]
        b = [2, 3, 4]
        fused = hybrid.rrf_fuse(a, b)
        # 4 only appears in b, must still be in output
        self.assertIn(4, fused)
        self.assertEqual(sorted(fused), [1, 2, 3, 4])

    def test_rrf_k_changes_influence_of_low_ranks(self):
        a = [1, 2, 3]
        b = [2, 3, 1]
        # With k=1, top-1 dominates more aggressively
        fused_k1 = hybrid.rrf_fuse(a, b, k=1)
        fused_k100 = hybrid.rrf_fuse(a, b, k=100)
        self.assertEqual(len(fused_k1), 3)
        self.assertEqual(len(fused_k100), 3)
        # Both should still have 1, 2, 3 — only relative weight changes.
        self.assertEqual(sorted(fused_k1), [1, 2, 3])
        self.assertEqual(sorted(fused_k100), [1, 2, 3])

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            hybrid.rrf_fuse([1, 2], [2, 1], k=0)
        with self.assertRaises(ValueError):
            hybrid.rrf_fuse([1, 2], [2, 1], k=-5)

    def test_deterministic_tie_break(self):
        a = [1, 2, 3]
        b = [3, 2, 1]
        # Items 1 and 3 are at the extremes (rank 0 in one ordering,
        # rank 2 in the other) and so have a strictly higher fused RRF
        # score than item 2 (rank 1 in both). Tie-break between 1 and 3
        # is by ascending rank_a, so item 1 wins.
        fused = hybrid.rrf_fuse(a, b)
        self.assertEqual(fused, [1, 3, 2])

    def test_strong_overlap_wins(self):
        # Item A is rank 0 in both orderings -> highest fused score.
        a = ["A", "B", "C", "D"]
        b = ["A", "C", "B", "D"]
        fused = hybrid.rrf_fuse(a, b)
        self.assertEqual(fused[0], "A")

    def test_empty_inputs(self):
        self.assertEqual(hybrid.rrf_fuse([], []), [])


class HybridRrfScoreTests(unittest.TestCase):
    def test_returns_per_item_score(self):
        scores = hybrid.hybrid_rrf_score([1, 2, 3], [1, 2, 3])
        # Item present in both should have a strictly higher score
        # than an item only in one.
        only_a = hybrid.hybrid_rrf_score([1, 2, 3], [4, 5, 6])
        self.assertGreater(scores[1], only_a[4])


class ModeValidationTests(unittest.TestCase):
    def test_default_mode_is_vector(self):
        self.assertEqual(hybrid.DEFAULT_HYBRID_MODE, "vector")

    def test_known_modes(self):
        self.assertEqual(
            sorted(hybrid.HYBRID_MODES), ["hybrid", "keyword", "vector"]
        )

    def test_normalize_passes_through_known(self):
        self.assertEqual(hybrid.normalize_mode("vector"), "vector")
        self.assertEqual(hybrid.normalize_mode("keyword"), "keyword")
        self.assertEqual(hybrid.normalize_mode("hybrid"), "hybrid")

    def test_normalize_none_returns_default(self):
        self.assertEqual(hybrid.normalize_mode(None), "vector")
        self.assertEqual(hybrid.normalize_mode(""), "vector")

    def test_normalize_unknown_raises(self):
        with self.assertRaises(ValueError):
            hybrid.normalize_mode("bogus")
        with self.assertRaises(ValueError):
            hybrid.normalize_mode("VECTOR")  # case-sensitive

    def test_supported_modes(self):
        self.assertEqual(
            sorted(hybrid.supported_modes()), ["hybrid", "keyword", "vector"]
        )


class SettingsImportTests(unittest.TestCase):
    """core/settings.py should expose RETRIEVAL_MODE = 'vector'."""

    def test_retrieval_mode_default(self):
        from core.settings import RETRIEVAL_MODE  # noqa: WPS433

        self.assertEqual(RETRIEVAL_MODE, "vector")


class RrfPropertiesTests(unittest.TestCase):
    """Property-style sanity checks on RRF output."""

    def test_no_duplicates_in_output(self):
        fused = hybrid.rrf_fuse([1, 2, 3, 4, 5], [3, 4, 5, 1, 2])
        self.assertEqual(len(fused), len(set(fused)))

    def test_output_length_equals_union_of_inputs(self):
        a = [1, 2, 3, 4, 5]
        b = [4, 5, 6, 7, 8]
        fused = hybrid.rrf_fuse(a, b)
        self.assertEqual(len(fused), len(set(a) | set(b)))


if __name__ == "__main__":
    unittest.main()
