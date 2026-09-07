"""Hybrid retrieval helpers: keyword scoring and RRF fusion.

Stdlib-only. No third-party dependencies. Designed to plug into
``RAGEngine.retrieve(mode=...)`` and to be unit-testable in isolation
(``tests/test_hybrid.py``).

Public surface:
    * ``keyword_score(query, text)`` -- length-normalized token overlap.
    * ``rrf_fuse(rank_a, rank_b, k=60)`` -- reciprocal rank fusion of two
      parallel orderings over the same items.
    * ``HYBRID_MODES`` -- the canonical list of supported retrieval modes.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

# Lowercase word tokens; matches the same pattern used in
# ``scripts/eval_retrieval.py`` so offline eval and online scoring agree.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# A small English stopword list. Trimmed to the common ones that hurt
# keyword overlap scores in YouTube transcripts. Same set as the eval script.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "for",
        "to", "is", "are", "was", "were", "be", "been", "being", "it",
        "this", "that", "these", "those", "i", "you", "we", "they",
        "he", "she", "him", "her", "his", "their", "our", "your", "my",
        "as", "at", "by", "with", "from", "so", "if", "not", "no", "do",
        "does", "did", "have", "has", "had", "will", "would", "can",
        "could", "should", "just", "like", "than", "then", "there",
        "what", "when", "where", "how", "why", "who", "which", "some",
        "any", "all", "more", "most", "other", "into", "out", "up",
        "down", "about", "over", "after", "before", "very", "too",
        "also", "really", "actually", "going", "got", "get", "one",
        "two", "three", "okay", "ok", "yeah", "right", "well", "thing",
        "things", "people", "time", "way", "make", "made", "say", "said",
        "go", "know", "think", "want", "need", "take", "see", "look",
        "use", "find", "give", "tell", "work", "call", "try",
    }
)


# Supported retrieval modes. ``"vector"`` is the default and matches the
# historical behavior of ``RAGEngine.retrieve`` before the hybrid option
# was added. ``"hybrid"`` is what most users will want; ``"keyword"`` is
# useful as a diagnostic for transcript-only matching.
HYBRID_MODES: tuple[str, ...] = ("vector", "keyword", "hybrid")
DEFAULT_HYBRID_MODE: str = "vector"

# Default RRF constant. 60 is the standard value used in the original
# RRF paper (Cormack et al., 2009); 1-100 all give similar results in
# practice, but 60 is the safest default.
DEFAULT_RRF_K: int = 60


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization; drops punctuation and stopwords.

    Matches the helper used by ``scripts/eval_retrieval.py`` so the
    offline eval and the production retriever produce comparable scores.
    """
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS]


def keyword_score(query: str, text: str) -> float:
    """Length-normalized token overlap between query and document text.

    Returns the fraction of query tokens (after stopword removal) that
    appear in ``text``. The score is in ``[0.0, 1.0]``. An empty query
    or query with only stopwords returns 0.0.
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    hits = sum(1 for t in q_tokens if t in text_tokens)
    return hits / len(q_tokens)


def _to_rank_map(order: Sequence[int]) -> dict[int, int]:
    """Turn a list of item-ids into a {item_id: rank} dict (0-indexed)."""
    return {item_id: rank for rank, item_id in enumerate(order)}


def rrf_fuse(
    order_a: Sequence[int],
    order_b: Sequence[int],
    k: int = DEFAULT_RRF_K,
) -> list[int]:
    """Reciprocal rank fusion of two orderings over the same item ids.

    Items present in only one of the two orderings are still included;
    they receive the maximum rank (i.e. ``len(order)``) in the missing
    ordering, so they only contribute from the side that observed them.

    Args:
        order_a: First ordering of item ids, most-relevant first.
        order_b: Second ordering of item ids, most-relevant first.
        k: RRF smoothing constant. Larger values reduce the influence of
            high ranks; ``k=60`` is the literature default.

    Returns:
        A new list of item ids, sorted by fused score (descending).
        Deterministic: ties are broken by ascending order_a rank, then
        ascending order_b rank.
    """
    if k <= 0:
        raise ValueError("RRF constant k must be positive.")
    # All ids, with default rank == len(order) for missing entries.
    fallback_a = len(order_a)
    fallback_b = len(order_b)
    rank_a = _to_rank_map(order_a)
    rank_b = _to_rank_map(order_b)
    all_ids = set(rank_a) | set(rank_b)
    scored: list[tuple[float, int, int, int]] = []
    for item_id in all_ids:
        ra = rank_a.get(item_id, fallback_a)
        rb = rank_b.get(item_id, fallback_b)
        score = 1.0 / (k + ra) + 1.0 / (k + rb)
        scored.append((score, ra, rb, item_id))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [t[3] for t in scored]


def hybrid_rrf_score(
    vector_order: Sequence[int],
    keyword_order: Sequence[int],
    k: int = DEFAULT_RRF_K,
) -> dict[int, float]:
    """Same as ``rrf_fuse`` but returns the per-item fused score, not the order.

    Useful when callers want to keep the score next to each item (e.g.
    to surface the fused score in the UI).
    """
    if k <= 0:
        raise ValueError("RRF constant k must be positive.")
    fallback_a = len(vector_order)
    fallback_b = len(keyword_order)
    rank_a = _to_rank_map(vector_order)
    rank_b = _to_rank_map(keyword_order)
    out: dict[int, float] = {}
    for item_id in set(rank_a) | set(rank_b):
        ra = rank_a.get(item_id, fallback_a)
        rb = rank_b.get(item_id, fallback_b)
        out[item_id] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out


def normalize_mode(mode: str | None) -> str:
    """Return a known mode, or raise ``ValueError`` with a helpful message."""
    if not mode:
        return DEFAULT_HYBRID_MODE
    if mode not in HYBRID_MODES:
        raise ValueError(
            f"Unknown retrieval mode: {mode!r}. Known modes: {list(HYBRID_MODES)}"
        )
    return mode


def supported_modes() -> Iterable[str]:
    """Return the canonical list of supported modes (read-only view)."""
    return HYBRID_MODES
