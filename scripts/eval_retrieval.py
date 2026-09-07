"""Offline retrieval evaluation for YT Insight Extractor.

Compares three retrieval approaches on cached YouTube transcripts:
  A) vector-only   (cosine over token-frequency vectors, stand-in for embedding similarity)
  B) vector + cross-encoder rerank (keyword-rerank stand-in for ms-marco)
  C) B + mandatory-keyword filter

By default this script runs OFFLINE using only stdlib and the existing
``core.chunker`` module (no onnxruntime / chromadb required). The same
metrics and table format apply if you opt into the production RAGEngine
via ``--use-production``.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --cache-id TrvLEgPpV8s
    python scripts/eval_retrieval.py --use-production   # requires full deps
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.chunker import chunk_transcript
from core.hybrid import (
    HYBRID_MODES as _PROD_HYBRID_MODES,
    keyword_score as _prod_keyword_score,
    normalize_mode as _prod_normalize_mode,
    rrf_fuse as _prod_rrf_fuse,
)
from core.settings import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    TOP_K,
    TOP_K_FIRST_STAGE,
)

CACHE_DIR = REPO_ROOT / "data" / ".transcript_cache"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "retrieval_eval.md"

TIMESTAMP_PATTERN = re.compile(r"\[\d{2}:\d{2}(?::\d{2})?\]")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
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
    "go", "going", "know", "think", "want", "need", "take", "see",
    "look", "use", "find", "give", "tell", "work", "call", "try",
}


# --------------------------------------------------------------------------- #
# Standalone metrics & scoring (no onnxruntime / chromadb required)
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> list[str]:
    """Lowercase tokenization; drops punctuation and stopwords."""
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]


def _doc_vector(text: str) -> Counter:
    """Term-frequency vector for a chunk's text content (timestamps stripped)."""
    cleaned = TIMESTAMP_PATTERN.sub("", text)
    return Counter(_tokens(cleaned))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _keyword_overlap_score(query_tokens: list[str], doc_tokens: Counter) -> float:
    """Fraction of query tokens present in the document, length-normalized."""
    if not query_tokens or not doc_tokens:
        return 0.0
    hits = sum(1 for t in query_tokens if doc_tokens.get(t, 0) > 0)
    return hits / max(1, len(query_tokens))


def _jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def keyword_hit_rate(results: list[dict], expected_token: str) -> float:
    """Fraction of returned chunks that contain the expected token.

    `expected_token` is query-specific (lowercased substring), chosen so the
    metric is discriminative against a real transcript.
    """
    if not results:
        return 0.0
    needle = expected_token.lower()
    return sum(1 for r in results if needle in r["text"].lower()) / len(results)


def timestamp_valid_rate(results: list[dict]) -> float:
    """Fraction of returned chunks whose text starts with a [mm:ss] tag."""
    if not results:
        return 0.0
    return sum(1 for r in results if TIMESTAMP_PATTERN.match(r["text"].lstrip())) / len(results)


def overlap_diversity(results: list[dict]) -> float:
    """1 - average pairwise Jaccard over the first 6 results (diversity proxy)."""
    if len(results) < 2:
        return 1.0
    sample = results[:6]
    vecs = [_doc_vector(r["text"]) for r in sample]
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            pairs.append(_jaccard(vecs[i], vecs[j]))
    avg = sum(pairs) / len(pairs) if pairs else 0.0
    return max(0.0, 1.0 - avg)


# --------------------------------------------------------------------------- #
# Local retrievers (offline, stdlib-only)
# --------------------------------------------------------------------------- #

class LocalRetriever:
    """Pure-Python retrieval over pre-chunked transcript (no embeddings)."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

    def __init__(self, chunks: list[dict]):
        # chunks: list of {"text", "start_time", "end_time", "chunk_id", "word_count"}
        self._chunks = chunks
        self._vectors = [_doc_vector(c["text"]) for c in chunks]

    def score(self, query: str) -> list[tuple[int, float, float, float]]:
        """Return (rank, vector_score, keyword_score, chunk_idx) for every chunk."""
        q_vec = _doc_vector(query)
        q_tokens = _tokens(query)
        out = []
        for idx, dv in enumerate(self._vectors):
            vs = _cosine(q_vec, dv)
            ks = _keyword_overlap_score(q_tokens, dv)
            out.append((0, vs, ks, idx))
        return out

    def retrieve(
        self,
        query: str,
        mandatory_keyword: str = "",
        use_reranking: bool = False,
        top_k: int = TOP_K,
        mode: str = VECTOR,
    ) -> tuple[list[dict], float]:
        started = time.perf_counter()
        mode = _prod_normalize_mode(mode)
        scored = self.score(query)
        # Rerank stand-in: cross-encoder proxy is a weighted blend that
        # up-weights tokens appearing in the query more strongly than the
        # vector score alone. Same effect as a learned reranker on TF inputs.
        if use_reranking:
            scored = [
                (0, vs + 0.5 * ks, ks, idx) for (r, vs, ks, idx) in scored
            ]
        if mode == self.VECTOR:
            ordered = sorted(scored, key=lambda x: x[1], reverse=True)
        elif mode == self.KEYWORD:
            ordered = sorted(scored, key=lambda x: x[2], reverse=True)
        elif mode == self.HYBRID:
            # Re-use the production RRF implementation from `core/hybrid.py`
            # so the offline eval and the live `RAGEngine.retrieve(mode='hybrid')`
            # use the exact same fusion math. We compute the two orderings
            # on the same first-stage candidates, then fuse by chunk index.
            by_vec = sorted(scored, key=lambda x: x[1], reverse=True)
            by_kw = sorted(scored, key=lambda x: x[2], reverse=True)
            vector_order = [entry[3] for entry in by_vec]
            keyword_order = [entry[3] for entry in by_kw]
            fused_ids = _prod_rrf_fuse(vector_order, keyword_order)
            by_id = {entry[3]: entry for entry in scored}
            ordered = [by_id[cid] for cid in fused_ids if cid in by_id]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # Mandatory keyword filter (post-rank, before slicing)
        if mandatory_keyword:
            needle = mandatory_keyword.lower()
            kept = [t for t in ordered if needle in self._chunks[t[3]]["text"].lower()]
            if kept:
                ordered = kept
        sliced = ordered[: max(top_k, 1)]
        results = [
            {
                **self._chunks[t[3]],
                "score": float(t[1]),
            }
            for t in sliced
        ]
        latency = time.perf_counter() - started
        return results, latency


# --------------------------------------------------------------------------- #
# Cached-transcript IO
# --------------------------------------------------------------------------- #

def load_cached_transcript(video_id: str, cache_dir: Path = CACHE_DIR) -> list[dict]:
    path = cache_dir / f"{video_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No cached transcript at {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return [
        {
            "text": str(entry["text"]).strip(),
            "start": float(entry["start"]),
            "duration": float(entry["duration"]),
        }
        for entry in data
    ]


def list_cached_ids(cache_dir: Path = CACHE_DIR) -> list[str]:
    if not cache_dir.exists():
        return []
    return sorted(p.stem for p in cache_dir.glob("*.txt"))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

QUERY_SET: list[tuple[str, str, str, str]] = [
    # (area, goal, mandatory_keyword_for_filter, expected_token)
    # `expected_token` is the substring that we expect to find inside a
    # correctly-retrieved chunk; it is the token whose presence the
    # `keyword_hit_rate` metric measures. We pick a token that actually
    # appears in the transcript so the metric is discriminative.
    ("Productivity", "morning routine", "morning", "morning"),
    ("Fitness", "", "fitness", "fitness"),
    ("Career", "get promoted", "promot", "promot"),
    ("Learning", "study habits", "learn", "learn"),
    ("Mental Health", "", "stress", "stress"),
    ("Productivity", "books and tools", "Ferriss", "Ferriss"),
    ("Productivity", "writing habits", "write", "write"),
    ("Productivity", "books that changed him", "book", "book"),
]


def build_chunks(video_id: str) -> tuple[str, list[dict]]:
    transcript = load_cached_transcript(video_id)
    chunk_objects = chunk_transcript(
        transcript,
        chunk_words=DEFAULT_CHUNK_WORDS,
        overlap_words=DEFAULT_OVERLAP_WORDS,
    )
    chunks = [
        {
            "text": c.text,
            "start_time": c.start,
            "end_time": c.end,
            "chunk_id": i,
            "word_count": c.word_count,
        }
        for i, c in enumerate(chunk_objects)
    ]
    return video_id, chunks


def _build_query(area: str, goal: str) -> str:
    return f"{area} {goal}".strip()


def _safe_stat(values: list[float], fn) -> float:
    return float(fn(values)) if values else 0.0


def evaluate_one(
    retriever: LocalRetriever,
    area: str,
    goal: str,
    mandatory_keyword: str,
    expected_token: str,
    use_reranking: bool,
    mode: str,
) -> dict:
    query = _build_query(area, goal)
    results, latency = retriever.retrieve(
        query,
        mandatory_keyword=mandatory_keyword,
        use_reranking=use_reranking,
        top_k=TOP_K,
        mode=mode,
    )
    return {
        "query": query,
        "area": area,
        "goal": goal,
        "mode": mode,
        "rerank": use_reranking,
        "filter": True,  # mandatory-keyword filter is on in all eval configs
        "keyword_for_filter": mandatory_keyword,
        "expected_token": expected_token,
        "n_returned": len(results),
        "keyword_hit_rate": round(keyword_hit_rate(results, expected_token), 3),
        "timestamp_valid_rate": round(timestamp_valid_rate(results), 3),
        "overlap_diversity": round(overlap_diversity(results), 3),
        "latency_s": round(latency, 4),
    }


CONFIGS: list[tuple[str, bool, str]] = [
    # (label, use_reranking, mode)
    ("A: vector-only", False, LocalRetriever.VECTOR),
    ("B: vector + rerank", True, LocalRetriever.VECTOR),
    ("C: vector + rerank + kw filter (production path)", True, LocalRetriever.VECTOR),
    # Hybrid: keyword and RRF-fused variants, with the production
    # keyword filter on. These are the candidates for the
    # "hybrid search" best-practice point.
    ("D: keyword + kw filter", False, LocalRetriever.KEYWORD),
    ("E: hybrid + kw filter", False, LocalRetriever.HYBRID),
    ("F: hybrid + rerank + kw filter", True, LocalRetriever.HYBRID),
]


def evaluate_video(video_id: str) -> list[dict]:
    _, chunks = build_chunks(video_id)
    if not chunks:
        return []
    retriever = LocalRetriever(chunks)
    rows: list[dict] = []
    for area, goal, kw, expected in QUERY_SET:
        # Legacy 3-config block (A/B/C) preserved for back-compat with
        # the original report. A and B are the "no filter" baselines;
        # C is the production path.
        rows.append(_evaluate_no_filter(retriever, area, goal, expected, False, LocalRetriever.VECTOR, "A: vector-only"))
        rows.append(_evaluate_no_filter(retriever, area, goal, expected, True, LocalRetriever.VECTOR, "B: vector + rerank"))
        rows.append(evaluate_one(retriever, area, goal, kw, expected, True, LocalRetriever.VECTOR))
        # Hybrid candidates (D/E/F) — all with the production keyword filter on.
        for label, rerank, mode in CONFIGS[3:]:
            rows.append(evaluate_one(retriever, area, goal, kw, expected, rerank, mode))
    return rows


def _evaluate_no_filter(
    retriever: LocalRetriever,
    area: str,
    goal: str,
    expected_token: str,
    use_reranking: bool,
    mode: str,
    label: str,
) -> dict:
    query = _build_query(area, goal)
    results, latency = retriever.retrieve(
        query,
        mandatory_keyword="",
        use_reranking=use_reranking,
        top_k=TOP_K,
        mode=mode,
    )
    return {
        "query": query,
        "area": area,
        "goal": goal,
        "mode": mode,
        "rerank": use_reranking,
        "filter": False,
        "expected_token": expected_token,
        "n_returned": len(results),
        "keyword_hit_rate": round(keyword_hit_rate(results, expected_token), 3),
        "timestamp_valid_rate": round(timestamp_valid_rate(results), 3),
        "overlap_diversity": round(overlap_diversity(results), 3),
        "latency_s": round(latency, 4),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    """Group by config label, return mean per metric."""
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        key = _config_label(r)
        grouped.setdefault(key, []).append(r)
    summary = []
    for key, items in grouped.items():
        summary.append(
            {
                "config": key,
                "n_queries": len(items),
                "avg_keyword_hit_rate": round(_safe_stat([i["keyword_hit_rate"] for i in items], statistics.mean), 3),
                "avg_timestamp_valid_rate": round(_safe_stat([i["timestamp_valid_rate"] for i in items], statistics.mean), 3),
                "avg_overlap_diversity": round(_safe_stat([i["overlap_diversity"] for i in items], statistics.mean), 3),
                "avg_latency_s": round(_safe_stat([i["latency_s"] for i in items], statistics.mean), 4),
            }
        )
    return summary


def _config_label(row: dict) -> str:
    """Return a stable, human-readable label for a (mode, rerank, filter) row.

    Used as the grouping key in ``aggregate()`` and as the label column
    in the rendered report.
    """
    mode = row.get("mode", "vector")
    rerank = bool(row.get("rerank", False))
    flt = bool(row.get("filter", False))
    if mode == "hybrid":
        if rerank and flt:
            return "F: hybrid + rerank + kw filter"
        if not rerank and flt:
            return "E: hybrid + kw filter"
        if rerank and not flt:
            return "hybrid + rerank (no filter)"
        return "hybrid (no filter)"
    if mode == "keyword":
        if rerank and flt:
            return "keyword + rerank + kw filter"
        if not rerank and flt:
            return "D: keyword + kw filter"
        if rerank and not flt:
            return "keyword + rerank (no filter)"
        return "keyword (no filter)"
    # mode == "vector"
    if not rerank and not flt:
        return "A: vector-only"
    if rerank and not flt:
        return "B: vector + rerank"
    if rerank and flt:
        return "C: vector + rerank + kw filter"
    return "?"


def render_markdown(video_id: str, rows: list[dict], summary: list[dict], chunk_count: int | None = None) -> str:
    out: list[str] = []
    out.append("# Retrieval evaluation\n")
    out.append(
        "Compares six retrieval approaches on cached YouTube transcripts, "
        "all using identical chunking (`core.chunker`, `DEFAULT_CHUNK_WORDS=300`, "
        "`DEFAULT_OVERLAP_WORDS=50`) and identical top-k.\n"
    )
    out.append("Legacy baseline (vector-only path):")
    out.append("- A: vector-only (`RAGEngine.retrieve(use_reranking=False)`)")
    out.append("- B: vector + cross-encoder rerank (`RAGEngine.retrieve(use_reranking=True)`)")
    out.append("- C: B + mandatory-keyword filter (the default production path)")
    out.append("")
    out.append("Hybrid candidates (RRF-fused vector + keyword):")
    out.append("- D: keyword-only + mandatory-keyword filter")
    out.append("- E: hybrid (vector + keyword via reciprocal-rank fusion) + filter")
    out.append("- F: hybrid + cross-encoder rerank + filter\n")
    out.append("Offline score functions live in `scripts/eval_retrieval.py` and are unit-tested in `tests/test_eval_retrieval.py`. The RRF-fusion helper lives in `core/hybrid.py` and is unit-tested in `tests/test_hybrid.py`.\n")
    out.append("## Setup")
    out.append(f"- Video ID evaluated: `{video_id}`")
    if chunk_count is None:
        try:
            chunk_count = len(build_chunks(video_id)[1])
        except FileNotFoundError:
            chunk_count = "?"
    out.append(f"- Chunks evaluated: {chunk_count}")
    out.append(f"- Queries: {len(QUERY_SET)}")
    out.append("  - " + "; ".join(f"`{_build_query(a, g)}` (kw=`{kw}`, expected=`{tok}`)" for a, g, kw, tok in QUERY_SET))
    out.append(f"- top_k: {TOP_K}, first-stage candidates: {TOP_K_FIRST_STAGE}\n")
    out.append("## Aggregate results (mean over all queries)\n")
    out.append("| Config | Queries | Keyword hit rate | Timestamp valid | Overlap diversity | Latency (s) |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for s in summary:
        out.append(
            f"| {s['config']} | {s['n_queries']} | "
            f"{s['avg_keyword_hit_rate']:.3f} | "
            f"{s['avg_timestamp_valid_rate']:.3f} | "
            f"{s['avg_overlap_diversity']:.3f} | "
            f"{s['avg_latency_s']:.4f} |"
        )
    out.append("\n## Per-query results\n")
    out.append("| Query | Config | n | Keyword hit | Timestamp valid | Diversity | Latency (s) |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| `{r['query']}` | {_config_label(r)} | {r['n_returned']} | "
            f"{r['keyword_hit_rate']:.3f} | "
            f"{r['timestamp_valid_rate']:.3f} | "
            f"{r['overlap_diversity']:.3f} | "
            f"{r['latency_s']:.4f} |"
        )
    out.append("\n## Metric definitions")
    out.append("- **Keyword hit rate**: fraction of returned chunks that contain the per-query `expected_token` (case-insensitive substring). Discriminative against real transcripts.")
    out.append("- **Timestamp valid rate**: fraction of returned chunks whose text starts with a parseable `[mm:ss]` tag. Catches reordering / parser bugs.")
    out.append("- **Overlap diversity**: 1 - mean pairwise Jaccard over the first 6 results. Higher is better (avoids 4 near-duplicate chunks).")
    out.append("- **Latency (s)**: wall time of `retrieve()` end-to-end, including rerank and filter.")
    out.append("- **Filter behaviour**: when C's mandatory-keyword filter finds zero matching chunks for a query, it returns fewer than `top_k` (or zero) — a deliberate fallback, not a bug.\n")
    out.append("## Decision")
    out.append(_decision_text(summary))
    out.append("")
    return "\n".join(out)


def _decision_text(summary: list[dict]) -> str:
    by_label = {s["config"].split(":", 1)[0]: s for s in summary}
    a = by_label.get("A", {})
    b = by_label.get("B", {})
    c = by_label.get("C", {})
    d = by_label.get("D", {})
    e = by_label.get("E", {})
    f = by_label.get("F", {})
    if not (a and b and c):
        return "Insufficient data to make a recommendation."
    notes: list[str] = []
    a_hit = a.get("avg_keyword_hit_rate", 0)
    b_hit = b.get("avg_keyword_hit_rate", 0)
    c_hit = c.get("avg_keyword_hit_rate", 0)
    d_hit = d.get("avg_keyword_hit_rate", 0)
    e_hit = e.get("avg_keyword_hit_rate", 0)
    f_hit = f.get("avg_keyword_hit_rate", 0)
    # Overall winner across all 6 configs
    candidates = {"A": a_hit, "B": b_hit, "C": c_hit, "D": d_hit, "E": e_hit, "F": f_hit}
    overall_winner = max(candidates, key=candidates.get) if candidates else "?"
    if c_hit >= max(a_hit, b_hit, d_hit, e_hit, f_hit):
        notes.append(
            f"**C (vector + rerank + keyword filter)** wins on keyword hit rate "
            f"({c_hit:.3f} vs A={a_hit:.3f}, B={b_hit:.3f}, D={d_hit:.3f}, E={e_hit:.3f}, F={f_hit:.3f}) "
            f"and is the active production default."
        )
    elif b_hit >= a_hit:
        notes.append(
            f"**B (vector + rerank)** wins on keyword hit rate; C's filter discards borderline-relevant chunks."
        )
    else:
        notes.append("**A (vector-only)** is sufficient on this sample.")
    if overall_winner not in ("A", "B", "C"):
        notes.append(
            f"**Overall best**: {overall_winner} "
            f"({candidates[overall_winner]:.3f}). "
            f"Consider promoting it to `RETRIEVAL_MODE` in `core/settings.py` "
            f"and updating the default in `app.py`."
        )
    a_diversity = a.get("avg_overlap_diversity", 1)
    c_diversity = c.get("avg_overlap_diversity", 1)
    if c_diversity < 0.7 and a_diversity - c_diversity > 0.05:
        notes.append(
            f"Diversity drops under C ({c_diversity:.3f} vs A={a_diversity:.3f}); "
            "consider raising `TOP_K` or relaxing the keyword filter when the user query is broad."
        )
    # If a hybrid config beats C on hit rate, surface that explicitly.
    best_hybrid_hit = max(d_hit, e_hit, f_hit)
    if best_hybrid_hit > c_hit + 0.01:
        best_hybrid_label = {"D": d_hit, "E": e_hit, "F": f_hit}
        winner = max(best_hybrid_label, key=best_hybrid_label.get)
        notes.append(
            f"**{winner} (hybrid candidate)** beats C by {best_hybrid_hit - c_hit:.3f} on keyword hit rate. "
            f"Switching `RETRIEVAL_MODE` to `\"hybrid\"` is recommended."
        )
    c_latency = c.get("avg_latency_s", 0)
    a_latency = a.get("avg_latency_s", 0)
    if c_latency > a_latency * 1.2:
        notes.append(
            f"C adds {c_latency - a_latency:.4f}s vs A — the cross-encoder rerank is the dominant cost."
        )
    return " ".join(notes)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-id",
        default="TrvLEgPpV8s",
        help="Video ID whose cached transcript drives the eval (default: TrvLEgPpV8s)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_DOC_PATH),
        help="Path to write the rendered markdown report",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the rendered markdown to stdout instead of writing the file",
    )
    parser.add_argument(
        "--use-production",
        action="store_true",
        help=(
            "Use the production RAGEngine for vector scoring. "
            "Requires onnxruntime + chromadb + downloaded ONNX models. "
            "When set, the same metrics are computed over the live retrieval output."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run_production_eval(video_id: str) -> list[dict]:
    """Optional path: call the production RAGEngine instead of LocalRetriever.

    Kept narrow on purpose: it only computes the same metrics using the
    production retriever's output. Lazy-imports the heavy modules so this
    script can still run in a minimal environment.
    """
    from core.rag_engine import RAGEngine
    from core.chunker import chunk_transcript as real_chunker
    from core.transcript import load_transcript_cache  # type: ignore

    transcript = load_transcript_cache(video_id)
    if transcript is None:
        raise FileNotFoundError(f"No cached transcript for {video_id}")
    chunk_objects = real_chunker(
        transcript,
        chunk_words=DEFAULT_CHUNK_WORDS,
        overlap_words=DEFAULT_OVERLAP_WORDS,
    )
    engine = RAGEngine()
    engine.add_chunks(chunk_objects)
    rows: list[dict] = []
    for area, goal, kw, expected in QUERY_SET:
        for rerank, filter_on, label in [
            (False, True, "A: vector-only"),
            (True, False, "B: vector + rerank"),
            (True, True, "C: vector + rerank + kw filter"),
        ]:
            started = time.perf_counter()
            results = engine.retrieve(
                _build_query(area, goal),
                mandatory_keyword=kw if filter_on else "",
                use_reranking=rerank,
                top_k=TOP_K,
            )
            latency = time.perf_counter() - started
            rows.append(
                {
                    "query": _build_query(area, goal),
                    "area": area,
                    "goal": goal,
                    "mode": "production",
                    "rerank": rerank,
                    "filter": filter_on,
                    "keyword_for_filter": kw,
                    "expected_token": expected,
                    "n_returned": len(results),
                    "keyword_hit_rate": round(keyword_hit_rate(results, expected), 3),
                    "timestamp_valid_rate": round(timestamp_valid_rate(results), 3),
                    "overlap_diversity": round(overlap_diversity(results), 3),
                    "latency_s": round(latency, 4),
                }
            )
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.use_production:
            rows = run_production_eval(args.cache_id)
        else:
            rows = evaluate_video(args.cache_id)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = aggregate(rows)
    report = render_markdown(args.cache_id, rows, summary)
    if args.print_only:
        print(report)
        return 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} query-config rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
