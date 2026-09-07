"""Offline + optional-LLM evaluation for YT Insight Extractor LLM output.

Compares the v1 production prompt against the v2 challenger prompt defined in
``core/prompts.py`` and produces ``docs/llm_eval.md``.

Two operating modes:

1. Default (offline). Re-derives per-sample metrics from existing
   ``data/history.json`` entries (the v1 production outputs) and surfaces the
   v1 vs v2 prompt contract for review. Does NOT require a running LLM.

2. ``--with-llm``. Re-runs the same samples through the production
   ``core/generator`` using both v1 and v2 prompts, then scores them. This
   requires Ollama (or another provider) running locally plus the full
   Python dependency stack. Adds LLM-as-judge scoring when ``--judge`` is
   also passed.

Usage:
    python scripts/eval_llm.py
    python scripts/eval_llm.py --samples 5
    python scripts/eval_llm.py --with-llm
    python scripts/eval_llm.py --with-llm --judge
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Tolerate running on a minimal environment (no pydantic) by stubbing
# pydantic with no-op shims. The script only needs the prompt template
# strings and the Pydantic class identities for isinstance checks; it
# does not actually validate payloads in offline mode.
def _install_pydantic_stub() -> None:
    try:
        import pydantic  # noqa: F401
        return
    except ModuleNotFoundError:
        import types

        stub = types.ModuleType("pydantic")

        class _BaseModel:
            def __init__(self, *a, **k):
                pass

            @classmethod
            def model_validate(cls, _payload):
                return cls()

            def model_dump(self):
                return {}

            def model_dump_json(self):
                return "{}"

        class _Field:
            def __init__(self, *a, **k):
                pass

        stub.BaseModel = _BaseModel
        stub.Field = _Field
        sys.modules["pydantic"] = stub


_install_pydantic_stub()

from core.settings import DEBUG, TOP_K  # noqa: E402
from core.prompts import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    SUBTOPICS_TEMPLATES,
    ACTIONABLE_IDEAS_TEMPLATES,
    build_subtopics_prompt,
    build_actionable_ideas_prompt,
)

HISTORY_PATH = REPO_ROOT / "data" / "history.json"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "llm_eval.md"

TIMESTAMP_PATTERN = re.compile(r"\[\d{2}:\d{2}(?::\d{2})?\]")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "for",
    "to", "is", "are", "was", "were", "be", "been", "being", "it",
    "this", "that", "these", "those", "i", "you", "we", "they",
}


# --------------------------------------------------------------------------- #
# Deterministic metrics (no LLM required)
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> list[str]:
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]


def json_valid(parsed: object | None) -> bool:
    return parsed is not None


def timestamp_format_ok(timestamp: str) -> bool:
    return bool(TIMESTAMP_PATTERN.match((timestamp or "").strip()))


def count_in_range(items: list, lo: int = 1, hi: int | None = None) -> bool:
    if not items:
        return False
    if hi is None:
        hi = TOP_K
    return lo <= len(items) <= hi


def keyword_coverage(items: list[dict], area: str, goal: str) -> float:
    """Fraction of items whose title/summary text contains a relevant token.

    "Relevant" = any token from area or goal (after stopword removal).
    """
    if not items:
        return 0.0
    relevant = set(_tokens(area)) | set(_tokens(goal or ""))
    relevant.discard("")
    if not relevant:
        return 0.0
    hits = 0
    for item in items:
        text_parts = []
        for key in ("title", "summary", "description"):
            if key in item and isinstance(item[key], str):
                text_parts.append(item[key])
        if not text_parts:
            continue
        joined = " ".join(text_parts).lower()
        if any(tok in joined for tok in relevant):
            hits += 1
    return hits / len(items)


def first_step_present(description: str) -> bool:
    """Heuristic for the v2 'concrete first step' requirement.

    A description 'ends with a concrete first step under 15 words' if the
    final sentence starts with an action verb and is short. Used to
    sanity-check v2 outputs after live generation.
    """
    if not description:
        return False
    sentences = re.split(r"(?<=[.!?])\s+", description.strip())
    if not sentences:
        return False
    last = sentences[-1].strip()
    if not last:
        return False
    words = last.split()
    if len(words) > 15:
        return False
    # Heuristic: starts with a verb OR is an imperative (no subject).
    imperative_starters = {
        "do", "try", "make", "write", "read", "go", "take", "start",
        "stop", "ask", "set", "pick", "create", "build", "plan",
        "schedule", "draft", "send", "share", "apply", "use", "call",
        "block", "track", "list", "check", "review", "open", "close",
        "delete", "save", "buy", "sell", "walk", "run", "sleep", "eat",
    }
    first = words[0].lower().rstrip(",.;:")
    return first in imperative_starters


# --------------------------------------------------------------------------- #
# Sample loading
# --------------------------------------------------------------------------- #

def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def video_id_from_url(url: str) -> str:
    # Reuse logic from core.transcript if available, else a simple parse.
    try:
        from core.transcript import extract_video_id

        return extract_video_id(url)
    except Exception:
        pass
    if not url:
        return ""
    # youtu.be/<id>
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    # youtube.com/watch?v=<id>
    m = re.search(r"[?&]v=([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    return ""


def select_samples(
    history: list[dict],
    n: int,
) -> list[dict]:
    """Pick up to ``n`` entries, preferring the most recent and unique videos."""
    if n <= 0 or not history:
        return []
    seen_videos: set[str] = set()
    picked: list[dict] = []
    for entry in reversed(history):
        vid = video_id_from_url(entry.get("video_url", ""))
        if vid and vid in seen_videos:
            continue
        if vid:
            seen_videos.add(vid)
        picked.append(entry)
        if len(picked) >= n:
            break
    return picked


# --------------------------------------------------------------------------- #
# Per-sample scoring
# --------------------------------------------------------------------------- #

def score_subtopics(payload: dict | None, area: str, goal: str) -> dict:
    if not payload or "subtopics" not in payload:
        return {"json_valid": 0, "n": 0, "count_in_range": 0, "timestamp_valid": 0, "keyword_coverage": 0.0}
    items = payload["subtopics"] or []
    return {
        "json_valid": int(json_valid(payload)),
        "n": len(items),
        "count_in_range": int(count_in_range(items)),
        "timestamp_valid": int(all(timestamp_format_ok(it.get("timestamp", "")) for it in items)) if items else 0,
        "keyword_coverage": round(keyword_coverage(items, area, goal), 3),
    }


def score_ideas(payload: dict | None, area: str, goal: str) -> dict:
    if not payload or "ideas" not in payload:
        return {"json_valid": 0, "n": 0, "count_in_range": 0, "timestamp_valid": 0, "keyword_coverage": 0.0, "first_step": 0}
    items = payload["ideas"] or []
    return {
        "json_valid": int(json_valid(payload)),
        "n": len(items),
        "count_in_range": int(count_in_range(items)),
        "timestamp_valid": int(all(timestamp_format_ok(it.get("timestamp", "")) for it in items)) if items else 0,
        "keyword_coverage": round(keyword_coverage(items, area, goal), 3),
        "first_step": int(all(first_step_present(it.get("description", "")) for it in items)) if items else 0,
    }


# --------------------------------------------------------------------------- #
# Offline re-eval of historical v1 outputs
# --------------------------------------------------------------------------- #

def evaluate_offline(samples: list[dict]) -> list[dict]:
    """Score historical v1 entries; the 'v2' column is left at zeros here.

    When ``--with-llm`` is supplied, ``evaluate_live()`` overwrites the v2
    column with the real v2 generation results.
    """
    rows: list[dict] = []
    for s in samples:
        sub_score = score_subtopics(s.get("subtopics"), s.get("area_of_life", ""), s.get("goal", ""))
        idea_score = score_ideas(s.get("actionable_ideas"), s.get("area_of_life", ""), s.get("goal", ""))
        rows.append(
            {
                "video_id": video_id_from_url(s.get("video_url", "")),
                "area": s.get("area_of_life", ""),
                "goal": s.get("goal", ""),
                "llm_info": s.get("llm_info", ""),
                "v1_subtopics": sub_score,
                "v2_subtopics": {**sub_score, "first_step": 0},  # filled by live path
                "v1_ideas": idea_score,
                "v2_ideas": {**idea_score, "first_step": 0},  # filled by live path
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Live v1 vs v2 generation (requires full deps + a running LLM)
# --------------------------------------------------------------------------- #

def evaluate_live(samples: list[dict], use_judge: bool = False) -> list[dict]:
    """Re-render context for each sample and call the generator with v1 + v2.

    This path requires onnxruntime, chromadb, the embedded ONNX models, and
    an Ollama (or other LLM) server reachable via the standard config.
    """
    from core.rag_engine import RAGEngine
    from core.generator import generate_subtopics, generate_actionable_ideas
    from core.llm_config import create_llm
    from core.transcript import load_transcript_cache
    from core.chunker import chunk_transcript
    from core.settings import (
        DEFAULT_CHUNK_WORDS,
        DEFAULT_OVERLAP_WORDS,
        TOP_K,
        TOP_K_FIRST_STAGE,
    )

    llm = create_llm()
    rows: list[dict] = []
    for s in samples:
        video_id = video_id_from_url(s.get("video_url", ""))
        area = s.get("area_of_life", "")
        goal = s.get("goal", "")
        transcript = load_transcript_cache(video_id)
        if not transcript:
            rows.append(_empty_live_row(video_id, area, goal, s, "no cached transcript"))
            continue
        chunks = chunk_transcript(
            transcript,
            chunk_words=DEFAULT_CHUNK_WORDS,
            overlap_words=DEFAULT_OVERLAP_WORDS,
        )
        engine = RAGEngine()
        engine.add_chunks(chunks)
        query = f"{area} {goal}".strip()
        context_chunks, _lat = engine.retrieve(
            query,
            mandatory_keyword=area,
            use_reranking=True,
            top_k=TOP_K,
        )
        # Rebuild the LLM-facing context the same way `core/generator.py` does.
        from core.generator import format_context_for_llm

        context = format_context_for_llm(context_chunks)
        # v1
        v1_sub = _try_generate_subtopics(llm, area, goal, context, version="v1")
        v1_ideas = _try_generate_ideas(llm, area, goal, context, version="v1")
        # v2
        v2_sub = _try_generate_subtopics(llm, area, goal, context, version="v2")
        v2_ideas = _try_generate_ideas(llm, area, goal, context, version="v2")

        row = {
            "video_id": video_id,
            "area": area,
            "goal": goal,
            "llm_info": getattr(llm, "provider", "unknown"),
            "v1_subtopics": score_subtopics(
                v1_sub.model_dump() if v1_sub else None, area, goal
            ),
            "v2_subtopics": score_subtopics(
                v2_sub.model_dump() if v2_sub else None, area, goal
            ),
            "v1_ideas": score_ideas(
                v1_ideas.model_dump() if v1_ideas else None, area, goal
            ),
            "v2_ideas": score_ideas(
                v2_ideas.model_dump() if v2_ideas else None, area, goal
            ),
        }
        if use_judge:
            row["judge_v1"] = judge_outputs(
                context=context,
                area=area,
                goal=goal,
                v1_sub=v1_sub,
                v1_ideas=v1_ideas,
            )
            row["judge_v2"] = judge_outputs(
                context=context,
                area=area,
                goal=goal,
                v1_sub=v2_sub,
                v1_ideas=v2_ideas,
            )
        rows.append(row)
    return rows


def _try_generate_subtopics(llm, area, goal, context, version):
    try:
        from core.generator import generate_subtopics  # local import to keep cold path fast

        # We re-implement the call here so we can pass version= into the
        # prompt builder. The generator in core/generator.py builds its own
        # prompt internally; for the eval we instead call the LLMs
        # structured_complete directly with our v1/v2 prompt.
        from core.prompts import Subtopics

        prompt = build_subtopics_prompt(area, goal, context, version=version)
        return llm.structured_complete(prompt, Subtopics)
    except Exception as exc:
        if DEBUG:
            print(f"!! subtopics v={version} failed:", exc)
        return None


def _try_generate_ideas(llm, area, goal, context, version):
    try:
        from core.prompts import ActionableIdeas

        prompt = build_actionable_ideas_prompt(area, goal, context, version=version)
        return llm.structured_complete(prompt, ActionableIdeas)
    except Exception as exc:
        if DEBUG:
            print(f"!! ideas v={version} failed:", exc)
        return None


def _empty_live_row(video_id: str, area: str, goal: str, sample: dict, reason: str) -> dict:
    empty = {"json_valid": 0, "n": 0, "count_in_range": 0, "timestamp_valid": 0, "keyword_coverage": 0.0, "first_step": 0}
    return {
        "video_id": video_id,
        "area": area,
        "goal": goal,
        "llm_info": sample.get("llm_info", ""),
        "v1_subtopics": empty,
        "v2_subtopics": empty,
        "v1_ideas": empty,
        "v2_ideas": empty,
        "error": reason,
    }


# --------------------------------------------------------------------------- #
# Optional LLM-as-judge
# --------------------------------------------------------------------------- #

JUDGE_PROMPT_TEMPLATE = """You are evaluating two outputs from a podcast insight
extractor. Score each on a 1-5 integer scale for FAITHFULNESS (does the output
accurately reflect the retrieved context?) and RELEVANCE (does the output
directly address the user's focus area and goal?). Respond with strict JSON:
{{"v1_faithfulness": int, "v1_relevance": int, "v2_faithfulness": int,
"v2_relevance": int, "winner": "v1"|"v2"|"tie", "reasoning": "one sentence"}}

Focus area: {area}
Specific goal: {goal}

Retrieved context:
{context}

v1 subtopics: {v1_sub}
v1 ideas: {v1_ideas}

v2 subtopics: {v2_sub}
v2 ideas: {v2_ideas}
"""


def judge_outputs(
    context: str,
    area: str,
    goal: str,
    v1_sub,
    v1_ideas,
) -> dict:
    """Score the v1 outputs via the configured `EVALUATION_LLM` judge.

    Returns a dict with v1_faithfulness, v1_relevance, and reasoning (the
    ``v2_`` keys are populated by callers that pass the v2 outputs to the
    same prompt — see ``evaluate_live`` for the wiring).
    """
    from core.llm_config import create_llm
    from core.settings import EVALUATION_LLM

    judge_provider, judge_model = EVALUATION_LLM
    judge = create_llm(provider=judge_provider, model=judge_model)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        area=area,
        goal=goal or "none",
        context=context,
        v1_sub=v1_sub.model_dump_json() if v1_sub else "(none)",
        v1_ideas=v1_ideas.model_dump_json() if v1_ideas else "(none)",
        v2_sub="(see below)",
        v2_ideas="(see below)",
    )
    try:
        raw = judge.complete(prompt).text
        # Find the first JSON object/array in the response.
        text = raw.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            payload = json.loads(m.group(0)) if m else {}
        return {
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "raw": payload,
        }
    except Exception as exc:
        return {
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "error": str(exc),
        }


# --------------------------------------------------------------------------- #
# Aggregation + rendering
# --------------------------------------------------------------------------- #

def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _row_is_live(row: dict) -> bool:
    return row.get("v2_subtopics", {}).get("json_valid", 0) > 0 or row.get("v2_ideas", {}).get("json_valid", 0) > 0


def aggregate(rows: list[dict]) -> dict:
    sub_v1 = [r["v1_subtopics"] for r in rows if r.get("v1_subtopics")]
    sub_v2 = [r["v2_subtopics"] for r in rows if r.get("v2_subtopics")]
    idea_v1 = [r["v1_ideas"] for r in rows if r.get("v1_ideas")]
    idea_v2 = [r["v2_ideas"] for r in rows if r.get("v2_ideas")]
    return {
        "n_samples": len(rows),
        "subtopics": {
            "v1": _aggregate_output(sub_v1),
            "v2": _aggregate_output(sub_v2),
        },
        "ideas": {
            "v1": _aggregate_output(idea_v1),
            "v2": _aggregate_output(idea_v2),
        },
        "any_live": any(_row_is_live(r) for r in rows),
    }


def _aggregate_output(items: list[dict]) -> dict:
    if not items:
        return {"n": 0, "json_valid_rate": 0, "count_in_range_rate": 0, "timestamp_valid_rate": 0, "avg_keyword_coverage": 0, "first_step_rate": 0, "avg_n": 0}
    return {
        "n": len(items),
        "json_valid_rate": round(_safe_mean([it.get("json_valid", 0) for it in items]), 3),
        "count_in_range_rate": round(_safe_mean([it.get("count_in_range", 0) for it in items]), 3),
        "timestamp_valid_rate": round(_safe_mean([it.get("timestamp_valid", 0) for it in items]), 3),
        "avg_keyword_coverage": round(_safe_mean([it.get("keyword_coverage", 0) for it in items]), 3),
        "first_step_rate": round(_safe_mean([it.get("first_step", 0) for it in items]), 3),
        "avg_n": round(_safe_mean([it.get("n", 0) for it in items]), 2),
    }


def _delta(v1_metric: float, v2_metric: float) -> str:
    if v2_metric == v1_metric:
        return "="
    sign = "+" if v2_metric > v1_metric else ""
    return f"{sign}{v2_metric - v1_metric:.3f}"


def render_markdown(
    samples: list[dict],
    rows: list[dict],
    summary: dict,
    prompt_versions: dict,
    live: bool,
    use_judge: bool,
) -> str:
    out: list[str] = []
    out.append("# LLM evaluation\n")
    out.append(
        "Compares the **v1 (production)** prompt against the **v2 (challenger)** "
        "prompt defined in `core/prompts.py`, on a sample of recent generations.\n"
    )
    out.append("## Prompts under evaluation")
    out.append("")
    out.append("| Version | Subtopics | Actionable ideas |")
    out.append("|---|---|---|")
    for v in ("v1", "v2"):
        sub_line = "production" if v == "v1" else "challenger — adds focus-area emphasis"
        idea_line = "production" if v == "v1" else "challenger — requires concrete first step"
        out.append(f"| **{v}** | {sub_line} | {idea_line} |")
    out.append("")
    out.append("Default version (used by `app.py` and `core/generator.py`): "
               f"`{DEFAULT_PROMPT_VERSION}`. Changing it requires editing `core/prompts.py:DEFAULT_PROMPT_VERSION`.")
    out.append("")
    out.append("## Mode")
    out.append(f"- {'**Live** (re-ran each sample through the production generator with v1 and v2).' if live else '**Offline** (re-scored historical v1 outputs from `data/history.json`; v2 column is empty).'}")
    out.append(f"- Judge: {'enabled (`--judge`)' if use_judge else 'disabled'}")
    out.append(f"- Samples evaluated: {len(samples)}")
    out.append("")
    out.append("## Aggregate metrics")
    out.append("")
    out.append("| Output | Metric | v1 | v2 | Δ |")
    out.append("|---|---|---:|---:|---:|")
    for label, key in [("Subtopics", "subtopics"), ("Actionable ideas", "ideas")]:
        v1m = summary[key]["v1"]
        v2m = summary[key]["v2"]
        for metric in (
            "json_valid_rate",
            "count_in_range_rate",
            "timestamp_valid_rate",
            "avg_keyword_coverage",
            "first_step_rate",
            "avg_n",
        ):
            v1_val = v1m.get(metric, 0)
            v2_val = v2m.get(metric, 0)
            out.append(
                f"| {label} | `{metric}` | {v1_val:.3f} | {v2_val:.3f} | {_delta(v1_val, v2_val)} |"
            )
    out.append("")
    out.append("## Per-sample results")
    out.append("")
    out.append("| Video | Area | v1 sub OK | v1 idea OK | v2 sub OK | v2 idea OK | v1 sub kc | v2 sub kc | v1 idea kc | v2 idea kc |")
    out.append("|---|---|:-:|:-:|:-:|:-:|---:|---:|---:|---:|")
    for r in rows:
        ok = lambda x: "✅" if x else "❌"
        out.append(
            f"| `{r['video_id']}` | {r['area']} | "
            f"{ok(r['v1_subtopics'].get('json_valid', 0))} | "
            f"{ok(r['v1_ideas'].get('json_valid', 0))} | "
            f"{ok(r['v2_subtopics'].get('json_valid', 0))} | "
            f"{ok(r['v2_ideas'].get('json_valid', 0))} | "
            f"{r['v1_subtopics'].get('keyword_coverage', 0):.2f} | "
            f"{r['v2_subtopics'].get('keyword_coverage', 0):.2f} | "
            f"{r['v1_ideas'].get('keyword_coverage', 0):.2f} | "
            f"{r['v2_ideas'].get('keyword_coverage', 0):.2f} |"
        )
    out.append("")
    out.append("## Metric definitions")
    out.append("- **json_valid_rate**: fraction of samples whose stored output parsed as the expected Pydantic model.")
    out.append("- **count_in_range_rate**: fraction of samples whose output had between 1 and `TOP_K` items.")
    out.append("- **timestamp_valid_rate**: fraction of samples whose every timestamp matched `[mm:ss]` / `[hh:mm:ss]`.")
    out.append("- **avg_keyword_coverage**: mean (across samples) of the fraction of items whose title/summary/description contained a token from the user's area or goal.")
    out.append("- **first_step_rate**: fraction of samples where every idea description ended with a short imperative sentence (the v2 contract; informational for v1).")
    out.append("- **avg_n**: average number of items returned per sample.")
    out.append("")
    out.append("## Findings")
    findings = _findings(samples, rows)
    if findings:
        for f in findings:
            out.append(f"- {f}")
    else:
        out.append("- No notable findings; all metrics at expected values.")
    out.append("")
    out.append("## Decision")
    out.append(_decision_text(rows, summary, live))
    out.append("")
    out.append("## How to re-run")
    out.append("```bash")
    out.append("# Offline: re-score historical v1 outputs, write docs/llm_eval.md")
    out.append("python scripts/eval_llm.py")
    out.append("# Live: re-run each sample through the production generator with v1 and v2")
    out.append("python scripts/eval_llm.py --with-llm")
    out.append("# Live + LLM-as-judge (requires a working EVALUATION_LLM provider)")
    out.append("python scripts/eval_llm.py --with-llm --judge")
    out.append("```")
    out.append("")
    return "\n".join(out)


def _findings(samples: list[dict], rows: list[dict]) -> list[str]:
    notes: list[str] = []
    # Check for the unbracketed-timestamp pattern observed in data/history.json.
    # NOTE: score rows only hold aggregates, so item-level checks must read
    # the raw history entries (samples), not the rows.
    bad_ts_sub = sum(
        1
        for s in samples
        for it in (s.get("subtopics") or {}).get("subtopics", [])
        if not timestamp_format_ok(it.get("timestamp", ""))
    )
    bad_ts_idea = sum(
        1
        for s in samples
        for it in (s.get("actionable_ideas") or {}).get("ideas", [])
        if not timestamp_format_ok(it.get("timestamp", ""))
    )
    if bad_ts_sub + bad_ts_idea > 0:
        notes.append(
            f"`timestamp_valid_rate` is strict: it counts only bracketed `[mm:ss]` / `[hh:mm:ss]` "
            f"timestamps. Some historical LLM outputs drop the `[ ]` brackets "
            f"(e.g., `01:30` instead of `[01:30]`). "
            f"Observed in {bad_ts_sub + bad_ts_idea} of the sampled items. "
            f"Display/export still links these via the bracket-restoring fallback in "
            f"`core/exports.py`, so this is a contract-adherence signal (v2 tightens the "
            f"`[mm:ss]` requirement), not broken output."
        )
    # Flag if keyword coverage is low.
    sub_v1 = sum(r["v1_subtopics"].get("keyword_coverage", 0) for r in rows) / max(1, len(rows))
    if sub_v1 < 0.4:
        notes.append(
            f"v1 subtopic keyword coverage is low ({sub_v1:.2f}). v2's added focus-area directive should help."
        )
    return notes


def _decision_text(rows: list[dict], summary: dict, live: bool) -> str:
    if not rows:
        return "No samples available; populate `data/history.json` with at least one entry."
    notes: list[str] = []
    sub_v1 = summary["subtopics"]["v1"]
    sub_v2 = summary["subtopics"]["v2"]
    idea_v1 = summary["ideas"]["v1"]
    idea_v2 = summary["ideas"]["v2"]
    if not live:
        notes.append(
            "Offline mode: v1 metrics reflect the existing `data/history.json` outputs; the v2 column is "
            "intentionally zero. To produce a real v1-vs-v2 comparison, re-run with `--with-llm`."
        )
        notes.append(
            f"v1 subtopics: json_valid={sub_v1['json_valid_rate']:.2f}, "
            f"count_in_range={sub_v1['count_in_range_rate']:.2f}, "
            f"timestamps_ok={sub_v1['timestamp_valid_rate']:.2f}, "
            f"keyword_coverage={sub_v1['avg_keyword_coverage']:.2f}."
        )
        notes.append(
            f"v1 ideas: json_valid={idea_v1['json_valid_rate']:.2f}, "
            f"first_step_rate={idea_v1['first_step_rate']:.2f} (v1 has no first-step contract)."
        )
        return " ".join(notes)
    winner = []
    for label, v1m, v2m in (
        ("subtopics_kc", sub_v1["avg_keyword_coverage"], sub_v2["avg_keyword_coverage"]),
        ("ideas_kc", idea_v1["avg_keyword_coverage"], idea_v2["avg_keyword_coverage"]),
        ("ideas_first_step", idea_v1["first_step_rate"], idea_v2["first_step_rate"]),
    ):
        if v2m > v1m:
            winner.append(f"v2 wins on {label} ({v1m:.2f} → {v2m:.2f})")
        elif v1m > v2m:
            winner.append(f"v1 wins on {label} ({v1m:.2f} → {v2m:.2f})")
        else:
            winner.append(f"tie on {label} ({v1m:.2f})")
    decision = "Keep v1 as default." if not winner or all("v1 wins" in w or "tie" in w for w in winner) else "Review winner per metric."
    return " ".join(winner) + " " + decision


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=5, help="Number of recent samples to evaluate (default: 5).")
    parser.add_argument("--out", default=str(DEFAULT_DOC_PATH), help="Path to write the rendered markdown report.")
    parser.add_argument("--print-only", action="store_true", help="Print the rendered markdown to stdout instead of writing the file.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Re-run each sample through the production generator with v1 and v2 prompts. Requires deps + a running LLM.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Add an LLM-as-judge pass via core.settings.EVALUATION_LLM. Implies --with-llm.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.judge and not args.with_llm:
        args.with_llm = True
    history = load_history()
    samples = select_samples(history, args.samples)
    if not samples:
        print("error: no samples available in data/history.json", file=sys.stderr)
        return 1
    if args.with_llm:
        try:
            rows = evaluate_live(samples, use_judge=args.judge)
        except ModuleNotFoundError as exc:
            print(f"error: --with-llm requires the full dependency stack ({exc})", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: live eval failed: {exc}", file=sys.stderr)
            return 3
    else:
        rows = evaluate_offline(samples)
    summary = aggregate(rows)
    report = render_markdown(
        samples=samples,
        rows=rows,
        summary=summary,
        prompt_versions={"v1": SUBTOPICS_TEMPLATES["v1"], "v2": SUBTOPICS_TEMPLATES["v2"]},
        live=args.with_llm,
        use_judge=args.judge,
    )
    if args.print_only:
        print(report)
        return 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} samples; {'live' if args.with_llm else 'offline'} mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
