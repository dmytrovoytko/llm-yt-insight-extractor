# LLM evaluation

Compares the **v1 (production)** prompt against the **v2 (challenger)** prompt defined in `core/prompts.py`, on a sample of recent generations.

## Prompts under evaluation

| Version | Subtopics | Actionable ideas |
|---|---|---|
| **v1** | production | production |
| **v2** | challenger — adds focus-area emphasis | challenger — requires concrete first step |

Default version (used by `app.py` and `core/generator.py`): `v1`. Changing it requires editing `core/prompts.py:DEFAULT_PROMPT_VERSION`.

## Mode
- **Offline** (re-scored historical v1 outputs from `data/history.json`; v2 column is empty).
- Judge: disabled
- Samples evaluated: 3

## Aggregate metrics

| Output | Metric | v1 | v2 | Δ |
|---|---|---:|---:|---:|
| Subtopics | `json_valid_rate` | 1.000 | 1.000 | = |
| Subtopics | `count_in_range_rate` | 1.000 | 1.000 | = |
| Subtopics | `timestamp_valid_rate` | 0.333 | 0.333 | = |
| Subtopics | `avg_keyword_coverage` | 0.750 | 0.750 | = |
| Subtopics | `first_step_rate` | 0.000 | 0.000 | = |
| Subtopics | `avg_n` | 4.000 | 4.000 | = |
| Actionable ideas | `json_valid_rate` | 1.000 | 1.000 | = |
| Actionable ideas | `count_in_range_rate` | 1.000 | 1.000 | = |
| Actionable ideas | `timestamp_valid_rate` | 0.000 | 0.000 | = |
| Actionable ideas | `avg_keyword_coverage` | 0.556 | 0.556 | = |
| Actionable ideas | `first_step_rate` | 0.000 | 0.000 | = |
| Actionable ideas | `avg_n` | 3.000 | 3.000 | = |

## Per-sample results

| Video | Area | v1 sub OK | v1 idea OK | v2 sub OK | v2 idea OK | v1 sub kc | v2 sub kc | v1 idea kc | v2 idea kc |
|---|---|:-:|:-:|:-:|:-:|---:|---:|---:|---:|
| `TrvLEgPpV8s` | Productivity | ✅ | ✅ | ✅ | ✅ | 0.75 | 0.75 | 0.00 | 0.00 |
| `xKvlK7OqZso` | Productivity | ✅ | ✅ | ✅ | ✅ | 0.75 | 0.75 | 0.67 | 0.67 |
| `hFL6qRIJZ_Y` | Learning | ✅ | ✅ | ✅ | ✅ | 0.75 | 0.75 | 1.00 | 1.00 |

## Metric definitions
- **json_valid_rate**: fraction of samples whose stored output parsed as the expected Pydantic model.
- **count_in_range_rate**: fraction of samples whose output had between 1 and `TOP_K` items.
- **timestamp_valid_rate**: fraction of samples whose every timestamp matched `[mm:ss]` / `[hh:mm:ss]`.
- **avg_keyword_coverage**: mean (across samples) of the fraction of items whose title/summary/description contained a token from the user's area or goal.
- **first_step_rate**: fraction of samples where every idea description ended with a short imperative sentence (the v2 contract; informational for v1).
- **avg_n**: average number of items returned per sample.

## Findings
- No notable findings; all metrics at expected values.

## Decision
Offline mode: v1 metrics reflect the existing `data/history.json` outputs; the v2 column is intentionally zero. To produce a real v1-vs-v2 comparison, re-run with `--with-llm`. v1 subtopics: json_valid=1.00, count_in_range=1.00, timestamps_ok=0.33, keyword_coverage=0.75. v1 ideas: json_valid=1.00, first_step_rate=0.00 (v1 has no first-step contract).

## How to re-run
```bash
# Offline: re-score historical v1 outputs, write docs/llm_eval.md
python scripts/eval_llm.py
# Live: re-run each sample through the production generator with v1 and v2
python scripts/eval_llm.py --with-llm
# Live + LLM-as-judge (requires a working EVALUATION_LLM provider)
python scripts/eval_llm.py --with-llm --judge
```
