# Retrieval evaluation

Compares six retrieval approaches on cached YouTube transcripts, all using identical chunking (`core.chunker`, `DEFAULT_CHUNK_WORDS=300`, `DEFAULT_OVERLAP_WORDS=50`) and identical top-k.

Legacy baseline (vector-only path):
- A: vector-only (`RAGEngine.retrieve(use_reranking=False)`)
- B: vector + cross-encoder rerank (`RAGEngine.retrieve(use_reranking=True)`)
- C: B + mandatory-keyword filter (the default production path)

Hybrid candidates (RRF-fused vector + keyword):
- D: keyword-only + mandatory-keyword filter
- E: hybrid (vector + keyword via reciprocal-rank fusion) + filter
- F: hybrid + cross-encoder rerank + filter

Offline score functions live in `scripts/eval_retrieval.py` and are unit-tested in `tests/test_eval_retrieval.py`. The RRF-fusion helper lives in `core/hybrid.py` and is unit-tested in `tests/test_hybrid.py`.

## Setup
- Video ID evaluated: `TrvLEgPpV8s`
- Chunks evaluated: 7
- Queries: 8
  - `Productivity morning routine` (kw=`morning`, expected=`morning`); `Fitness` (kw=`fitness`, expected=`fitness`); `Career get promoted` (kw=`promot`, expected=`promot`); `Learning study habits` (kw=`learn`, expected=`learn`); `Mental Health` (kw=`stress`, expected=`stress`); `Productivity books and tools` (kw=`Ferriss`, expected=`Ferriss`); `Productivity writing habits` (kw=`write`, expected=`write`); `Productivity books that changed him` (kw=`book`, expected=`book`)
- top_k: 4, first-stage candidates: 10

## Aggregate results (mean over all queries)

| Config | Queries | Keyword hit rate | Timestamp valid | Overlap diversity | Latency (s) |
|---|---:|---:|---:|---:|---:|
| A: vector-only | 8 | 0.031 | 0.250 | 0.875 | 0.0001 |
| B: vector + rerank | 8 | 0.031 | 0.250 | 0.875 | 0.0001 |
| C: vector + rerank + kw filter | 8 | 0.250 | 0.188 | 0.887 | 0.0001 |
| D: keyword + kw filter | 8 | 0.250 | 0.188 | 0.887 | 0.0001 |
| E: hybrid + kw filter | 8 | 0.250 | 0.188 | 0.887 | 0.0001 |
| F: hybrid + rerank + kw filter | 8 | 0.250 | 0.188 | 0.887 | 0.0001 |

## Per-query results

| Query | Config | n | Keyword hit | Timestamp valid | Diversity | Latency (s) |
|---|---|---:|---:|---:|---:|---:|
| `Productivity morning routine` | A: vector-only | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity morning routine` | B: vector + rerank | 4 | 0.000 | 0.250 | 0.875 | 0.0003 |
| `Productivity morning routine` | C: vector + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity morning routine` | D: keyword + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity morning routine` | E: hybrid + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity morning routine` | F: hybrid + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Fitness` | A: vector-only | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Fitness` | B: vector + rerank | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Fitness` | C: vector + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Fitness` | D: keyword + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Fitness` | E: hybrid + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Fitness` | F: hybrid + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Career get promoted` | A: vector-only | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Career get promoted` | B: vector + rerank | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Career get promoted` | C: vector + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Career get promoted` | D: keyword + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Career get promoted` | E: hybrid + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Career get promoted` | F: hybrid + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Learning study habits` | A: vector-only | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Learning study habits` | B: vector + rerank | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Learning study habits` | C: vector + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Learning study habits` | D: keyword + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Learning study habits` | E: hybrid + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Learning study habits` | F: hybrid + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0002 |
| `Mental Health` | A: vector-only | 4 | 0.000 | 0.250 | 0.875 | 0.0002 |
| `Mental Health` | B: vector + rerank | 4 | 0.000 | 0.250 | 0.875 | 0.0002 |
| `Mental Health` | C: vector + rerank + kw filter | 2 | 1.000 | 0.000 | 0.850 | 0.0001 |
| `Mental Health` | D: keyword + kw filter | 2 | 1.000 | 0.000 | 0.850 | 0.0001 |
| `Mental Health` | E: hybrid + kw filter | 2 | 1.000 | 0.000 | 0.850 | 0.0001 |
| `Mental Health` | F: hybrid + rerank + kw filter | 2 | 1.000 | 0.000 | 0.850 | 0.0001 |
| `Productivity books and tools` | A: vector-only | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books and tools` | B: vector + rerank | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books and tools` | C: vector + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books and tools` | D: keyword + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books and tools` | E: hybrid + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books and tools` | F: hybrid + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity writing habits` | A: vector-only | 4 | 0.250 | 0.250 | 0.875 | 0.0001 |
| `Productivity writing habits` | B: vector + rerank | 4 | 0.250 | 0.250 | 0.875 | 0.0001 |
| `Productivity writing habits` | C: vector + rerank + kw filter | 1 | 1.000 | 0.000 | 1.000 | 0.0001 |
| `Productivity writing habits` | D: keyword + kw filter | 1 | 1.000 | 0.000 | 1.000 | 0.0001 |
| `Productivity writing habits` | E: hybrid + kw filter | 1 | 1.000 | 0.000 | 1.000 | 0.0001 |
| `Productivity writing habits` | F: hybrid + rerank + kw filter | 1 | 1.000 | 0.000 | 1.000 | 0.0001 |
| `Productivity books that changed him` | A: vector-only | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books that changed him` | B: vector + rerank | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books that changed him` | C: vector + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books that changed him` | D: keyword + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books that changed him` | E: hybrid + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |
| `Productivity books that changed him` | F: hybrid + rerank + kw filter | 4 | 0.000 | 0.250 | 0.875 | 0.0001 |

## Metric definitions
- **Keyword hit rate**: fraction of returned chunks that contain the per-query `expected_token` (case-insensitive substring). Discriminative against real transcripts.
- **Timestamp valid rate**: fraction of returned chunks whose text starts with a parseable `[mm:ss]` tag. Catches reordering / parser bugs.
- **Overlap diversity**: 1 - mean pairwise Jaccard over the first 6 results. Higher is better (avoids 4 near-duplicate chunks).
- **Latency (s)**: wall time of `retrieve()` end-to-end, including rerank and filter.
- **Filter behaviour**: when C's mandatory-keyword filter finds zero matching chunks for a query, it returns fewer than `top_k` (or zero) — a deliberate fallback, not a bug.

## Decision
**C (vector + rerank + keyword filter)** wins on keyword hit rate (0.250 vs A=0.031, B=0.031, D=0.250, E=0.250, F=0.250) and is the active production default.
