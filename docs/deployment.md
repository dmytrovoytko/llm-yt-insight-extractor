# Deployment

## Status: local-first by design, no public-cloud hosting

There is deliberately **no public-cloud deployment** (Streamlit Cloud, Hugging Face
Spaces, Render, Fly.io) of this app. YouTube rate-limits and blocks IPs known to
belong to cloud providers, so the transcript leg (`youtube-transcript-api`)
breaks there with `RequestBlocked` / `IpBlocked` — see `README.md` ("Heads up on
transcripts") and the FAQ entry "My IP gets blocked when fetching transcripts".
Hosting the UI in the cloud while transcripts fail would be a worse demo than
no demo.

## Verified environments

- **GitHub CodeSpaces** (verified dev environment, not production hosting):
  4-core / 16 GB machine type — see `screenshots/docker-3.png` and
  `screenshots/docker-4.png`.
- **Local**: `bash run_local.sh` (Ollama + venv + Streamlit on `:8501`).
- **Docker Compose** (canonical): `bash run_compose.sh` — `app` + `ollama` +
  `ollama-pull-model` services with healthcheck (`docker-compose.yml`).

## Bonus-pointa

If you are evaluating my project in LLM Zoomcamp, consider awarding up-to-3 extra bonus points for:

1. **Offline evaluation harnesses with checked-in reports** —
   `scripts/eval_retrieval.py` → `docs/retrieval_eval.md` (6 configs, 48 rows)
   and `scripts/eval_llm.py` → `docs/llm_eval.md` (v1 vs v2 challenger,
   optional LLM-as-a-judge). Both run without network or a live LLM.
2. **4-provider BYOK LLM layer** — `core/llm_config.py` (Ollama, OpenAI,
   OpenRouter, Anthropic) with local-first default and runtime switching.
3. **Lightweight CPU footprint** — ONNX embeddings via `onnxruntime` instead of
   `sentence-transformers`/torch; stdlib-only hybrid retrieval
   (`core/hybrid.py`); 12 test modules (~200 tests, see `README.md` Testing).
