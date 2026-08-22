# YT Insight Extractor

**Focus your learning. Extract exactly what you need from top podcasts and turn it into your personal growth plan.**

## 📌 Intro

YT Insight Extractor is an AI-powered local assistant built in Python and Streamlit. It transforms long-form YouTube interviews (like Tim Ferriss, Andrew Huberman, or Diary of a CEO) into structured, actionable knowledge. Instead of spending hours listening to a 3-hour podcast to find advice relevant to your specific goals, this app distills the video into targeted subtopics, summaries, and a concrete action plan.

## ⚠️ Problem Statement

There is an abundance of high-quality, long-form educational content on YouTube. However, extracting practical, personalized value from 2-to-3-hour interviews is highly inefficient. Listeners often have specific areas of life they want to improve, but are forced to consume the entire video, taking extensive notes to find the few actionable insights that apply to them.

## 💡 Solution

This assistant uses a Retrieval-Augmented Generation (RAG) pipeline driven by local LLMs (via Ollama). By providing a YouTube link and a specific personal development goal, the app extracts the transcript, chunks it with timestamp overlaps, and saves it into a vector database. A streamlined LlamaIndex pipeline (simple chain, no agents) then queries this vector store against the user's area of life and goal to generate a structured summary of subtopics and a formatted list of actionable ideas—with clickable video timestamps.

> YouTube Transcript API: Unfortunately, YouTube has started blocking most IPs that are known to belong to cloud providers (like AWS, Google Cloud Platform, Azure, etc.), which means you will most likely run into RequestBlocked or IpBlocked exceptions when deploying your code to any cloud solutions. Same can happen to the IP of your self-hosted solution, if you are doing too many requests. 

## ✨ Features

-   **💡 Insights page:** staged processing UI (extract transcript → chunk & vectorize → RAG & summarize) with per-step status
-   Subtopic summaries and actionable ideas with **clickable YouTube timestamps** (`[mm:ss](https://youtu.be/...)`)
-   **🕘 History:** every run is saved locally (JSON) with 👍/👎 feedback; view, export, or delete past results
-   **📊 Report Dashboard:** usage KPIs — total runs, goals set, avg/min/max processing time, distributions by area of life, LLM, and feedback
-   **⚙️ Configuration:** switch LLM provider/model at runtime (Ollama, OpenAI, Anthropic, OpenRouter — BYOK)
-   Optional **cross-encoder re-ranking** of retrieved chunks and a strict **mandatory keyword** content filter
-   Export results as **Markdown** or **JSON**

## 🏗️ Solution Architecture

```
YouTube URL
   │
   ▼
[1] Transcript extraction ── core/transcript.py
   │                         (youtube-transcript-api + pytubefix title; local cache in data/.transcript_cache/)
   ▼
[2] Timestamp chunking ───── core/chunker.py
   │                         ([mm:ss]-prefixed chunks with configurable word overlap)
   ▼
[3] Embedding ────────────── core/embedder.py + core/hf_download.py
   │                         (ONNX all-MiniLM-L6-v2 via onnxruntime/tokenizers)
   ▼
[4] In-memory vector store ─ core/rag_engine.py
   │                         (ChromaDB wrapped by llama_index.vector_stores.chroma)
   ▼
[5] Retrieval ────────────── top-k vector similarity over query = Area of Life + Goal,
   │                         optional ms-marco cross-encoder re-rank and mandatory-keyword filter
   ▼
[6] Structured generation ── core/generator.py
   │                         (Pydantic schemas from core/prompts.py; LLM providers via core/llm_config.py)
   ▼
Results UI: subtopics & ideas tabs with clickable timestamps · history store (core/history.py) · exports (core/exports.py)
```

Each stage is a small, independently testable module in `core/`; `app.py` orchestrates them into the staged pipeline shown in the UI.

##  :toolbox: Technical Stack

-   **Python:** 3.14
-   **Frontend:** Streamlit
-   **Transcript Extraction:** `youtube-transcript-api` (Python Library)
-   **Database/Vector Store:** ChromaDB (in-memory, `llama_index.vector_stores.chroma`))
-   **Embedding Model:** `all-MiniLM-L6-v2` via `llama_index.core.embeddings`, `onnxruntime`, `tokenizers` (lightweight, no `sentence-transformers` with `torch`)
-   **LLM Engine:** support of multiple providers (local, cloud) - for RAG & Summarization
-   **LLM Providers/Models Supported:**
    - Model must support structured outputs (JSON Schema)
    - Local: Ollama (Llama3.2:1b, IBM Granite 4)
    - Cloud: OpenRouter (`google/gemma-4-26b-a4b-it:free` as default)
    - Cloud: OpenAI (`gpt-5-mini` as default)
    - Cloud: Anthropic (`claude-sonnet-5` as default)
-   **Central Configuration via `settings.py`:**
    - App-wide settings and defaults.
    - Production LLM provider/model defaults, can be changed in UI.
-   **Framework:** LlamaIndex (Simple Chain, no Agents for MVP)
-   **Persistence:** Local History (local JSON store)


## Project Structure

```
llm-yt-insight-extractor/
│
├── app.py                    # Streamlit UI, history management, export, reports
├── requirements.txt          # Pinned dependencies
├── .env.example              # Environment variables (Ollama URL, API_KEYs etc)
│
├── .streamlit/
│   └── config.toml
│
├── core/
│   ├── settings.py           # App-wide settings and defaults
│   ├── transcript.py         # YouTube fetching, 60-min validation
│   ├── chunker.py            # Timestamp chunking logic with overlaps
│   ├── embedder.py           # Embedding for ChromaDB vector search
│   ├── hf_download.py        # Embedding model downloader
│   ├── rag_engine.py         # LlamaIndex index creation, retrieval, synthesizer
│   ├── generator.py          # LLM generation pipeline for structured output
│   ├── llm_config.py         # LLM configuration supporting multiple providers
│   ├── prompts.py            # Prompt templates (Subtopics, Summaries, Actionable Ideas)
│   ├── history.py            # Persistence of generated content history
│   └── exports.py            # Markdown and JSON generators
│
├── tests/                    #  
│   └── test-*.py             # Unittest tests for modules
│
├── data/
│   ├── .transcript_cache/    # Local transcript cache storage for testing
│   └── history.json          # Local session history
│
├── models/                   # Local embedding model files
│
└── README.md
```

## Ollama models for Local LLM Setup

I tested several lightweight LLMs from Ollama and found that the following models work well for this MVP:

- `ibm/granite` 3.3/4/4.1 often fail to produce structured outputs, but can be used for RAG and summarization in general.
- `llama3.2:1b` is the best choice for structured outputs and is used as default in this MVP. It is fast enough, lightweight, and produces structured outputs.

## 🚀 Instructions to reproduce

### :hammer_and_wrench: Setup

Install exact Python dependencies from `requirements.txt` and do **not** install `sentence-transformers` for this MVP. Embeddings run locally as ONNX models via `onnxruntime` and `tokenizers` — no Torch required. The model files (`Xenova/all-MiniLM-L6-v2` embeddings, `Xenova/ms-marco-MiniLM-L-6-v2` reranker) are downloaded from the Hugging Face Hub by `core/hf_download.py`.

```bash
pip install -r requirements.txt
python3 onnx_download.py   # fetch ONNX embedding + reranker models into models/
```

### 🖥️ One-command local run

`run_local.sh` automates a full local setup: loads `.env`, installs/starts Ollama, pulls `$OLLAMA_MODEL`, creates a virtualenv, installs dependencies, downloads the ONNX models (`onnx_download.py`), and starts Streamlit on port 8501.

```bash
cp .env.example .env   # adjust values if needed
bash run_local.sh
```

Then open http://localhost:8501.

### 🐳 Docker & Docker Compose

This repository includes a `Dockerfile` and `docker-compose.yml` for local deployment with Ollama.

Copy `.env.example` to `.env` and adjust values if needed.

```bash
cp .env.example .env
```

Start both services with:

```bash
docker compose up --build
```

Then open:

```bash
http://localhost:8501
```

The app connects to Ollama through the compose service name `ollama` at `http://ollama:11434`.

## ⚙️ Configuration

All settings come from environment variables (see `.env.example`). Defaults also live in `core/settings.py`, and the provider/model can be switched at runtime on the app's ⚙️ Configuration page.

| Variable | Purpose | Default |
|---|---|---|
| `USE_OLLAMA` | Consumed by `run_local.sh`: install/start Ollama and pull the model | `true` |
| `OLLAMA_HOST` | Ollama server URL (`http://ollama:11434` under docker compose) | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default local model | `llama3.2:1b` |
| `OLLAMA_TIMEOUT` | Ollama request timeout (seconds) | `300` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | BYOK OpenAI access | `gpt-5-mini` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | BYOK Anthropic access | `claude-sonnet-5` |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | BYOK OpenRouter access | `google/gemma-4-26b-a4b-it:free` |

Note: `.env` is loaded automatically by `run_local.sh` and by docker compose; for manual runs, export the variables yourself.

## 🧪 Testing

Use command below to run all tests:

```bash
python -m unittest discover -s tests/
```

or run a specific test file:

```bash
python -m unittest tests/test_llm_config.py
```

## Next steps

I plan to:
- add LLM-as-a-Judge functionality to validate the quality of generated summaries and actionable ideas.
- fine-tune the prompt templates for better relevances and hallucination reduction.
- test other lightweight Ollama models with structured outputs.

Stay tuned!

## Support

🙏 Thank you for your attention and time!

- If you experience any issue while following this instruction (or something left unclear), please add it to [Issues](/issues), I'll be glad to help/fix. And your feedback, questions & suggestions are welcome as well!
- Feel free to fork and submit pull requests.

If you find this project helpful, please ⭐️star⭐️ my repo 
https://github.com/dmytrovoytko/llm-yt-insight-extractor to help other people discover it 🙏

Made with ❤️ in Ukraine 🇺🇦 Dmytro Voytko
