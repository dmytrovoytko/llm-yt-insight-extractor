# YT Insight Extractor

**Focus your learning. Extract exactly what you need from top podcasts and turn it into your personal growth plan.**

## 📌 Intro

YT Insight Extractor is an AI-powered local assistant built in Python and Streamlit. It transforms long-form YouTube interviews (like Tim Ferriss, Andrew Huberman, or Diary of a CEO) into structured, actionable knowledge. Instead of spending hours listening to a 3-hour podcast to find advice relevant to your specific goals, this app distills the video into targeted subtopics, summaries, and a concrete action plan.

## ⚠️ Problem Statement

There is an abundance of high-quality, long-form educational content on YouTube. However, extracting practical, personalized value from 2-to-3-hour interviews is highly inefficient. Listeners often have specific areas of life they want to improve, but are forced to consume the entire video, taking extensive notes to find the few actionable insights that apply to them.

## 💡 Solution

This assistant uses a Retrieval-Augmented Generation (RAG) pipeline driven by local LLMs (via Ollama). By providing a YouTube link and a specific personal development goal, the app extracts the transcript, chunks it with timestamp overlaps, and saves it into a vector database. A streamlined LangChain pipeline then queries this database against the user's prompt to generate a chronologically tabbed summary of subtopics and a formatted list of actionable ideas—with clickable video timestamps.

> YouTube Transcript API: Unfortunately, YouTube has started blocking most IPs that are known to belong to cloud providers (like AWS, Google Cloud Platform, Azure, etc.), which means you will most likely run into RequestBlocked or IpBlocked exceptions when deploying your code to any cloud solutions. Same can happen to the IP of your self-hosted solution, if you are doing too many requests. 

## 🏗️ Solution Architecture

...

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
    - Cloud: OpenRouter (`google/gemma-4-26b-a4b-it` as default)
    - Cloud: OpenAI (`gpt-5-mini` as default)
    - Cloud: Anthropic (`claude-sonnet-5` as default)
-   **Central Configuration via `settings.py`):**
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

Install exact Python dependencies from `requirements.txt` and do not install `sentence-transformers` for this MVP. The project uses `all-MiniLM-L6-v2` through `llama-index.embeddings.huggingface` and relies on the `transformers` library instead of `sentence-transformers`.

```bash
pip install -r requirements.txt
```

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
