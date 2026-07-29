# YT Insight Extractor

**Focus your learning. Extract exactly what you need from top podcasts and turn it into your personal growth plan.**

## 📌 Intro

YT Insight Extractor is an AI-powered local assistant built in Python and Streamlit. It transforms long-form YouTube interviews (like Tim Ferriss, Andrew Huberman, or Diary of a CEO) into structured, actionable knowledge. Instead of spending hours listening to a 3-hour podcast to find advice relevant to your specific goals, this app distills the video into targeted subtopics, summaries, and a concrete action plan.

## ⚠️ Problem Statement

There is an abundance of high-quality, long-form educational content on YouTube. However, extracting practical, personalized value from 2-to-3-hour interviews is highly inefficient. Listeners often have specific areas of life they want to improve, but are forced to consume the entire video, taking extensive notes to find the few actionable insights that apply to them.

## 💡 Solution

This assistant uses a Retrieval-Augmented Generation (RAG) pipeline driven by local LLMs (via Ollama). By providing a YouTube link and a specific personal development goal, the app extracts the transcript, chunks it with timestamp overlaps, and saves it into a vector database. A streamlined LangChain pipeline then queries this database against the user's prompt to generate a chronologically tabbed summary of subtopics and a strictly formatted list of the Top 5 actionable ideas—with clickable video timestamps.

## 🏗️ Solution Architecture

## 🚀 Setup

Install exact Python dependencies from `requirements.txt` and do not install `sentence-transformers` for this MVP. The project uses `all-MiniLM-L6-v2` through `llama-index.embeddings.huggingface` and relies on the `transformers` library instead of `sentence-transformers`.

```bash
pip install -r requirements.txt
```
