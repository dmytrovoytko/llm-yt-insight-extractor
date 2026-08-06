# LLM Defaults
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"  # granite4.1:3b granite4:350m "granite3.3:2b
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"
# DEFAULT_HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# RAG 
TOP_K = 4 # 5

# Generator
GENERATOR_TEST_DEBUG = False # True only for dev to test UI faster - returns fixed outputs without calling LLM

# Embedding
DEFAULT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2" # lightweight model with ONNX

# Chunking
DEFAULT_CHUNK_WORDS = 300
DEFAULT_OVERLAP_WORDS = 50

# Transcripts
VALIDATE_DURATION = False # True
DURATION_TRESHOLD = 60 * 60  # 60 minutes
USE_TRANSCRIPT_CACHE = True
TRANSCRIPT_CACHE_EXTENSION = ".txt"

# Debug messages
DEBUG = False # True

