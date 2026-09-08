# no-ollama variant
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data/.transcript_cache \
    && mkdir -p /app/models

# Expose Streamlit port
EXPOSE 8501

# outside this docker Ollama
ENV OLLAMA_HOST=http://localhost:11434

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

RUN python onnx_download.py

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
