#!/bin/bash

if [ "$USE_OLLAMA" = "true" ]; then
    # Ollama should be started by docker compose
    echo "Ollama should be ready on $OLLAMA_HOST"
fi

# Start the Python application
echo "Starting Python application..."
python onnx_download.py
exec streamlit run app.py
