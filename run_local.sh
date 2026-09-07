#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Check if ".env" exists, source it
if [[ -e ".env" ]]; then
    echo "📄 Loading environment variables from .env file..."
    set -a # automatically export all variables
    source .env
    set +a # turn off automatic export
else
    echo
    echo "❌ No .env file with paramaters found. Exiting."
    exit 1
fi

# Ensure necessary directories exist
mkdir -p data models

echo "====================================================="
echo "🚀 Running locally..."
echo "====================================================="

# 2. Check if USE_OLLAMA is true
if [ "$USE_OLLAMA" = "true" ]; then
    echo "🦙 USE_OLLAMA is true. Setting up Ollama..."
    
    # Check if Ollama is installed
    if ! command -v ollama &> /dev/null; then
        echo "⬇️ Ollama not found. Installing..."
        # Official Ollama installation script
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "✅ Ollama is already installed."
    fi

    # Check if Ollama server is already running
    if ! pgrep -x "ollama" > /dev/null; then
        echo "🏃 Starting Ollama server in the background..."
        # Start ollama serve in the background, redirecting output to a log file
        nohup ollama serve > ollama.log 2>&1 &
        
        # Give the server a few seconds to spin up
        sleep 3 
        echo "✅ Ollama server started (Logs: ollama.log)"
    else
        echo "✅ Ollama server is already running."
    fi
    
    # Pull a default model
    echo "⬇️ Pulling Ollama model $OLLAMA_MODEL..."
    ollama pull $OLLAMA_MODEL
fi

# 3. Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 could not be found. Please install Python 3."
    exit 1
fi

# 4. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# 5. Activate the virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# 6. Install requirements
echo "📥 Installing dependencies..."
# pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# 7. Download ONNX models
echo "📥 Downloading ONNX models..."
python3 onnx_download.py

# 8. Run the Streamlit app locally
echo "🏃 Starting Streamlit locally..."
streamlit run app.py --server.port=8501

