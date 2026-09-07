#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

if [[ -e ".env" ]] then
    # loading script parameters from .env
    set -a            
    source .env
else
    echo
    echo "No .env file with paramaters found. Exiting."
    exit 1
fi

# Ensure necessary directories exist
mkdir -p data models

echo "====================================================="
echo "🚀 Running in Docker Compose..."
echo "====================================================="

# Run Docker Compose
echo "🏗️ Building and starting containers..."
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml up --build
