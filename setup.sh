#!/bin/bash

# AI-OPS Setup Script
# This script sets up the AI-OPS environment with all required dependencies

set -e  # Exit on error

echo "======================================"
echo "AI-OPS Setup Script"
echo "======================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found"

# Create environment
echo ""
echo "Creating conda environment 'aiops' with Python 3.12..."
conda env create -f environment.yml

# Activate environment (note: this only works in the script context)
echo ""
echo "Environment created. To activate it, run:"
echo "  conda activate aiops"
echo ""

# Give instructions for spaCy model
echo "After activating the environment, download the required spaCy model:"
echo "  python -m spacy download en_core_web_md"
echo ""

# Give instructions for Ollama
echo "Next steps:"
echo "1. Install and configure Ollama: https://github.com/ollama/ollama"
echo "2. Run: ollama run mistral"
echo "3. Start the API server: python -m uvicorn src.api:app --host 0.0.0.0 --port 8000"
echo "4. Start the CLI: python ai_ops_cli.py"
echo ""

echo "======================================"
echo "Setup Complete!"
echo "======================================"
