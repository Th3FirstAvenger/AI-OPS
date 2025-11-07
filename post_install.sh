#!/bin/bash

# AI-OPS Post-Installation Script
# Run this after activating the conda environment

set -e

echo "======================================"
echo "AI-OPS Post-Installation"
echo "======================================"
echo ""

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "aiops" ]]; then
    echo "⚠️  Warning: You're not in the 'aiops' conda environment"
    echo "Please run: conda activate aiops"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install additional dependencies for Python 3.12+ compatibility
echo "Installing Python 3.12+ compatible dependencies..."
pip install newspaper4k lxml_html_clean

echo ""
echo "✓ Dependencies installed successfully"
echo ""

# Download spaCy model
echo "Downloading spaCy English model (en_core_web_md)..."
python -m spacy download en_core_web_md

echo ""
echo "✓ spaCy model downloaded successfully"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import spacy; nlp = spacy.load('en_core_web_md'); print('✓ spaCy model loaded successfully')"
python -c "import newspaper; print('✓ newspaper4k loaded successfully')"

echo ""
echo "======================================"
echo "Post-Installation Complete!"
echo "======================================"
echo ""
echo "You can now start AI-OPS:"
echo "  # Start API server:"
echo "  python -m uvicorn src.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "  # Or start CLI directly:"
echo "  python ai_ops_cli.py"
echo ""
