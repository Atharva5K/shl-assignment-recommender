#!/usr/bin/env bash
# Build script for Render deployment
# Installs CPU-only PyTorch (much smaller than GPU version),
# then remaining dependencies, then pre-builds the FAISS index.

set -e

echo ">>> Installing CPU-only PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ">>> Installing remaining dependencies..."
pip install -r requirements.txt

echo ">>> Building FAISS index..."
python -m scripts.build_index

echo ">>> Build complete!"
