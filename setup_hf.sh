#!/bin/bash
# Setup script for Hugging Face CLI

echo "Setting up Hugging Face CLI..."

# Install huggingface_hub
pip install --upgrade huggingface_hub

# Login to Hugging Face (optional, for private repos)
echo "To login to Hugging Face, run:"
echo "huggingface-cli login"
echo ""
echo "Or:"
echo "python -m huggingface_hub.cli.login"

# Download from Hugging Face Space
echo ""
echo "To download from Hugging Face Space, use:"
echo "huggingface-cli download xtoazt/newllm --repo-type=space"
echo ""
echo "Or:"
echo "python -m huggingface_hub download xtoazt/newllm --repo-type=space"

