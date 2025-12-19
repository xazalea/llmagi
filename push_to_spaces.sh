#!/bin/bash
# Script to push to Hugging Face Spaces

echo "🚀 Pushing to Hugging Face Spaces..."
echo ""

# Check if hf CLI is available
if command -v hf &> /dev/null; then
    echo "✓ Using hf CLI"
    
    # Login if needed
    echo "Checking authentication..."
    hf whoami &> /dev/null || {
        echo "Please login to Hugging Face:"
        hf login
    }
    
    # Push using hf CLI
    echo ""
    echo "Pushing to space..."
    hf upload xtoazt/newllm . --repo-type=space --exclude=".git,__pycache__,*.pyc,*.pt,*.pth,checkpoints,data_cache,venv,env,.venv"
    
else
    echo "Using git with authentication token"
    echo ""
    echo "To push, you need to:"
    echo "1. Get an access token from: https://huggingface.co/settings/tokens"
    echo "2. Use it as password when prompted"
    echo ""
    echo "Or run:"
    echo "  git push https://<YOUR_TOKEN>@huggingface.co/spaces/xtoazt/newllm main"
    echo ""
    echo "Setting up git remote..."
    git remote remove huggingface 2>/dev/null
    git remote add huggingface https://huggingface.co/spaces/xtoazt/newllm
    
    echo ""
    echo "Now run:"
    echo "  git push huggingface main"
    echo ""
    echo "When prompted for password, use your Hugging Face access token"
fi

