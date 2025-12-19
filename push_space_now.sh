#!/bin/bash
# Push to Hugging Face Spaces NOW

echo "🚀 Pushing to Hugging Face Spaces..."
echo ""

# Ensure we're on main branch
git checkout main

# Add essential files
git add app.py README.md requirements.txt lib/ implementations/ 2>/dev/null

# Commit if there are changes
if ! git diff --staged --quiet; then
    git commit -m "Update Space files for deployment"
fi

# Push to huggingface remote
echo "Pushing to Hugging Face Spaces..."
echo "You may need to authenticate with your token"
echo ""

# Try using hf CLI first
if command -v hf &> /dev/null; then
    echo "Using hf CLI to upload..."
    hf upload xtoazt/newllm . --repo-type=space --exclude=".git,__pycache__,*.pyc,*.pt,*.pth,checkpoints,data_cache,venv,env,.venv,newllm/.git" || {
        echo ""
        echo "hf CLI upload failed, trying git push..."
        git push huggingface main
    }
else
    echo "Using git push..."
    echo "If prompted, use your Hugging Face access token as password"
    echo "Get token from: https://huggingface.co/settings/tokens"
    echo ""
    git push huggingface main
fi

echo ""
echo "✅ Push complete! Your Space should be building now."
echo "Check status at: https://huggingface.co/spaces/xtoazt/newllm"

