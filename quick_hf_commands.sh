#!/bin/bash
# Quick Hugging Face commands

# The correct command syntax:
# Use python3 -m huggingface_hub instead of "hf"

echo "Hugging Face CLI Commands:"
echo ""
echo "Download from Space:"
echo "python3 -m huggingface_hub download xtoazt/newllm --repo-type=space"
echo ""
echo "Download to specific directory:"
echo "python3 -m huggingface_hub download xtoazt/newllm --repo-type=space --local-dir ./downloaded"
echo ""
echo "Login (if needed):"
echo "python3 -m huggingface_hub.cli.login"
echo ""
echo "Upload to Space:"
echo "python3 -m huggingface_hub upload xtoazt/newllm ./local_dir --repo-type=space"

