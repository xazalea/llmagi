# Push to Hugging Face Spaces - Instructions

## Quick Push (Using hf CLI)

If you have `hf` CLI installed and logged in:

```bash
# Login (if not already)
hf login

# Push files
hf upload xtoazt/newllm . --repo-type=space --exclude=".git,__pycache__,*.pyc,*.pt,*.pth,checkpoints,data_cache"
```

## Push Using Git (Recommended)

### Step 1: Get Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Create a new token with **write** permissions
3. Copy the token

### Step 2: Push

```bash
# Option A: Use token in URL
git push https://<YOUR_TOKEN>@huggingface.co/spaces/xtoazt/newllm main

# Option B: Use git remote (will prompt for password)
git remote add huggingface https://huggingface.co/spaces/xtoazt/newllm
git push huggingface main
# When prompted for password, paste your access token
```

### Step 3: Verify

Your Space should be available at:
https://huggingface.co/spaces/xtoazt/newllm

## Files to Push

Essential files:
- `app.py` - Gradio application
- `README.md` - Space metadata
- `requirements.txt` - Dependencies
- `lib/` - Core library
- `implementations/` - Implementations

Files to exclude:
- `.git/`
- `__pycache__/`
- `*.pyc`
- `*.pt`, `*.pth` (model checkpoints)
- `checkpoints/`
- `data_cache/`
- `venv/`, `env/`

## Using the Script

```bash
# Run the push script
./push_to_spaces.sh
```

## Troubleshooting

### Authentication Error

If you get "Device not configured":
1. Get access token from https://huggingface.co/settings/tokens
2. Use it as password when pushing

### Large Files

If files are too large:
```bash
# Use git LFS for large files
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
```

### Push Specific Files Only

```bash
# Add only essential files
git add app.py README.md requirements.txt lib/ implementations/
git commit -m "Deploy to Spaces"
git push huggingface main
```

## Quick Command

```bash
# One-liner (replace YOUR_TOKEN with your actual token)
git push https://YOUR_TOKEN@huggingface.co/spaces/xtoazt/newllm main
```

