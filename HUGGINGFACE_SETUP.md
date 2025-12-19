# Hugging Face Setup Guide

## Installing Hugging Face CLI

### Option 1: Using pip (Recommended)

```bash
pip install --upgrade huggingface_hub
```

### Option 2: Using conda

```bash
conda install -c conda-forge huggingface_hub
```

## Using Hugging Face CLI

### Login (Optional)

If you need to access private repositories or upload models:

```bash
huggingface-cli login
```

Or using Python:

```bash
python -m huggingface_hub.cli.login
```

### Download from Hugging Face Space

The correct command syntax:

```bash
# Using huggingface-cli
huggingface-cli download xtoazt/newllm --repo-type=space

# Or using Python module
python -m huggingface_hub download xtoazt/newllm --repo-type=space
```

### Download to Specific Directory

```bash
huggingface-cli download xtoazt/newllm --repo-type=space --local-dir ./downloaded_space
```

### Upload to Hugging Face Space

```bash
# Upload entire directory
huggingface-cli upload xtoazt/newllm ./local_directory --repo-type=space

# Upload specific files
huggingface-cli upload xtoazt/newllm ./file.py --repo-type=space
```

## Alternative: Using Python API

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download single file
file_path = hf_hub_download(
    repo_id="xtoazt/newllm",
    repo_type="space",
    filename="app.py"
)

# Download entire repository
snapshot_download(
    repo_id="xtoazt/newllm",
    repo_type="space",
    local_dir="./downloaded_space"
)
```

## Common Commands

### List files in repository

```bash
huggingface-cli scan-cache
```

### Clear cache

```bash
huggingface-cli scan-cache --dir ~/.cache/huggingface
```

### Check if logged in

```bash
huggingface-cli whoami
```

## Troubleshooting

### Command not found: hf

The command `hf` doesn't exist. Use `huggingface-cli` instead:

```bash
# Wrong
hf download ...

# Correct
huggingface-cli download ...
```

### Command not found: huggingface-cli

If `huggingface-cli` is not found, use the Python module:

```bash
python -m huggingface_hub download ...
```

Or add to PATH:

```bash
# Find installation location
python -c "import huggingface_hub; print(huggingface_hub.__file__)"

# Add to PATH (example)
export PATH="$PATH:$(python -m site --user-base)/bin"
```

## Quick Reference

```bash
# Install
pip install --upgrade huggingface_hub

# Login
huggingface-cli login

# Download
huggingface-cli download <repo_id> --repo-type=space

# Upload
huggingface-cli upload <repo_id> <local_path> --repo-type=space

# Using Python
python -m huggingface_hub download <repo_id> --repo-type=space
```

