# Fix: Hugging Face Command Error

## Problem

You tried to run:
```bash
hf download xtoazt/newllm --repo-type=space
```

But got:
```
zsh: command not found: hf
```

## Solution

The command `hf` doesn't exist. Use one of these alternatives:

### Option 1: Using Python Module (Recommended)

```bash
python3 -m huggingface_hub download xtoazt/newllm --repo-type=space
```

### Option 2: Install huggingface-cli

```bash
# Install
pip install --upgrade huggingface_hub

# Then use
huggingface-cli download xtoazt/newllm --repo-type=space
```

### Option 3: Add to PATH

If `huggingface-cli` is installed but not in PATH:

```bash
# Find installation
python3 -m site --user-base

# Add to PATH (add to ~/.zshrc)
export PATH="$PATH:$(python3 -m site --user-base)/bin"

# Reload shell
source ~/.zshrc
```

## Quick Reference

### Download from Space

```bash
# Correct command
python3 -m huggingface_hub download xtoazt/newllm --repo-type=space

# With local directory
python3 -m huggingface_hub download xtoazt/newllm --repo-type=space --local-dir ./downloaded
```

### Upload to Space

```bash
python3 -m huggingface_hub upload xtoazt/newllm ./local_directory --repo-type=space
```

### Login

```bash
python3 -m huggingface_hub.cli.login
```

## Verify Installation

```bash
# Check if module is available
python3 -c "import huggingface_hub; print('OK')"

# Check version
python3 -m huggingface_hub --version
```

## Common Issues

### Issue: "No module named huggingface_hub"

**Solution:**
```bash
python3 -m pip install --user huggingface_hub
```

### Issue: "Command not found: huggingface-cli"

**Solution:** Use Python module instead:
```bash
python3 -m huggingface_hub download ...
```

### Issue: Permission denied

**Solution:** Use `--user` flag:
```bash
python3 -m pip install --user huggingface_hub
```

## Example: Download Your Space

```bash
# Make sure huggingface_hub is installed
python3 -m pip install --user huggingface_hub

# Download the space
python3 -m huggingface_hub download xtoazt/newllm --repo-type=space --local-dir ./newllm_space
```

