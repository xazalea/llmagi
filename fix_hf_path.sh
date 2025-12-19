#!/bin/bash
# Fix PATH to include hf command

echo "Adding Hugging Face CLI to PATH..."

# Add to PATH for current session
export PATH="$PATH:/Users/rohan/Library/Python/3.9/bin"

# Add to ~/.zshrc for permanent fix
if ! grep -q "/Users/rohan/Library/Python/3.9/bin" ~/.zshrc 2>/dev/null; then
    echo "" >> ~/.zshrc
    echo "# Hugging Face CLI" >> ~/.zshrc
    echo 'export PATH="$PATH:/Users/rohan/Library/Python/3.9/bin"' >> ~/.zshrc
    echo "✓ Added to ~/.zshrc"
else
    echo "✓ Already in ~/.zshrc"
fi

echo ""
echo "Now you can use:"
echo "  hf download xtoazt/newllm --repo-type=space"
echo ""
echo "Or reload your shell:"
echo "  source ~/.zshrc"

