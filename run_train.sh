#!/bin/bash
set -e

echo "===== WandB login ====="
if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin $WANDB_API_KEY
else
  echo "WANDB_API_KEY not set → running without WandB"
fi
echo "===== HuggingFace login ====="

if [ -n "$HF_TOKEN" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "HF_TOKEN not set → skipping HuggingFace login"
fi

echo "===== Starting training ====="
python train.py
echo "===== Training finished ====="
