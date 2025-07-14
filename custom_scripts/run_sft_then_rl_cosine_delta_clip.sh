#!/usr/bin/env bash

# Run Supervised Fine-Tuning (SFT)
echo "Starting SFT..."
bash qwen_sft.sh

# Check if SFT succeeded
if [ $? -ne 0 ]; then
    echo "SFT failed. Exiting."
    exit 1
fi

echo "SFT completed successfully. Starting RL with cosine+delta+clip reward..."

# Set environment variables for cosine+delta+clip reward
export reward_manager=custom
export reward_mechanism=cosine+clip+delta

# Run Reinforcement Learning (RL) with cosine+delta+clip reward
bash qwen_rl.sh

if [ $? -ne 0 ]; then
    echo "RL training failed."
    exit 1
fi

echo "RL training with cosine+delta+clip reward completed successfully."
