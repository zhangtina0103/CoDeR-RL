#!/usr/bin/env python3
"""
Example script showing how to use the CosineRewardManager with Tool-N1.

This script demonstrates how to integrate the cosine reward logic from demystify-long-cot
into your Tool-N1 reward system without modifying the original reward manager.
"""

import sys
import os
sys.path.append('/data1/zhangty25/Tool-N1-Mod')

from verl.workers.cosine_reward_manager import CosineRewardManager
from verl.utils.custom_reward import compute_score_example  # You'll need to implement this


def example_usage():
    """Example of how to use the CosineRewardManager."""

    # Example 1: Pure cosine reward
    print("=== Example 1: Pure Cosine Reward ===")
    cosine_manager = CosineRewardManager(
        tokenizer=None,  # You'll need to pass your actual tokenizer
        num_examine=1,
        mechanism='cosine',
        compute_score=compute_score_example,  # Your scoring function
        # Cosine parameters
        min_value_wrong=-1.0,
        max_value_wrong=-0.1,
        min_value_correct=0.1,
        max_value_correct=1.0,
        max_len=2048,
        exceed_length_penalty=-1.0,
        repetition_max_penalty=0.0,
        repetition_ngram_size=3
    )

    # Example 2: Cosine + Clip reward
    print("=== Example 2: Cosine + Clip Reward ===")
    cosine_clip_manager = CosineRewardManager(
        tokenizer=None,
        num_examine=1,
        mechanism='cosine+clip',
        eta=0.5,  # Clip threshold
        compute_score=compute_score_example,
        # Cosine parameters
        min_value_wrong=-1.0,
        max_value_wrong=-0.1,
        min_value_correct=0.1,
        max_value_correct=1.0,
        max_len=2048,
        exceed_length_penalty=-1.0,
        repetition_max_penalty=0.1,  # Enable repetition penalty
        repetition_ngram_size=3
    )

    # Example 3: Cosine + Delta reward
    print("=== Example 3: Cosine + Delta Reward ===")
    cosine_delta_manager = CosineRewardManager(
        tokenizer=None,
        num_examine=1,
        mechanism='cosine+delta',
        compute_score=compute_score_example,
        # Cosine parameters
        min_value_wrong=-1.0,
        max_value_wrong=-0.1,
        min_value_correct=0.1,
        max_value_correct=1.0,
        max_len=2048,
        exceed_length_penalty=-1.0,
        repetition_max_penalty=0.0,
        repetition_ngram_size=3
    )

    # Example 4: Cosine + Clip + Delta reward (all three mechanisms)
    print("=== Example 4: Cosine + Clip + Delta Reward ===")
    cosine_clip_delta_manager = CosineRewardManager(
        tokenizer=None,
        num_examine=1,
        mechanism='cosine+clip+delta',
        eta=0.5,
        compute_score=compute_score_example,
        # Cosine parameters
        min_value_wrong=-1.0,
        max_value_wrong=-0.1,
        min_value_correct=0.1,
        max_value_correct=1.0,
        max_len=2048,
        exceed_length_penalty=-1.0,
        repetition_max_penalty=0.1,
        repetition_ngram_size=3
    )

    print("All cosine reward managers created successfully!")
    print("\nAvailable mechanisms:")
    print("- 'cosine': Pure cosine scaling based on length")
    print("- 'cosine+clip': Cosine scaling + clipping")
    print("- 'cosine+delta': Cosine scaling + delta reward")
    print("- 'cosine+clip+delta': All three mechanisms combined")
    print("- 'clip': Original clip mechanism")
    print("- 'delta': Original delta mechanism")
    print("- 'clip+delta': Original clip+delta mechanism")


def explain_cosine_logic():
    """Explain the cosine reward logic from demystify-long-cot."""
    print("\n=== Cosine Reward Logic Explanation ===")
    print("The cosine reward logic from demystify-long-cot implements length-based scaling:")
    print()
    print("1. **Shorter correct solutions** are rewarded more than longer ones")
    print("2. **Longer wrong solutions** are rewarded more than shorter ones")
    print("3. **Shorter solutions** are more risk-averse (wrong penalized more than correct rewarded)")
    print()
    print("The formula used is:")
    print("  r = min_value + 0.5 * (max_value - min_value) * (1.0 + cos(progress * π))")
    print("  where progress = generation_length / max_length")
    print()
    print("This creates a cosine curve that:")
    print("- Starts high for correct answers (encourages short, correct solutions)")
    print("- Starts low for wrong answers (discourages short, wrong solutions)")
    print("- Gradually changes as length increases")
    print("- Applies penalties for exceeding max length or repetition")


if __name__ == "__main__":
    example_usage()
    explain_cosine_logic()
