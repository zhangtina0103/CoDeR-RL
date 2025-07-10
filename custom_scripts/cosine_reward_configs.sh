#!/usr/bin/env bash

# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Cosine Reward Configuration Examples
# This script shows different cosine reward configurations for experimentation

echo "=== Cosine Reward Configuration Examples ==="
echo

# Conservative (Strong Length Bias)
echo "1. Conservative Configuration (Strong Length Bias):"
echo "   - Rewards shorter correct solutions much more than longer ones"
echo "   - Strongly penalizes shorter wrong solutions"
echo "   export cosine_min_value_wrong='-2.0'"
echo "   export cosine_max_value_wrong='-0.5'"
echo "   export cosine_min_value_correct='0.5'"
echo "   export cosine_max_value_correct='2.0'"
echo

# Moderate (Balanced)
echo "2. Moderate Configuration (Balanced):"
echo "   - Balanced length bias"
echo "   - Standard cosine scaling"
echo "   export cosine_min_value_wrong='-1.0'"
echo "   export cosine_max_value_wrong='-0.1'"
echo "   export cosine_min_value_correct='0.1'"
echo "   export cosine_max_value_correct='1.0'"
echo

# Lenient (Weak Length Bias)
echo "3. Lenient Configuration (Weak Length Bias):"
echo "   - Weak length bias"
echo "   - More forgiving of longer solutions"
echo "   export cosine_min_value_wrong='-0.5'"
echo "   export cosine_max_value_wrong='-0.1'"
echo "   export cosine_min_value_correct='0.1'"
echo "   export cosine_max_value_correct='0.5'"
echo

# Aggressive (Very Strong Length Bias)
echo "4. Aggressive Configuration (Very Strong Length Bias):"
echo "   - Very strong preference for shorter solutions"
echo "   - Heavy penalties for wrong answers"
echo "   export cosine_min_value_wrong='-3.0'"
echo "   export cosine_max_value_wrong='-0.2'"
echo "   export cosine_min_value_correct='1.0'"
echo "   export cosine_max_value_correct='3.0'"
echo

# With Repetition Penalty
echo "5. With Repetition Penalty:"
echo "   - Adds repetition penalty to discourage repetitive text"
echo "   export cosine_repetition_max_penalty='0.2'"
echo "   export cosine_repetition_ngram_size='3'"
echo

# Custom Length Threshold
echo "6. Custom Length Threshold:"
echo "   - Adjust the threshold for correct vs wrong answers"
echo "   export cosine_correct_threshold='0.7'  # Higher threshold"
echo "   export cosine_correct_threshold='0.3'  # Lower threshold"
echo

echo "=== Usage Examples ==="
echo

echo "To use Conservative configuration:"
echo "export cosine_min_value_wrong='-2.0'"
echo "export cosine_max_value_wrong='-0.5'"
echo "export cosine_min_value_correct='0.5'"
echo "export cosine_max_value_correct='2.0'"
echo "bash custom_scripts/qwen_rl_cosine_reward.sh"
echo

echo "To use Moderate configuration with repetition penalty:"
echo "export cosine_min_value_wrong='-1.0'"
echo "export cosine_max_value_wrong='-0.1'"
echo "export cosine_min_value_correct='0.1'"
echo "export cosine_max_value_correct='1.0'"
echo "export cosine_repetition_max_penalty='0.1'"
echo "export cosine_repetition_ngram_size='3'"
echo "bash custom_scripts/qwen_rl_cosine_reward.sh"
echo

echo "=== Parameter Explanation ==="
echo
echo "cosine_min_value_wrong: Minimum reward for wrong answers (should be negative)"
echo "cosine_max_value_wrong: Maximum reward for wrong answers (should be negative)"
echo "cosine_min_value_correct: Minimum reward for correct answers (should be positive)"
echo "cosine_max_value_correct: Maximum reward for correct answers (should be positive)"
echo "cosine_max_len: Maximum allowed generation length"
echo "cosine_exceed_length_penalty: Penalty for exceeding max length"
echo "cosine_repetition_max_penalty: Maximum penalty for repetition (0 to disable)"
echo "cosine_repetition_ngram_size: N-gram size for repetition detection"
echo "cosine_correct_threshold: Threshold to determine correct vs wrong answers"
echo

echo "=== Formula ==="
echo
echo "The cosine reward uses the formula:"
echo "r = min_value + 0.5 * (max_value - min_value) * (1.0 + cos(progress * π))"
echo "where progress = generation_length / max_length"
echo
echo "This creates a cosine curve that:"
echo "- Starts high for correct answers (encourages short, correct solutions)"
echo "- Starts low for wrong answers (discourages short, wrong solutions)"
echo "- Gradually changes as length increases"
echo "- Applies penalties for exceeding max length or repetition"
