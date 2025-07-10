# Cosine Reward RL Training for Tool-N1

This document explains how to run RL training with cosine reward integration in Tool-N1.

## Overview

The cosine reward RL training integrates the cosine reward logic from `demystify-long-cot` into the Tool-N1 RL training pipeline. This provides length-based reward scaling that encourages shorter, correct solutions while discouraging longer, wrong solutions.

## Files Created

1. **`verl/verl/trainer/main_ppo_cosine.py`** - Modified PPO trainer with cosine reward support
2. **`custom_scripts/qwen_rl_cosine_reward.sh`** - Main RL training script with cosine reward
3. **`custom_scripts/cosine_reward_configs.sh`** - Configuration examples and explanations
4. **`COSINE_RL_TRAINING_README.md`** - This documentation

## Quick Start

### 1. Basic Training

```bash
cd /data1/zhangty25/Tool-N1-Mod
bash custom_scripts/qwen_rl_cosine_reward.sh
```

### 2. Custom Configuration

```bash
# Set your desired cosine parameters
export cosine_min_value_wrong="-1.0"
export cosine_max_value_wrong="-0.1"
export cosine_min_value_correct="0.1"
export cosine_max_value_correct="1.0"
export cosine_max_len="2048"
export cosine_exceed_length_penalty="-1.0"
export cosine_repetition_max_penalty="0.0"
export cosine_repetition_ngram_size="3"
export cosine_correct_threshold="0.5"

# Run training
bash custom_scripts/qwen_rl_cosine_reward.sh
```

## Configuration Examples

### Conservative (Strong Length Bias)
```bash
export cosine_min_value_wrong="-2.0"
export cosine_max_value_wrong="-0.5"
export cosine_min_value_correct="0.5"
export cosine_max_value_correct="2.0"
```

### Moderate (Balanced)
```bash
export cosine_min_value_wrong="-1.0"
export cosine_max_value_wrong="-0.1"
export cosine_min_value_correct="0.1"
export cosine_max_value_correct="1.0"
```

### Lenient (Weak Length Bias)
```bash
export cosine_min_value_wrong="-0.5"
export cosine_max_value_wrong="-0.1"
export cosine_min_value_correct="0.1"
export cosine_max_value_correct="0.5"
```

### With Repetition Penalty
```bash
export cosine_repetition_max_penalty="0.2"
export cosine_repetition_ngram_size="3"
```

## Training Script Details

### Main Script: `qwen_rl_cosine_reward.sh`

This script sets up the training environment with cosine reward parameters:

```bash
# Training parameters
export N_GPUS=4
export BASE_MODEL="/home/zhangty25/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct"
export DATA_DIR="/data1/zhangty25/Tool-N1/verl/verl/data"
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND="XFORMERS"

# Cosine reward parameters
export reward_manager="cosine"
export cosine_min_value_wrong="-1.0"
export cosine_max_value_wrong="-0.1"
export cosine_min_value_correct="0.1"
export cosine_max_value_correct="1.0"
export cosine_max_len="2048"
export cosine_exceed_length_penalty="-1.0"
export cosine_repetition_max_penalty="0.0"
export cosine_repetition_ngram_size="3"
export cosine_correct_threshold="0.5"
```

### Modified Trainer: `main_ppo_cosine.py`

The modified trainer automatically detects cosine reward configuration and applies the cosine scaling:

- **Environment-based configuration**: Reads cosine parameters from environment variables
- **Automatic integration**: Applies cosine scaling in the reward computation
- **Backward compatibility**: Falls back to original reward mechanisms if cosine is not configured

## Parameter Explanation

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `cosine_min_value_wrong` | Minimum reward for wrong answers | -1.0 | Negative values |
| `cosine_max_value_wrong` | Maximum reward for wrong answers | -0.1 | Negative values |
| `cosine_min_value_correct` | Minimum reward for correct answers | 0.1 | Positive values |
| `cosine_max_value_correct` | Maximum reward for correct answers | 1.0 | Positive values |
| `cosine_max_len` | Maximum allowed generation length | 2048 | Positive integers |
| `cosine_exceed_length_penalty` | Penalty for exceeding max length | -1.0 | Negative values |
| `cosine_repetition_max_penalty` | Maximum penalty for repetition | 0.0 | 0.0 to disable |
| `cosine_repetition_ngram_size` | N-gram size for repetition detection | 3 | Positive integers |
| `cosine_correct_threshold` | Threshold for correct vs wrong | 0.5 | 0.0 to 1.0 |

## Cosine Reward Formula

The cosine reward uses the formula:
```
r = min_value + 0.5 * (max_value - min_value) * (1.0 + cos(progress * π))
```

Where:
- `progress = generation_length / max_length`
- For correct answers: uses `min_value_correct` and `max_value_correct`
- For wrong answers: uses `max_value_wrong` and `min_value_wrong` (swapped for negative numbers)

## Training Process

1. **Environment Setup**: Cosine parameters are read from environment variables
2. **Reward Computation**: Original rewards are computed by the reward model
3. **Cosine Scaling**: Rewards are scaled based on generation length using cosine formula
4. **Length Calculation**: Generation lengths are calculated for each sample
5. **Threshold Application**: Correct/wrong classification based on `cosine_correct_threshold`
6. **Penalty Application**: Repetition and length penalties are applied if configured

## Monitoring and Logging

The training script includes comprehensive logging:

- **Cosine scaling logs**: Shows original vs cosine-scaled rewards
- **Length statistics**: Generation lengths for each batch
- **Parameter validation**: Ensures cosine parameters are valid
- **Performance metrics**: Training progress and reward statistics

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the cosine reward model is in the correct path
2. **Parameter Validation**: Check that wrong values are negative and correct values are positive
3. **Memory Issues**: Adjust batch size or max length if needed
4. **Configuration Errors**: Verify environment variables are set correctly

### Debug Mode

To enable debug logging, add:
```bash
export VERL_LOG_LEVEL="DEBUG"
```

### Validation

Run the configuration script to validate your settings:
```bash
bash custom_scripts/cosine_reward_configs.sh
```

## Advanced Usage

### Multi-GPU Training

```bash
export N_GPUS=8
export ROLLOUT_TP_SIZE=4
bash custom_scripts/qwen_rl_cosine_reward.sh
```

### Custom Model Path

```bash
export BASE_MODEL="/path/to/your/model"
bash custom_scripts/qwen_rl_cosine_reward.sh
```

### Custom Data Path

```bash
export DATA_DIR="/path/to/your/data"
bash custom_scripts/qwen_rl_cosine_reward.sh
```

## Integration with Existing Workflows

The cosine reward training is designed to be a drop-in replacement for existing RL training:

1. **Same interface**: Uses the same command-line interface
2. **Same data format**: Compatible with existing datasets
3. **Same model format**: Works with existing model checkpoints
4. **Same logging**: Integrates with existing logging systems

## Performance Considerations

- **Memory usage**: Cosine scaling adds minimal memory overhead
- **Computation time**: Length calculation adds negligible time
- **Scalability**: Works with multi-GPU and distributed training
- **Checkpointing**: Compatible with existing checkpoint formats

## Best Practices

1. **Start with moderate settings**: Use balanced configuration first
2. **Monitor reward distributions**: Check that rewards are reasonable
3. **Validate on small batches**: Test configuration before full training
4. **Adjust gradually**: Make small changes to parameters
5. **Use repetition penalty**: Enable for better text quality
6. **Monitor length statistics**: Ensure generation lengths are reasonable
