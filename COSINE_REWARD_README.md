# Cosine Reward Integration for Original Tool-N1

This document explains how to use the cosine reward logic from `demystify-long-cot` integrated into the **original** Tool-N1 reward model.

## Overview

The cosine reward logic implements length-based scaling that:
- **Rewards shorter correct solutions** more than longer ones
- **Rewards longer wrong solutions** more than shorter ones
- **Makes shorter solutions more risk-averse** (wrong penalized more than correct rewarded)

## Files Created

1. **`verl/verl/workers/reward_model/megatron/cosine_reward_model.py`** - The cosine reward model (extends original MegatronRewardModel)
2. **`original_cosine_reward_example.py`** - Example usage script
3. **`COSINE_REWARD_README.md`** - This documentation

## How to Use

### 1. Import the CosineMegatronRewardModel

```python
from verl.workers.reward_model.megatron.cosine_reward_model import CosineMegatronRewardModel
```

### 2. Replace MegatronRewardModel with CosineMegatronRewardModel

```python
# Instead of:
# reward_model = MegatronRewardModel(config, model_config, reward_model_module, megatron_config)

# Use:
reward_model = CosineMegatronRewardModel(
    config=config,
    model_config=model_config,
    reward_model_module=reward_model_module,
    megatron_config=megatron_config,
    sft_tokenizer=sft_tokenizer,
    rm_tokenizer=rm_tokenizer,
    # Cosine reward parameters
    min_value_wrong=-1.0,
    max_value_wrong=-0.1,
    min_value_correct=0.1,
    max_value_correct=1.0,
    max_len=2048,
    exceed_length_penalty=-1.0,
    repetition_max_penalty=0.0,
    repetition_ngram_size=3,
    correct_threshold=0.5
)
```

### 3. The cosine reward logic is automatically applied in `compute_reward()`

The `CosineMegatronRewardModel` extends the original `MegatronRewardModel` and automatically applies cosine scaling to rewards based on generation length.

## Cosine Reward Logic

The cosine reward uses the formula:
```
r = min_value + 0.5 * (max_value - min_value) * (1.0 + cos(progress * π))
```

Where:
- `progress = generation_length / max_length`
- For correct answers: uses `min_value_correct` and `max_value_correct`
- For wrong answers: uses `max_value_wrong` and `min_value_wrong` (swapped for negative numbers)

### Parameter Explanation

- **`min_value_wrong`/`max_value_wrong`**: Range for wrong answers (should be negative)
- **`min_value_correct`/`max_value_correct`**: Range for correct answers (should be positive)
- **`max_len`**: Maximum allowed generation length
- **`exceed_length_penalty`**: Penalty for exceeding max length
- **`repetition_max_penalty`**: Maximum penalty for repetition (0 to disable)
- **`repetition_ngram_size`**: N-gram size for repetition detection
- **`correct_threshold`**: Threshold to determine correct vs wrong answers

## Integration with Original Tool-N1

The `CosineMegatronRewardModel` is a **drop-in replacement** for the original `MegatronRewardModel`. It maintains the same interface while adding cosine reward functionality.

### Key Changes

1. **Additional constructor parameters** for cosine reward configuration
2. **Modified `compute_reward()` method** that applies cosine scaling
3. **Helper methods** for repetition penalty and cosine scaling
4. **Automatic length calculation** and cosine scaling

### Example Integration

```python
# In your training script or config file:
from verl.workers.reward_model.megatron.cosine_reward_model import CosineMegatronRewardModel

# When creating the reward model:
reward_model = CosineMegatronRewardModel(
    config=config,
    model_config=model_config,
    reward_model_module=reward_model_module,
    megatron_config=megatron_config,
    sft_tokenizer=sft_tokenizer,
    rm_tokenizer=rm_tokenizer,
    # Add your cosine parameters here
    min_value_wrong=-1.0,
    max_value_wrong=-0.1,
    min_value_correct=0.1,
    max_value_correct=1.0,
    max_len=2048,
    exceed_length_penalty=-1.0,
    repetition_max_penalty=0.0,
    repetition_ngram_size=3,
    correct_threshold=0.5
)
```

## Configuration Examples

### Conservative (Strong Length Bias)
```python
min_value_wrong=-2.0, max_value_wrong=-0.5
min_value_correct=0.5, max_value_correct=2.0
```

### Moderate (Balanced)
```python
min_value_wrong=-1.0, max_value_wrong=-0.1
min_value_correct=0.1, max_value_correct=1.0
```

### Lenient (Weak Length Bias)
```python
min_value_wrong=-0.5, max_value_wrong=-0.1
min_value_correct=0.1, max_value_correct=0.5
```

## Key Features

1. **Non-intrusive**: Doesn't modify the original reward model files
2. **Drop-in replacement**: Direct replacement for MegatronRewardModel
3. **Configurable**: All cosine parameters can be tuned
4. **Backward compatible**: Maintains all original functionality
5. **Automatic scaling**: Cosine scaling applied automatically in compute_reward()

## Testing

Run the example script to see how it works:
```bash
cd /data1/zhangty25/Tool-N1-Mod
python original_cosine_reward_example.py
```

## Integration Steps

1. **Backup your original config files**
2. **Update your reward model import**
3. **Replace MegatronRewardModel with CosineMegatronRewardModel**
4. **Add cosine parameters to your config**
5. **Test with a small batch first**
6. **Monitor logs for cosine scaling**

## Notes

- The cosine reward logic is based on the implementation from `demystify-long-cot`
- The repetition penalty is a simplified version; you can enhance it if needed
- The correct/wrong threshold is configurable via `correct_threshold` parameter
- The cosine scaling is applied automatically in the `compute_reward()` method
- All original functionality is preserved
