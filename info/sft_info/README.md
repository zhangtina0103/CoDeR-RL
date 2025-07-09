---
library_name: peft
license: other
base_model: /home/zhangty25/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: outputs
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# outputs

This model is a fine-tuned version of [/home/zhangty25/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct](https://huggingface.co//home/zhangty25/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct) on the tool_sft dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2103

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1.5e-05
- train_batch_size: 4
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.1787        | 3.2061 | 500  | 0.2132          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.0
- Pytorch 2.4.0+cu121
- Datasets 3.6.0
- Tokenizers 0.21.1