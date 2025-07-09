#!/usr/bin/env bash
mkdir -p logs  # create logs dir to avoid tee errors
export WANDB_MODE=offline

# lora_rank=16 instead of 8
export PROJ_NAME="qwen2.5-3b_sft"
export OUTPUT_DIR="/data1/zhangty25/Tool-N1/outputs"
export MODEL_PATH="/home/zhangty25/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct"
export LOG_DIR="logs/$PROJ_NAME.txt"

export LR=1.0e-5
export EPOCH=6
export BATCH_SIZE=4
export G_ACC=8

source /data1/zhangty25/miniforge3/envs/tooln1/bin/activate


cd LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train examples/qwen_sft.yaml \
    learning_rate=$LR \
    num_train_epochs=$EPOCH \
    per_device_train_batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$G_ACC \
    output_dir=$OUTPUT_DIR \
    run_name=$PROJ_NAME \
    model_name_or_path=$MODEL_PATH 2>&1 | tee -a "${LOG_DIR}"
