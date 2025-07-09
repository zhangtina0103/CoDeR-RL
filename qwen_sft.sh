export PROJ_NAME="qwen_sft"
export OUTPUT_DIR="path/to/output/dir"
export MODEL_PATH="path/to/model/Qwen2.5-7B-Instruct"
export LOG_DIR="logs/$PROJ_NAME.txt"

export LR=2.0e-5
export EPOCH=20
export BATCH_SIZE=4
export G_ACC=8

cd LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train examples/qwen_sft.yaml \
    learning_rate=$LR \
    num_train_epochs=$EPOCH \
    per_device_train_batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$G_ACC \
    output_dir=$OUTPUT_DIR \
    run_name=$PROJ_NAME \
    model_name_or_path=$MODEL_PATH 2>&1 | tee -a "${LOG_DIR}"
