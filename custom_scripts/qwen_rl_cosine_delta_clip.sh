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

#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=3,4,5,6
export N_GPUS=4
export BASE_MODEL="/home/zhangty25/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct"
export DATA_DIR="/data1/zhangty25/Tool-N1/verl/verl/data"
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND="XFORMERS"

export GPU_UT=0.6
export BA_SIZE=512
export MAX_PROMPT_LEN=4096
export PRO_NAME="qwen_cosine"
export EXPERIMENT_NAME="qwen_cosine_reward"
export LOG_DIR="/data1/zhangty25/Tool-N1/logs/qwen_cosine_reward.txt"

export LR=1e-6
export ENTROPY=0
export MAX_RES=8192
export TEMPERATURE=0.7
export EPOCH=5
export KL_COE=0.001

# Reward manager (using cosine+delta+clip)
export reward_manager="cosine+delta+clip"
export cosine_min_value_wrong="-1.0"
export cosine_max_value_wrong="-0.1"
export cosine_min_value_correct="0.1"
export cosine_max_value_correct="1.0"
export cosine_max_len="2048"
export cosine_exceed_length_penalty="-1.0"
export cosine_repetition_max_penalty="0.0"
export cosine_repetition_ngram_size="3"
export cosine_correct_threshold="0.5"

# Run the training with cosine+delta+clip reward manager

# Run the training with cosine+delta+clip reward manager
export reward_manager="cosine+delta+clip"
export EXPERIMENT_NAME="qwen_cosine_delta_clip_reward"
export LOG_DIR="/data1/zhangty25/Tool-N1/logs/qwen_cosine_delta_clip_reward.txt"

python3 -m verl.trainer.main_ppo_cosine \
algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=$KL_COE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UT \
actor_rollout_ref.rollout.temperature=$TEMPERATURE \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.model.use_remove_padding=True \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=$BA_SIZE \
data.val_batch_size=1312 \
data.max_prompt_length=$MAX_PROMPT_LEN \
data.max_response_length=$MAX_RES \
algorithm.kl_ctrl.kl_coef=$KL_COE \
trainer.critic_warmup=0 \
trainer.logger=['swanlab'] \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=10 \
trainer.project_name=$PRO_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
+actor_rollout_ref.actor.save_optimizer=True \
trainer.total_epochs=$EPOCH \
+reward_manager=$reward_manager 2>&1 | tee -a "$LOG_DIR"
