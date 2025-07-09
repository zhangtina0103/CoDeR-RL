# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
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
export PRO_NAME="qwen_custom"
export EXPERIMENT_NAME="qwen_custom_reward"
export LOG_DIR="path/to/logs/qwen_custom_reward.txt"

export LR=1e-6
export ENTROPY=0
export MAX_RES=8192
export TEMPERATURE=0.7
export EPOCH=5
export KL_COE=0.001


bash verl/examples/agent/qwen_custom_reward.sh
