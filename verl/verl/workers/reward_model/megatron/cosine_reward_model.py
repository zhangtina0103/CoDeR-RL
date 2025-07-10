# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Megatron Reward Model with Cosine Reward Scaling.
"""

import math
import logging
from tensordict import TensorDict
from verl import DataProto
import torch
import torch.distributed

from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
from verl import DataProto
from verl.utils.torch_functional import broadcast_dict_tensor, split_dict_tensor_into_batches
from verl.workers.reward_model.base import BasePPORewardModel
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func


class CosineMegatronRewardModel(BasePPORewardModel):

    def __init__(self,
                 config,
                 model_config,
                 reward_model_module: torch.nn.ModuleList,
                 megatron_config,
                 sft_tokenizer=None,
                 rm_tokenizer=None,
                 # Cosine reward parameters
                 min_value_wrong=-1.0,
                 max_value_wrong=-0.1,
                 min_value_correct=0.1,
                 max_value_correct=1.0,
                 max_len=2048,
                 exceed_length_penalty=-1.0,
                 repetition_max_penalty=0.0,
                 repetition_ngram_size=3,
                 correct_threshold=0.5):
        self.config = config
        self.reward_model_module = reward_model_module
        self.megatron_config = megatron_config
        self.model_config = model_config
        self.device = 'cuda'
        self.sft_tokenizer = sft_tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.use_different_tokenizer = rm_tokenizer is not None

        # Cosine reward parameters
        self.min_value_wrong = min_value_wrong
        self.max_value_wrong = max_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_correct = max_value_correct
        self.max_len = max_len
        self.exceed_length_penalty = exceed_length_penalty
        self.repetition_max_penalty = repetition_max_penalty
        self.repetition_ngram_size = repetition_ngram_size
        self.correct_threshold = correct_threshold

        # Validate wrong values are negative
        if min_value_wrong > 0 or max_value_wrong > 0:
            raise ValueError("Wrong values should not be positive")

        logger = logging.getLogger("CosineMegatronRewardModel")
        logger.info(
            f"Initialized cosine reward model with"
            f" min_value_wrong: {min_value_wrong}, max_value_wrong: {max_value_wrong},"
            f" min_value_correct: {min_value_correct}, max_value_correct: {max_value_correct},"
            f" max_len: {max_len}, exceed_length_penalty: {exceed_length_penalty},"
            f" repetition_max_penalty: {repetition_max_penalty}, repetition_ngram_size: {repetition_ngram_size}")

        if self.config.param_offload:
            self.offload_params_to_cpu()

    def _get_repetition_penalty(self, generation: str) -> float:
        """Calculate repetition penalty based on n-gram repetition."""
        if self.repetition_max_penalty <= 0:
            return 0.0

        # Simple n-gram repetition detection
        words = generation.split()
        if len(words) < self.repetition_ngram_size:
            return 0.0

        ngrams = []
        for i in range(len(words) - self.repetition_ngram_size + 1):
            ngrams.append(' '.join(words[i:i + self.repetition_ngram_size]))

        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - len(unique_ngrams) / len(ngrams)

        return -self.repetition_max_penalty * repetition_ratio

    def _apply_cosine_scaling(self, score: float, gen_length: int, generation: str = "") -> float:
        """Apply cosine scaling based on generation length.

        The general idea is:
        - Shorter correct solutions should be rewarded over longer ones.
        - Longer wrong solutions should be rewarded over shorter ones.
        - Shorter solutions should be more risk averse (wrong penalized more than correct rewarded).
        """
        if gen_length >= self.max_len:
            return self.exceed_length_penalty

        # Determine if this is a correct or wrong answer
        if score >= self.correct_threshold:
            min_value = self.min_value_correct
            max_value = self.max_value_correct
            rep_penalty = 0
        else:
            # For wrong answers, swap min/max for cosine formula to work with negative numbers
            min_value = self.max_value_wrong
            max_value = self.min_value_wrong
            rep_penalty = self._get_repetition_penalty(generation)

        progress = gen_length / self.max_len
        cosine = math.cos(progress * math.pi)
        r = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        r += rep_penalty

        return r

    def re_encode_by_rm_tokenizer(self, data: DataProto) -> DataProto:
        assert self.use_different_tokenizer, 're-encode need rm tokenizer not be None!'
        # need to use rm tokenizer to re-generate input_ids, attention_mask and position_ids
        # 1. remove pad for each sequence
        # 2. decode by sft_tokenizer, remove sft system prompts
        # 3. encode by rm_tokenizer with rm system prompts, get rm_input_ids
        # 4. generate attention_mask and position_ids
        input_ids = data.batch['input_ids']  # (bs, seq_len)
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        ori_values = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
        ori_bs, ori_seqlen = input_ids.size(0), input_ids.size(1)
        input_ids_for_rm = []
        attention_mask_for_rm = []
        position_ids_for_rm = []
        print_decode = True
        ori_seqlen = ori_seqlen + 128
        for id, mask in zip(input_ids, attention_mask):
            # 1. remove pad for each sequence
            non_zero_indices = torch.nonzero(mask).view(-1)
            begin_pos, end_pos = non_zero_indices[0].item(), non_zero_indices[-1].item()
            valid_id = id[begin_pos:end_pos + 1]
            # 2. decode by sft_tokenizer, remove sft system prompts
            decode_result = self.sft_tokenizer.decode(valid_id)
            # workaround
            decode_with_rm_chat = decode_result.replace("<|user|>\n", "[INST] ").replace(
                "</s>\n<|assistant|>\n", " [/INST]").replace("</s> \n<|assistant|>\n", " [/INST]") + "</s>"
            if print_decode and torch.distributed.get_rank() == 0:
                # only print first decode result
                print(f'device {torch.cuda.current_device()}: sft decode result:\n{decode_result}\n \
                        \ndevice {torch.cuda.current_device()}: sft decode result with rm chat template:\n{decode_with_rm_chat}\n\n'
                     )
                print_decode = False
            # 3. encode by rm_tokenizer
            rm_input_ids = self.rm_tokenizer(decode_with_rm_chat,
                                             return_tensors='pt')['input_ids'][0].to(input_ids.device)
            # 4. generate attention_mask and position_ids
            rm_attention_mask = torch.ones_like(rm_input_ids, device=input_ids.device)
            cur_seqlen = rm_input_ids.shape[-1]
            # NOTE(gh): the later reward compute will process the shape (bs, seqlen_pad_128)
            if cur_seqlen > ori_seqlen:
                print(f'warninig: rm encode seqlen {cur_seqlen} > sft encode seqlen {ori_seqlen}')
                rm_input_ids = rm_input_ids[:ori_seqlen]
                rm_attention_mask = rm_attention_mask[:ori_seqlen]
            else:
                # right padding
                rm_input_ids = pad_sequence_to_length(rm_input_ids, ori_seqlen, self.rm_tokenizer.pad_token_id)
                rm_attention_mask = pad_sequence_to_length(rm_attention_mask, ori_seqlen, 0)
            rm_position_ids = torch.arange(0, ori_seqlen, device=input_ids.device)
            input_ids_for_rm.append(torch.unsqueeze(rm_input_ids, dim=0))
            attention_mask_for_rm.append(torch.unsqueeze(rm_attention_mask, dim=0))
            position_ids_for_rm.append(torch.unsqueeze(rm_position_ids, dim=0))
        input_ids_for_rm = torch.cat(input_ids_for_rm, dim=0)
        attention_mask_for_rm = torch.cat(attention_mask_for_rm, dim=0)
        position_ids_for_rm = torch.cat(position_ids_for_rm, dim=0)

        # (bs, seqlen) will not change, but input_ids, attention_mask and position_ids will change
        # NOTE(gh): need to replace into origin values after compute reward!
        data.batch['input_ids'] = input_ids_for_rm
        data.batch['attention_mask'] = attention_mask_for_rm
        data.batch['position_ids'] = position_ids_for_rm

        return data, ori_values

    @torch.no_grad()
    def compute_reward(self, data: DataProto) -> DataProto:
        if self.config.param_offload:
            self.load_params_to_cuda()

        if self.use_different_tokenizer:
            data, ori_values = self.re_encode_by_rm_tokenizer(data)

        input_ids = data.batch['input_ids']  # (bs, seq_len')
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']

        responses = data.batch['responses']
        batch_size = responses.size(0)
        response_length = responses.size(1)

        with torch.no_grad():
            output = self.forward_batch(data)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                logits = torch.cat([o['logits'] for o in output], dim=0)
            else:
                logits = torch.empty(
                    (input_ids.shape[0], input_ids.shape[1]),
                    dtype=torch.bfloat16,  # TODO(sgm): check why is bfloat16
                    device=input_ids.device)
            # broadcast across pp ranks
            torch.distributed.broadcast(tensor=logits,
                                        src=mpu.get_pipeline_model_parallel_last_rank(),
                                        group=mpu.get_pipeline_model_parallel_group(),
                                        async_op=False)

        # (bs, seqlen', hidden_size) -> (bs, seqlen', 1) -> (bs, seqlen')
        token_level_rewards = logits
        # find the last token reward
        ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)  # (bs, 1)
        rewards = torch.gather(token_level_rewards, dim=1, index=ends)  # (bs, 1)

        if self.use_different_tokenizer:
            data.batch.update(ori_values)
            input_ids = ori_values['input_ids']
            attention_mask = ori_values['attention_mask']
            position_ids = ori_values['position_ids']

        token_level_rewards = rewards.expand(attention_mask.shape[0], attention_mask.shape[1])  # (bs, ori_seqlen)

        # assign last valid token reward to ori position
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bs,)
        eos_mask = torch.zeros_like(attention_mask)
        eos_mask[torch.arange(batch_size), eos_mask_idx] = 1.

        token_level_rewards = token_level_rewards * eos_mask
        token_level_rewards = token_level_rewards[:, -response_length:]

        # Apply cosine scaling to the rewards
        logger = logging.getLogger("CosineMegatronRewardModel")
        logger.info(f"[CosineMegatronRewardModel] Applying cosine scaling to rewards")

        # Get generation lengths for cosine scaling
        generation_lengths = []
        for i in range(batch_size):
            # Calculate the actual generation length (excluding padding)
            gen_length = attention_mask[i, -response_length:].sum().item()
            generation_lengths.append(gen_length)

        # Apply cosine scaling to each reward
        for i in range(batch_size):
            original_reward = token_level_rewards[i, -1].item()  # Get the last token reward
            gen_length = generation_lengths[i]

            # Apply cosine scaling
            cosine_reward = self._apply_cosine_scaling(original_reward, gen_length)

            # Update the reward tensor
            token_level_rewards[i, -1] = cosine_reward

            if i < 3:  # Log first few examples
                logger.info(f"[CosineMegatronRewardModel] Sample {i}: original={original_reward:.3f}, "
                          f"gen_length={gen_length}, cosine={cosine_reward:.3f}")

        if self.config.param_offload:
            self.offload_params_to_cpu()
        else:
            # add empty cache after each compute
            torch.cuda.empty_cache()

        batch = TensorDict({'rm_scores': token_level_rewards}, batch_size=input_ids.shape[0])

        return DataProto(batch=batch)

    def forward_batch(self, data: DataProto):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.batch = data.batch.contiguous()
        broadcast_dict_tensor(data.batch,
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              group=mpu.get_pipeline_model_parallel_group())

        # split into micro-batches
        if self.config is not None and 'ppo_micro_batch_size_per_gpu' in self.config:
            infer_batch_size = self.config.ppo_micro_batch_size_per_gpu
        else:
            infer_batch_size = data.batch.batch_size[0]

        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
        batches = split_dict_tensor_into_batches(data.batch, batch_size=infer_batch_size)

        def loss_func(output):
            return output

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            return model(batch)

        return get_forward_backward_func(
            forward_step,
            self.reward_model_module,
            batches,
            loss_func,
            forward_only=True,
            tensor_shape=compute_transformers_input_shapes(self.megatron_config),
            decoder_sequence_length=self.megatron_config.sequence_parallel,
            dtype=torch.bfloat16,
        )

    def offload_params_to_cpu(self):
        for module in self.reward_model_module:
            for param in module.parameters():
                if param.device.type == 'cuda':
                    param.data = param.data.cpu()

    def load_params_to_cuda(self):
        for module in self.reward_model_module:
            for param in module.parameters():
                if param.device.type == 'cpu':
                    param.data = param.data.cuda()
