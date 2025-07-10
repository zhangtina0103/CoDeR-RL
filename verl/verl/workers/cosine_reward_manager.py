import math
import logging
import torch
from verl import DataProto
from verl.utils.custom_reward import compute_delta_reward, compute_clip_reward


class CosineRewardManager:
    """Reward manager supporting Delta, Clip, and Cosine mechanisms.

    Integrates cosine reward logic from demystify-long-cot for length-based scaling.
    """
    def __init__(self, tokenizer, num_examine, mechanism='cosine', eta=None, compute_score=None,
                 # Cosine reward parameters
                 min_value_wrong=-1.0,
                 max_value_wrong=-0.1,
                 min_value_correct=0.1,
                 max_value_correct=1.0,
                 max_len=2048,
                 exceed_length_penalty=-1.0,
                 repetition_max_penalty=0.0,
                 repetition_ngram_size=3):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.mechanism = mechanism
        self.eta = eta
        self.compute_score = compute_score

        # Cosine reward parameters
        self.min_value_wrong = min_value_wrong
        self.max_value_wrong = max_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_correct = max_value_correct
        self.max_len = max_len
        self.exceed_length_penalty = exceed_length_penalty
        self.repetition_max_penalty = repetition_max_penalty
        self.repetition_ngram_size = repetition_ngram_size

        # Validate wrong values are negative
        if min_value_wrong > 0 or max_value_wrong > 0:
            raise ValueError("Wrong values should not be positive")

        logger = logging.getLogger("CosineRewardManager")
        logger.info(
            f"Initialized cosine reward manager with"
            f" min_value_wrong: {min_value_wrong}, max_value_wrong: {max_value_wrong},"
            f" min_value_correct: {min_value_correct}, max_value_correct: {max_value_correct},"
            f" max_len: {max_len}, exceed_length_penalty: {exceed_length_penalty},"
            f" repetition_max_penalty: {repetition_max_penalty}, repetition_ngram_size: {repetition_ngram_size}")

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

    def _apply_cosine_scaling(self, score: float, gen_length: int) -> float:
        """Apply cosine scaling based on generation length.

        The general idea is:
        - Shorter correct solutions should be rewarded over longer ones.
        - Longer wrong solutions should be rewarded over shorter ones.
        - Shorter solutions should be more risk averse (wrong penalized more than correct rewarded).
        """
        if gen_length >= self.max_len:
            return self.exceed_length_penalty

        # Determine if this is a correct or wrong answer
        if score >= 0.5:  # Assuming 0.5 is the threshold for correct
            min_value = self.min_value_correct
            max_value = self.max_value_correct
            rep_penalty = 0
        else:
            # For wrong answers, swap min/max for cosine formula to work with negative numbers
            min_value = self.max_value_wrong
            max_value = self.min_value_wrong
            rep_penalty = self._get_repetition_penalty("")  # Could pass actual generation here

        progress = gen_length / self.max_len
        cosine = math.cos(progress * math.pi)
        r = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        r += rep_penalty

        return r

    def __call__(self, data: DataProto):
        logger = logging.getLogger("CosineRewardManager")
        logger.info(f"[CosineRewardManager] Using mechanism: {self.mechanism}")

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # Get base score
            base_score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # Apply cosine scaling if using cosine mechanism
            if self.mechanism == 'cosine':
                final_score = self._apply_cosine_scaling(base_score, valid_response_length)
            elif self.mechanism == 'cosine+clip':
                cosine_score = self._apply_cosine_scaling(base_score, valid_response_length)
                eta = self.eta if self.eta is not None else cosine_score
                final_score = compute_clip_reward(torch.tensor([cosine_score]), eta).item()
            elif self.mechanism == 'cosine+delta':
                cosine_score = self._apply_cosine_scaling(base_score, valid_response_length)
                final_score = compute_delta_reward(torch.tensor([cosine_score])).item()
            elif self.mechanism == 'cosine+clip+delta':
                cosine_score = self._apply_cosine_scaling(base_score, valid_response_length)
                eta = self.eta if self.eta is not None else cosine_score
                clipped_score = compute_clip_reward(torch.tensor([cosine_score]), eta)
                final_score = compute_delta_reward(clipped_score).item()
            else:
                # Fall back to original mechanisms
                final_score = base_score

            reward_tensor[i, valid_response_length - 1] = final_score

        logger.info(f"[CosineRewardManager] Raw reward_tensor sample: {reward_tensor.flatten()[:10]}")

        # Apply additional mechanisms if not already applied
        if self.mechanism == 'clip':
            eta = self.eta
            if eta is None:
                eta = reward_tensor.mean()
            logger.info(f"[CosineRewardManager] Using eta: {eta}")
            reward_tensor = compute_clip_reward(reward_tensor, eta)
        elif self.mechanism == 'delta':
            reward_tensor = compute_delta_reward(reward_tensor)
        elif self.mechanism == 'clip+delta':
            eta = self.eta
            if eta is None:
                eta = reward_tensor.mean()
            logger.info(f"[CosineRewardManager] Using eta: {eta}")
            reward_tensor = compute_clip_reward(reward_tensor, eta)
            reward_tensor = compute_delta_reward(reward_tensor)

        logger.info(f"[CosineRewardManager] Post-processed reward_tensor sample: {reward_tensor.flatten()[:10]}")
        return reward_tensor
