"""
L1's RewardManager (main_ppo.py), on the reward loop of verl 0.9.

As in L1, each response is scored by `math_reward_fn` on the decoded prompt
and response, with the row's target `reward_model.num_tokens` (negative for
a maximum, -1 for none) and the response's length in tokens. L1 scored a
whole batch at once; verl now asks for one response at a time.
"""

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import (
    RewardManagerBase,
)

from math_reward import math_reward_fn
from rewards_types import RewardConfig


class RewardManager(RewardManagerBase):
    """L1's reward manager: correctness plus the LCPO length reward."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        """Reads L1's `reward_config` block from the run's config."""
        super().__init__(config, tokenizer, compute_score)
        self.reward_config = RewardConfig(
            alpha=config.reward_config.alpha,
            sigmoid_reward=config.reward_config.sigmoid_reward,
            linear_reward=config.reward_config.linear_reward,
            multiplier_reward=config.reward_config.multiplier_reward,
        )

    async def run_single(self, data: DataProto) -> dict:
        """Scores one response as L1's `process_item` did."""
        data_item = data[0]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        attention_mask = data_item.batch["attention_mask"]
        valid_prompt_length = attention_mask[:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        response_ids = data_item.batch["responses"]
        valid_response_length = attention_mask[prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        sequences_str = self.tokenizer.decode(
            torch.cat((valid_prompt_ids, valid_response_ids))
        )
        reward_model = data_item.non_tensor_batch["reward_model"]
        result = await self.loop.run_in_executor(
            None,
            lambda: math_reward_fn(
                solution_str=sequences_str,
                ground_truth=reward_model["ground_truth"],
                num_tokens=reward_model["num_tokens"],
                valid_response_length=valid_response_length,
                reward_config=self.reward_config,
                return_delta_score=True,
            ),
        )
        score, delta_score = (
            result if isinstance(result, tuple) else (result, 0.0)
        )
        return {
            "reward_score": float(score),
            "reward_extra_info": {"delta_score": float(delta_score)},
        }
