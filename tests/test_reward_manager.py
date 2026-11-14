"""The ported reward manager gives L1's rewards on verl's DataProto."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from reward_manager import RewardManager

WORDS = {0: "", 1: "<think>\n", 2: "hm ", 3: "</think>", 4: "\\boxed{7}"}
"""A toy vocabulary: pad, the prompt's opening, thinking, the answer."""


class Tokenizer:
    """Decodes the toy vocabulary."""

    def decode(self, ids):
        """Joins the words of `ids`."""
        return "".join(WORDS[int(i)] for i in ids)


def item(think_tokens, num_tokens, answer="7"):
    """One left-padded prompt and a response of `think_tokens` + 2 tokens."""
    prompt = torch.tensor([0, 0, 1])
    response = torch.tensor([2] * think_tokens + [3, 4] + [0, 0])
    used = think_tokens + 2
    attention_mask = torch.cat(
        [torch.tensor([0, 0, 1]), torch.ones(used), torch.zeros(2)]
    ).long()
    return DataProto.from_dict(
        tensors={
            "prompts": prompt[None],
            "responses": response[None],
            "attention_mask": attention_mask[None],
        },
        non_tensors={
            "reward_model": np.array(
                [{"ground_truth": answer, "num_tokens": num_tokens}],
                dtype=object,
            ),
        },
    )


def score(data, linear_reward=True):
    """Scores `data` with L1-Exact's reward settings."""
    config = OmegaConf.create(
        {
            "reward_config": {
                "alpha": 0.0003,
                "sigmoid_reward": False,
                "linear_reward": linear_reward,
                "multiplier_reward": False,
            }
        }
    )
    manager = RewardManager(config, Tokenizer(), None)
    result = manager.loop.run_until_complete(manager.run_single(data))
    return result["reward_score"]


def test_exact_target_met_and_correct():
    assert score(item(98, 100)) == pytest.approx(1.0)


def test_exact_penalty_is_linear_in_the_gap():
    assert score(item(1998, 1000)) == pytest.approx(1.0 - 0.0003 * 1000)


def test_wrong_answer_keeps_the_length_term():
    assert score(item(98, 600, answer="8")) == pytest.approx(-0.0003 * 500)


def test_max_under_budget_earns_more_than_half():
    used = 500
    expected = 1 + (min(1, 0.0003 * (1000 - used)) + 1) / 2
    assert score(item(used - 2, -1000)) == pytest.approx(expected)


def test_max_over_budget_earns_less_than_half():
    used = 3000
    expected = 1 + (max(-1, -0.0003 * (used - 1000)) + 1) / 2
    assert score(item(used - 2, -1000)) == pytest.approx(expected)


def test_no_length_reward_scores_correctness_alone():
    assert score(item(98, 100), linear_reward=False) == 1.0
