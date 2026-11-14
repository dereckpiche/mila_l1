# mila_l1

[L1](https://github.com/cmu-l3/l1) (Aggarwal & Welleck, 2025, *Controlling How
Long A Reasoning Model Thinks With Reinforcement Learning*) on a current stack:
verl 0.9.1, vLLM 0.24, torch 2.11, transformers 5.9, and Qwen3.5.

L1 is the source. Its reward (`math_reward.py`, `rewards_types.py`), grader
(`utils.py`), data (`deepscaler/data`, `scripts/data/deepscaler_dataset.py`)
and training scripts (`scripts/train/run_l1_{exact,max}.sh`) are L1's, with
the same flags and hyperparameters. Only what the new verl requires changed:

| L1 | here |
|---|---|
| `main_ppo.py` builds `RayPPOTrainer(reward_fn=RewardManager(...))` | `main_ppo.py` runs verl's own PPO main on `config/l1_trainer.yaml` |
| `RewardManager` scores a batch | `reward_manager.py`: the same scoring, one response at a time, on verl's reward loop |
| `config/ppo_trainer.yaml`, a copy of old verl's | `config/l1_trainer.yaml`: verl's `ppo_trainer` plus L1's `reward_config` |
| `rollout.val_temperature`, `rollout.n_val` | `rollout.val_kwargs.{temperature,n,do_sample}` |
| `VLLM_ATTENTION_BACKEND=XFORMERS` | not needed |
| starts from `DeepScaleR-1.5B-Preview` | starts from Qwen3.5, with thinking on (`enable_thinking`) |

Command-line overrides come last in the scripts, so a run on fewer GPUs than
L1's eight can change any setting.

## Rewards

With `n` the response's tokens and `N` the target in the prompt:

- L1-Exact ("Think for N tokens."): `correct − α·|N − n|`.
- L1-Max ("Think for maximum N tokens."): `correct + (clip(α·(N − n), −1, 1) + 1) / 2`.

`correct` is 1 for a right `\boxed{}` answer after `</think>`, else 0, and
α = 0.0003.

## Run

```bash
uv sync
# data/exact (L1-Exact) and data/max (L1-Max: half exact, half max) are in the
# repo; scripts/data/deepscaler_dataset.py writes them from deepscaler/data.
bash scripts/train/run_l1_exact.sh --model Qwen/Qwen3.5-2B
bash scripts/train/run_l1_max.sh \
  --model checkpoints/deepscaler/l1_exact/global_step_700/actor/huggingface \
  reward_config.linear_reward=True reward_config.alpha=0.0003
```

L1 is MIT-licensed (LICENSE).
