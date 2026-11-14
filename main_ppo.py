"""
L1's entry point, on verl 0.9: verl's own PPO main, with L1's config.

config/l1_trainer.yaml is verl's `ppo_trainer` plus L1's `reward_config`
block and its reward manager (reward_manager.py).
"""

import hydra
from verl.trainer import main_ppo as verl_main_ppo


@hydra.main(config_path="config", config_name="l1_trainer", version_base=None)
def main(config):
    """Runs verl's PPO main on the composed config."""
    verl_main_ppo.main.__wrapped__(config)


if __name__ == "__main__":
    main()
