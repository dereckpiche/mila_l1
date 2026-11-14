#!/bin/bash
# L1-Max (L1's run_l1_max.sh), on verl 0.9 with Qwen3.5: starts from L1-Exact.
# Data from:
#   python scripts/data/deepscaler_dataset.py --use_both_both \
#     --local_dir $HOME/deepscaler/data7
# L1 passed its reward_config on the command line, after the script's own
# arguments, e.g. reward_config.linear_reward=True reward_config.alpha=0.0003.
set -x

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="checkpoints/deepscaler/l1_exact/global_step_700/actor/huggingface"
fi

export PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH}

# Train over a single node: L1's 8 A100-80GB GPUs, or every GPU Slurm gives the job.
python3 main_ppo.py \
    algorithm.adv_estimator=grpo \
    data.train_files=data/max/train.parquet \
    data.val_files=data/max/aime.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4500 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='mila_l1' \
    trainer.experiment_name='l1_max' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${SLURM_GPUS_ON_NODE:-8} \
    ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_ON_NODE:-null} \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 "${@:1}"
