set -euo pipefail
set -x

export RAY_DEDUP_LOGS=0
export TORCHDYNAMO_DISABLE=1
export VERL_ENABLE_VLLM_LORA_HIJACK=1

TRAIN_FILE=${TRAIN_FILE:-/home/deepmath/train.parquet}
TEST_FILE=${TEST_FILE:-/home/deepmath/test.parquet}
MODEL_PATH=${MODEL_PATH:-/mnt/model/gcl/Qwen2.5-7B-Instruct}
LOG_FILE=${LOG_FILE:-0306_2a100_qwen3-1.7b_pergpubs8_totalbs16_8k_origin_verl.log}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TP_SIZE=${TP_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}
ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-True}
FAST_SMOKE_RUN=${FAST_SMOKE_RUN:-1}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-64}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-64}
USE_LORA=${USE_LORA:-1}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29666}
export MASTER_ADDR
export MASTER_PORT

# Prefer using currently idle NPUs for quick validation.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-12,13}

# Fast-fail preflight: catch vllm-ascend/vllm import-layout issues in seconds,
# instead of waiting for full data/model initialization before crashing.
if [ "${VLLM_ASCEND_PREFLIGHT:-1}" = "1" ]; then
  PYTHONUNBUFFERED=1 python3 - <<'PY'
import importlib
import traceback

checks = [
    "vllm_ascend.platform",
    "vllm_ascend.worker",
    "vllm_ascend.worker.worker_v1",
]
failed = []
for mod in checks:
    try:
        importlib.import_module(mod)
        print(f"[preflight] ok: {mod}")
    except Exception as exc:
        failed.append((mod, exc))
        print(f"[preflight] fail: {mod}: {type(exc).__name__}: {exc}")
        traceback.print_exc()

if failed:
    raise SystemExit(
        f"[preflight] import checks failed ({len(failed)} modules). "
        "Fix compatibility before launching training.")

print("[preflight] vllm-ascend import checks passed.")
PY
fi

EXTRA_DATA_ARGS=()
if [ "${FAST_SMOKE_RUN}" = "1" ]; then
  # Keep each debug iteration short: sample tiny data and skip expensive
  # full-dataset overlong-prompt filtering in smoke mode.
  EXTRA_DATA_ARGS+=(
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"
    data.filter_overlong_prompts=False
  )
else
  EXTRA_DATA_ARGS+=(
    data.filter_overlong_prompts=True
  )
fi

EXTRA_MODEL_ARGS=()
if [ "${USE_LORA}" = "1" ]; then
  EXTRA_MODEL_ARGS+=(
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    actor_rollout_ref.model.target_modules='all-linear'
  )
fi

if [ "${USE_LORA}" = "1" ]; then
  # vLLM TensorLoRARequest in this fork uses a fixed local adapter path.
  # Ensure the placeholder adapter config exists to avoid HF/network lookup.
  python3 - <<'PY'
import json
import os
from pathlib import Path

lora_rank = int(os.environ.get("LORA_RANK", "16"))
lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
adapter_dir = Path("simon_lora_path")
adapter_dir.mkdir(parents=True, exist_ok=True)
adapter_cfg = {
    "base_model_name_or_path": "",
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "lora_alpha": lora_alpha,
    "lora_dropout": 0.0,
    "peft_type": "LORA",
    "r": lora_rank,
    "target_modules": "all-linear",
    "task_type": "CAUSAL_LM",
}
(adapter_dir / "adapter_config.json").write_text(
    json.dumps(adapter_cfg, ensure_ascii=True, indent=2), encoding="utf-8"
)
PY
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    "${EXTRA_DATA_ARGS[@]}" \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    "${EXTRA_MODEL_ARGS[@]}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
    actor_rollout_ref.rollout.layered_summon="${ROLLOUT_LAYERED_SUMMON}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.total_epochs=1 \
    trainer.device=npu \
    "$@" 2>&1 | tee "${LOG_FILE}"
