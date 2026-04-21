set -euo pipefail
set -x

export RAY_DEDUP_LOGS=0
# NPU 环境下先关闭 torch compile/dynamo，优先保证兼容性和稳定性。
export TORCHDYNAMO_DISABLE=1
# 当前仓库的 vLLM-ascend LoRA 路径需要这个开关来接管 LoRA 请求。
export VERL_ENABLE_VLLM_LORA_HIJACK=1
# redundant 推理/在线迁移实验开关。
export VERL_REDUNDANT_ONLINE_MIGRATION_ENABLED=true
# redundant 场景下放宽 NCCL 心跳超时，避免长轮次生成时被误判卡死。
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export REDUNDANT_PARTIAL_PROMPT_ACCEPT_RATIO=${REDUNDANT_PARTIAL_PROMPT_ACCEPT_RATIO:-0.75}

# 以下改成环境变量默认值，方便在 NPU 机器上反复覆盖调参，不必每次改脚本正文。
TRAIN_FILE=${TRAIN_FILE:-/home/deepmath/train_1k.parquet}
TEST_FILE=${TEST_FILE:-/home/deepmath/train_1k.parquet}
MODEL_PATH=${MODEL_PATH:-/mnt/model/gcl/Qwen2.5-7B-Instruct}
LOG_FILE=${LOG_FILE:-redundant_npu_train.log}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TP_SIZE=${TP_SIZE:-1}
# redundant 模式会在原始 prompt batch 上额外扩一部分请求。
# 当前配置下 redundancy_p=0.25、4 张 NPU，因此 train_batch_size 需要取 16 的倍数，
# 这样 expanded_requests 才能被 4 个 rollout worker 平均切分。
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}
ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-True}
# 下面这组是调试/冒烟跑开关；默认关闭，主脚本直接使用全量训练/验证集。
FAST_SMOKE_RUN=${FAST_SMOKE_RUN:-0}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-64}
# 下面这组是 NPU 版新增的 LoRA 相关参数。
USE_LORA=${USE_LORA:-1}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
# NPU/Ray 分布式启动时显式指定 master 地址和端口，避免多进程初始化不一致。
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29666}
export MASTER_ADDR
export MASTER_PORT

# 默认只暴露指定 NPU 卡，避免误占满整机。
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-12,13,14,15}

# 预检查：在真正起训练前，先验证 vllm-ascend 关键模块能否正常 import，
# 这样环境有问题时可以快速失败，而不是等模型和数据初始化很久后再报错。
if [ "${VLLM_ASCEND_PREFLIGHT:-0}" = "1" ]; then
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
        "Fix compatibility before launching training."
    )

print("[preflight] vllm-ascend import checks passed.")
PY
fi

EXTRA_DATA_ARGS=()
if [ "${FAST_SMOKE_RUN}" = "1" ]; then
  # 冒烟模式下只抽少量训练/验证样本，并关闭全量超长样本过滤，缩短调试时间。
  EXTRA_DATA_ARGS+=(
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"
    data.filter_overlong_prompts=False
  )
else
  # 正常训练时恢复为全量数据过滤逻辑；redundant 脚本额外限制验证样本，避免验证阶段太慢。
  EXTRA_DATA_ARGS+=(
    data.filter_overlong_prompts=True
    data.val_max_samples=256
  )
fi

EXTRA_MODEL_ARGS=()
if [ "${USE_LORA}" = "1" ]; then
  # NPU 版新增：把 LoRA 配置通过命令行传给 actor/model。
  EXTRA_MODEL_ARGS+=(
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    actor_rollout_ref.model.target_modules='all-linear'
  )
fi

if [ "${USE_LORA}" = "1" ]; then
  # 当前分支的 vLLM TensorLoRARequest 会读取固定本地 adapter 目录。
  # 这里提前生成一个占位 adapter_config，避免运行时去做 HF/远端查找。
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

# 相比 NPU 基础版 run_grpo.sh，这里额外打开 redundant 推理实验：
# 1. 启用 activation offload，给 redundant 轮次预留更多显存。
# 2. rollout 采样使用 temperature/top-p/top-k。
# 3. 打开 redundant_inference_enabled 及其相关在线迁移参数。
# 4. 禁用自动恢复，避免误加载不同 world size 的旧 checkpoint。
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
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
    actor_rollout_ref.rollout.layered_summon="${ROLLOUT_LAYERED_SUMMON}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
    actor_rollout_ref.nccl_timeout=3600 \
    actor_rollout_ref.rollout.redundant_inference_enabled=True \
    actor_rollout_ref.rollout.redundancy_p=0.25 \
    actor_rollout_ref.rollout.redundant_partial_prompt_accept_ratio="${REDUNDANT_PARTIAL_PROMPT_ACCEPT_RATIO}" \
    actor_rollout_ref.rollout.redundant_short_rounds_per_cycle=4 \
    actor_rollout_ref.rollout.redundant_flush_tail_at_epoch_end=True \
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
    trainer.resume_mode=disable \
    trainer.default_local_dir=/root/Focus-and-Spec-ascend/verl/wyj_test/checkpoints/gsm8k_redundant_fresh \
    trainer.total_epochs=1 \
    trainer.device=npu \
    "$@" 2>&1 | tee "${LOG_FILE}"
