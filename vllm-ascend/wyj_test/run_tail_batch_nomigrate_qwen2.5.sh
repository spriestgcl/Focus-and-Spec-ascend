#!/usr/bin/env bash
set -euo pipefail

# cd /home/Focus-and-Spec

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES
# IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
# DP_SIZE="${#_GPU_IDS[@]}"

export ASCEND_RT_VISIBLE_DEVICES="2,3,4,5,12,13,14,15"
DP_SIZE=8

# MODEL_PATH="/home/models/Qwen/Qwen3-8B"
MODEL_PATH=/mnt/model/model_weights/models/Qwen/Qwen2.5-7B
# DATASET_PATH="/home/data/gsm8k/train.parquet"
DATASET_PATH="/vllm-workspace/vllm-ascend/wyj_test/deepmath-103k/train.parquet"

ENABLE_THINKING="true"
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  THINKING_FLAG="--enable-thinking"
  ROLLOUT_TEMPERATURE=0.6
  ROLLOUT_TOP_P=0.95
  ROLLOUT_TOP_K=20
  ROLLOUT_MIN_P=0.0
  LOG_SUFFIX=thinking
else
  THINKING_FLAG="--disable-thinking"
  ROLLOUT_TEMPERATURE=0.7
  ROLLOUT_TOP_P=0.8
  ROLLOUT_TOP_K=20
  ROLLOUT_MIN_P=0.0
  LOG_SUFFIX=nothinking
fi

DATASET_INDICES="2,6,11,16,23,29,34,37,43,48,51,59,63,69,72,75,81,87,91,96,103,106,113,115,122,129,133,135,140,145,150,158"

OUTPUT_DIR="/vllm-workspace/vllm-ascend/wyj_test/outputs"
LOG_DIR="/vllm-workspace/vllm-ascend/wyj_test/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
DATASET_DIR_NAME="$(basename "$(dirname "${DATASET_PATH}")")"
DATASET_FILE_NAME="$(basename "${DATASET_PATH}")"
DATASET_STEM="${DATASET_FILE_NAME%.*}"

GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192
MAX_TOKENS=8192
REPEAT_PER_PROMPT=8

SYNC_EVERY_STEPS=256
REPORT_EVERY_DECODE_STEPS=1024
TAIL_CONCENTRATE_REQUEST_KEYS="2-0,6-2,6-5,16-6,29-2,29-4,34-1,72-0,72-7,103-1,103-2,113-3,113-6,158-1,158-4"

RUN_NAME="${MODEL_NAME}_${DATASET_DIR_NAME}_${DATASET_STEM}_tailbatch32_dp${DP_SIZE}_${LOG_SUFFIX}_8k_nomigrate"

python /vllm-workspace/vllm-ascend/wyj_test/tail_batch_migration.py \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET_PATH}" \
  --dataset-indices "${DATASET_INDICES}" \
  --output-path "${OUTPUT_DIR}/${RUN_NAME}.jsonl" \
  --dp-size "${DP_SIZE}" \
  --tp-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 128 \
  --repeat-per-prompt "${REPEAT_PER_PROMPT}" \
  --no-migration-enabled \
  --tail-concentrate-enabled \
  --tail-concentrate-top-k 15 \
  --tail-concentrate-ranks "0,1" \
  --tail-concentrate-request-keys "${TAIL_CONCENTRATE_REQUEST_KEYS}" \
  "${THINKING_FLAG}" \
  --temperature "${ROLLOUT_TEMPERATURE}" \
  --top-p "${ROLLOUT_TOP_P}" \
  --top-k "${ROLLOUT_TOP_K}" \
  --min-p "${ROLLOUT_MIN_P}" \
  --max-tokens "${MAX_TOKENS}" \
  --sync-every-steps "${SYNC_EVERY_STEPS}" \
  --report-every-decode-steps "${REPORT_EVERY_DECODE_STEPS}" 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
