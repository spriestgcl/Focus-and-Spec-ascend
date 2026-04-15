#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export ASCEND_RT_VISIBLE_DEVICES="2,3,4,5,12,13,14,15"
# /mnt/model/lcw/slai-ascend-auto-adapt/adaptations/qwen_qwen2_5_14b_instruct/models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
# MODEL_PATH="/mnt/model/lcw/slai-ascend-auto-adapt/adaptations/qwen_qwen2_5_14b_instruct/models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MODEL_PATH=/mnt/model/model_weights/models/Qwen/Qwen2.5-7B
# DATASET_PATH="/home/data/gsm8k/train.parquet"
DATASET_PATH="/vllm-workspace/vllm-ascend/wyj_test/deepmath-103k/train.parquet"

# Qwen3 sampling settings are tied to thinking mode.
# - thinking=true  -> temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
# - thinking=false -> temperature=0.7, top_p=0.8,  top_k=20, min_p=0.0
ENABLE_THINKING="false"
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

OUTPUT_DIR="/vllm-workspace/vllm-ascend/wyj_test/outputs"
LOG_DIR="/vllm-workspace/vllm-ascend/wyj_test/logs"
DP_SIZE=8
REPEAT_PER_PROMPT=8
REDUNDANCY_P=0.25
export REDUNDANT_INFERENCE_ENABLED=false
export TAIL_PARTIAL_COMPLETE_ENABLED=false
export TAIL_PARTIAL_COMPLETE_RATIO=0.6

if [[ "${REDUNDANT_INFERENCE_ENABLED}" == "true" ]]; then
  REDUNDANT_TAG="redun"
else
  REDUNDANT_TAG="vanilla"
fi

if [[ "${TAIL_PARTIAL_COMPLETE_ENABLED}" == "true" ]]; then
  PARTIAL_COMPLETE_TAG="tailpartial${TAIL_PARTIAL_COMPLETE_RATIO/./p}"
else
  PARTIAL_COMPLETE_TAG="tailstrict"
fi

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
DATASET_DIR_NAME="$(basename "$(dirname "${DATASET_PATH}")")"
DATASET_FILE_NAME="$(basename "${DATASET_PATH}")"
DATASET_STEM="${DATASET_FILE_NAME%.*}"
RUN_NAME="${MODEL_NAME}_${DATASET_DIR_NAME}_${DATASET_STEM}_dp${DP_SIZE}_${LOG_SUFFIX}_${REDUNDANT_TAG}_${PARTIAL_COMPLETE_TAG}_rp${REDUNDANCY_P}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

python /vllm-workspace/vllm-ascend/wyj_test/dp_inference.py \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET_PATH}" \
  --output-path "${OUTPUT_DIR}/${RUN_NAME}.jsonl" \
  --dp-size "${DP_SIZE}" \
  --tp-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --batch-size 32 \
  --redundancy-p "${REDUNDANCY_P}" \
  --repeat-per-prompt "${REPEAT_PER_PROMPT}" \
  "${THINKING_FLAG}" \
  --temperature "${ROLLOUT_TEMPERATURE}" \
  --top-p "${ROLLOUT_TOP_P}" \
  --top-k "${ROLLOUT_TOP_K}" \
  --min-p "${ROLLOUT_MIN_P}" \
  --max-tokens 8192 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
