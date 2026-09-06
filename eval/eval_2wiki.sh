#!/usr/bin/env bash
# CONCURRENT_PROCESS_NUM=10 outperforms CONCURRENT_PROCESS_NUM=1
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="${CUDA_HOME}/bin:${PATH}"

PROJECT_ROOT="$(pwd)"


#### Below are what need to be changed for different run
URL=${SUBAGENT_URL:-"http://127.0.0.1:8012/v1"} # SGLang Router URL
SUBAGENT_MODEL=${SUBAGENT_MODEL:-"Qwen3.5-4B"}
CHUNK_MAX_LENGTH=${CHUNK_MAX_LENGTH:-4096}
CONCURRENT_PROCESS_NUM=${CONCURRENT_PROCESS_NUM:-6}
DATA_PATH="../data/2wiki_val" # Select val data source here, hotpotqa or 2wiki
OUT_DIR="${PROJECT_ROOT}/2wiki_eval_outputs/eval_outputs_2wiki_chunk${CHUNK_MAX_LENGTH}"
NUM_SAMPLES=1024
#######

TOKENIZER_PATH=${TOKENIZER_PATH:-"../Qwen3.5-4B"}
MODELS=(
  "${MODEL_PATH:-../checkpoints/Qwen3.5-4B}"
)

TOKENIZER_ID="$(basename "${TOKENIZER_PATH}")"
PREPROCESSED_PATH="${DATA_PATH}/eval_preprocessed_${TOKENIZER_ID}_${CHUNK_MAX_LENGTH}.pkl"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${PREPROCESSED_PATH}" ]]; then
  echo "Preprocessed dataset not found, generating: ${PREPROCESSED_PATH}"
  python "../preprocess_eval_data.py" \
    --test-folder "${DATA_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --chunk-max-length "${CHUNK_MAX_LENGTH}"
fi

for MODEL_PATH in "${MODELS[@]}"; do
  TAG="$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")_$(basename "${MODEL_PATH}")_${NUM_SAMPLES}"
  OUTPUT_PATH="${OUT_DIR}/${TAG}.jsonl"
  EXTRA_ARGS=()
  if [[ -n "${NUM_SAMPLES}" ]]; then
    EXTRA_ARGS+=(--num-samples "${NUM_SAMPLES}")
  fi
  echo "Running offline eval for: ${MODEL_PATH}"
  python "${PROJECT_ROOT}/eval.py" \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --data-path "${DATA_PATH}" \
    --chunk-max-length "${CHUNK_MAX_LENGTH}" \
    --output-path "${OUTPUT_PATH}" \
    --max-iterations 16 \
    --subagent-base-url "${URL}" \
    --subagent-model "${SUBAGENT_MODEL}" \
    --subagent-temperature 0.7 \
    --concurrent-process-num "${CONCURRENT_PROCESS_NUM}" \
    "${EXTRA_ARGS[@]}"
done