#!/bin/bash
#
# E2C Model Evaluation Script
# Evaluate trained E2C models on math/medical benchmarks
#
# Usage:
#   bash scripts/eval.sh                                    # Quick test (GSM8K)
#   bash scripts/eval.sh --dataset math --sample 8         # All math benchmarks
#   bash scripts/eval.sh --dataset med --sample 4          # Medical benchmarks
#   bash scripts/eval.sh --model path/to/checkpoint        # Custom checkpoint

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Default configuration
MODEL_PATH="${MODEL_PATH:-TingheOliver/Explore-Execute-Chain-Qwen}"
SUBFOLDER="${SUBFOLDER:-Qwen3-8B-E2C-SFT-RL}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

DATASET="${DATASET:-gsm8k}"
SAMPLE_NUM="${SAMPLE_NUM:-1}"
BATCH_SIZE="${BATCH_SIZE:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:--1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

N_GPUS="${N_GPUS:-1}"
SEED="${SEED:-0}"
SAVE_PATH="${SAVE_PATH:-evaluation/e2c-eval}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL_PATH="$2"; shift 2 ;;
        --subfolder)  SUBFOLDER="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --sample)     SAMPLE_NUM="$2"; shift 2 ;;
        --gpus)       N_GPUS="$2"; shift 2 ;;
        --temp)       TEMPERATURE="$2"; shift 2 ;;
        --save-path)  SAVE_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "E2C Evaluation Configuration"
echo "=========================================="
echo "Model Path:     $MODEL_PATH"
[ -n "$SUBFOLDER" ] && echo "Subfolder:      $SUBFOLDER"
[ -n "$CHECKPOINT_PATH" ] && echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Dataset(s):     $DATASET"
echo "Sample Num:     $SAMPLE_NUM"
echo "Temperature:    $TEMPERATURE"
echo "GPUs:           $N_GPUS"
echo "Save Path:      $SAVE_PATH"
echo "=========================================="
echo ""

mkdir -p "$SAVE_PATH"

# Build model argument
if [ -n "$SUBFOLDER" ]; then
    MODEL_ARG="model.model_path=$MODEL_PATH model.subfolder=$SUBFOLDER"
else
    MODEL_ARG="model.model_path=$MODEL_PATH"
fi
[ -n "$CHECKPOINT_PATH" ] && MODEL_ARG="$MODEL_ARG model.checkpoint_path=$CHECKPOINT_PATH"

export PYTHONPATH="${PROJECT_ROOT}/e2c:${PYTHONPATH}"

echo "Starting evaluation..."
echo ""

torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=29500 \
    e2c/inference/eval.py \
    --config-path="../config" \
    --config-name="eval" \
    $MODEL_ARG \
    eval.dataset="['$DATASET']" \
    eval.sample_num=$SAMPLE_NUM \
    eval.batch_size=$BATCH_SIZE \
    eval.max_new_tokens=$MAX_NEW_TOKENS \
    eval.temperature=$TEMPERATURE \
    eval.top_p=$TOP_P \
    eval.seed=$SEED \
    eval.save_path="$SAVE_PATH" \
    eval.backend=hf

EVAL_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation complete."
    echo "=========================================="
    echo "Results saved to: $SAVE_PATH"
    echo "  - ${SAVE_PATH}/${DATASET}/result_${SEED}_merged.json"
    echo "  - ${SAVE_PATH}/${DATASET}/static_${SEED}_merged.json"
else
    echo "Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "=========================================="
    exit $EVAL_EXIT_CODE
fi
