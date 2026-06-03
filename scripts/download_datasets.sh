#!/usr/bin/env bash
# Download E2C datasets from HuggingFace.
# Usage: bash download_datasets.sh [--dataset sft|rl|eval|all] [--mirror]

set -eo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

DATASET_TYPE="all"
RAW_DIR="${DATA_DIR}/raw"
EVAL_DIR="${DATA_DIR}/evaluation"

USE_MIRROR=false
if [[ -n "${HF_ENDPOINT}" && "${HF_ENDPOINT}" == *"hf-mirror.com"* ]]; then
    USE_MIRROR=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)        DATASET_TYPE="$2"; shift 2 ;;
        --mirror|--use-mirror) USE_MIRROR=true; shift ;;
        --no-mirror)      USE_MIRROR=false; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dataset TYPE    Download type: sft | rl | eval | all (default: all)"
            echo "  --mirror          Use hf-mirror.com (recommended inside China)"
            echo "  --no-mirror       Use huggingface.co (default)"
            echo "  -h, --help        Show this help"
            echo ""
            echo "Environment variables:"
            echo "  HF_ENDPOINT       Set to a URL containing 'hf-mirror.com' to auto-enable mirror"
            echo ""
            echo "Examples:"
            echo "  $0                           # Download everything"
            echo "  $0 --dataset eval --mirror   # Evaluation data via mirror"
            echo "  HF_ENDPOINT=https://hf-mirror.com $0"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: unknown argument $1${NC}"
            exit 1
            ;;
    esac
done

if [ "$USE_MIRROR" = true ]; then
    HF_BASE_URL="https://hf-mirror.com"
    SOURCE_NAME="hf-mirror.com"
else
    HF_BASE_URL="https://huggingface.co"
    SOURCE_NAME="huggingface.co"
fi

echo -e "${GREEN}=== E2C Dataset Download ===${NC}"
echo "Project root:  ${PROJECT_ROOT}"
echo "Data dir:      ${DATA_DIR}"
echo "Dataset type:  ${DATASET_TYPE}"
echo -e "Source:        ${BLUE}${SOURCE_NAME}${NC}"
echo ""

mkdir -p "${RAW_DIR}/sft"
mkdir -p "${RAW_DIR}/rl"
mkdir -p "${EVAL_DIR}"

download_sft_data() {
    echo -e "${YELLOW}[SFT] Downloading SFT training data...${NC}"

    local SFT_URL="${HF_BASE_URL}/datasets/TingheOliver/Explore-Execute-Chain-Datasets/resolve/main/e2c-sft.parquet"
    local SFT_FILE="${RAW_DIR}/sft/e2c-sft.parquet"

    if [ -f "${SFT_FILE}" ]; then
        echo -e "${YELLOW}File already exists: ${SFT_FILE}${NC}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && echo "Skipped." && return 0
    fi

    echo "Downloading to ${SFT_FILE}"
    wget -O "${SFT_FILE}" "${SFT_URL}" || {
        echo -e "${RED}wget failed, trying curl...${NC}"
        curl -L -o "${SFT_FILE}" "${SFT_URL}" || {
            echo -e "${RED}Download failed. Check network connection.${NC}"
            return 1
        }
    }
    echo -e "${GREEN}SFT data downloaded (77.7 MB)${NC}"
}

download_rl_data() {
    echo -e "${YELLOW}[RL] Downloading RL training data...${NC}"

    # Files on HuggingFace: ef-rl.parquet, ef-rl-valid.parquet
    # Saved locally as:     e2c-rl.parquet, e2c-rl-valid.parquet
    local RL_TRAIN_URL="${HF_BASE_URL}/datasets/TingheOliver/Explore-Execute-Chain-Datasets/resolve/main/ef-rl.parquet"
    local RL_VALID_URL="${HF_BASE_URL}/datasets/TingheOliver/Explore-Execute-Chain-Datasets/resolve/main/ef-rl-valid.parquet"
    local RL_TRAIN_FILE="${RAW_DIR}/rl/e2c-rl.parquet"
    local RL_VALID_FILE="${RAW_DIR}/rl/e2c-rl-valid.parquet"

    if [ -f "${RL_TRAIN_FILE}" ]; then
        echo "  Train data already exists, skipping."
    else
        echo "Downloading RL train data..."
        wget -O "${RL_TRAIN_FILE}" "${RL_TRAIN_URL}" || {
            curl -L -o "${RL_TRAIN_FILE}" "${RL_TRAIN_URL}" || { echo -e "${RED}RL train download failed.${NC}"; return 1; }
        }
        echo -e "${GREEN}RL train data downloaded (19.4 MB)${NC}"
    fi

    if [ -f "${RL_VALID_FILE}" ]; then
        echo "  Validation data already exists, skipping."
    else
        echo "Downloading RL validation data..."
        wget -O "${RL_VALID_FILE}" "${RL_VALID_URL}" || {
            curl -L -o "${RL_VALID_FILE}" "${RL_VALID_URL}" || { echo -e "${RED}RL validation download failed.${NC}"; return 1; }
        }
        echo -e "${GREEN}RL validation data downloaded (706 KB)${NC}"
    fi

    echo -e "${GREEN}RL data ready (saved as e2c-rl*)${NC}"
}

download_eval_data() {
    echo -e "${YELLOW}[EVAL] Downloading evaluation benchmarks...${NC}"

    local EVAL_DATASETS=(
        # Math
        "aime24" "aime25" "amc23" "gsm8k"
        "math-algebra" "math500" "minerva" "olympiad_bench"
        # Medical
        "anatomy" "clinical_knowledge" "college_biology" "college_medicine"
        "medical_genetics" "medmcqa" "medqa" "professional_medicine"
    )

    local base_url="${HF_BASE_URL}/datasets/TingheOliver/Explore-Execute-Chain-Datasets/resolve/main/evaluation"

    for dataset_name in "${EVAL_DATASETS[@]}"; do
        local output_file="${EVAL_DIR}/${dataset_name}.parquet"
        if [ -f "${output_file}" ]; then
            echo "  - ${dataset_name}: already exists, skipping"
            continue
        fi
        echo "  - Downloading ${dataset_name}..."
        wget -q -O "${output_file}" "${base_url}/${dataset_name}.parquet" || {
            curl -s -L -o "${output_file}" "${base_url}/${dataset_name}.parquet" || {
                echo -e "${RED}    Failed: ${dataset_name}${NC}"
                continue
            }
        }
        echo -e "${GREEN}    ${dataset_name} done${NC}"
    done

    echo -e "${GREEN}Evaluation data downloaded${NC}"
}

case "${DATASET_TYPE}" in
    sft)  download_sft_data ;;
    rl)   download_rl_data ;;
    eval) download_eval_data ;;
    all)
        download_sft_data
        download_rl_data
        download_eval_data
        ;;
    *)
        echo -e "${RED}Error: unknown dataset type '${DATASET_TYPE}'${NC}"
        echo "Supported: sft, rl, eval, all"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== Download complete ===${NC}"
echo "Data locations:"
echo "  - SFT raw:    ${RAW_DIR}/sft/"
echo "  - RL raw:     ${RAW_DIR}/rl/"
echo "  - Evaluation: ${EVAL_DIR}/"
echo ""
echo "Next step: bash scripts/prepare_all_data.sh"
