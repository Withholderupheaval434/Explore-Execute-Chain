#!/usr/bin/env bash
# Download the released E2C model from HuggingFace.
# Usage: bash scripts/download_model.sh [--mirror]

set -eo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL_NAME="TingheOliver/Explore-Execute-Chain-Qwen"
USE_MIRROR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL_NAME="$2"; shift 2 ;;
        --mirror|--use-mirror) USE_MIRROR=true; shift ;;
        --no-mirror)    USE_MIRROR=false; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model MODEL   HuggingFace model ID (default: TingheOliver/Explore-Execute-Chain-Qwen)"
            echo "  --mirror        Use hf-mirror.com (recommended inside China)"
            echo "  --no-mirror     Use huggingface.co (default)"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                 # Download the released model"
            echo "  $0 --mirror        # Download via mirror"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: unknown argument $1${NC}"
            exit 1
            ;;
    esac
done

if [ "$USE_MIRROR" = true ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    SOURCE_NAME="hf-mirror.com"
else
    export HF_ENDPOINT=https://huggingface.co
    SOURCE_NAME="huggingface.co"
fi

echo -e "${GREEN}=== E2C Model Download ===${NC}"
echo "Model:  ${MODEL_NAME}"
echo -e "Source: ${BLUE}${SOURCE_NAME}${NC}"
echo ""

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

python3 -c "import transformers" 2>/dev/null || {
    echo -e "${YELLOW}transformers not installed. Installing...${NC}"
    pip install transformers torch -q
}

echo "Downloading model (this may take a while)..."
echo ""

python3 << EOF
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "${MODEL_NAME}"

if os.environ.get("HF_ENDPOINT", "").find("hf-mirror") >= 0:
    print(f"Using mirror: {os.environ['HF_ENDPOINT']}")

print(f"Model: {model_name}")
print("")

try:
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"  Tokenizer ready (vocab size: {len(tokenizer)})")

    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model ready")
    print("")
    print("Download complete.")
    print(f"Cached at: {model.config.name_or_path}")

except Exception as e:
    print(f"Download failed: {e}")
    raise SystemExit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Done ===${NC}"
    echo "Use in code:"
    echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
    echo "  model = AutoModelForCausalLM.from_pretrained('${MODEL_NAME}')"
else
    echo -e "${RED}Download failed.${NC}"
    exit 1
fi
