#!/bin/bash
# Run E2C Test-Time Scaling on AIME 2024

set -e
cd "$(dirname "$0")"

echo "E2C Test-Time Scaling Experiment"
echo "================================"

# Check data exists
if [ ! -f "data/aime2024/aime2024.parquet" ]; then
    echo "Error: data/aime2024/aime2024.parquet not found."
    echo "Please download AIME 2024 data first. See data/aime2024/README.md"
    exit 1
fi

python run_tts.py "$@"
