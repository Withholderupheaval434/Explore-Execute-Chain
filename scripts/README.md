# Scripts

All training, evaluation, and data preparation scripts. Run from the project root.

## Data

### `prepare_all_data.sh`

Downloads and preprocesses all training and evaluation data in one go.

```bash
bash scripts/prepare_all_data.sh

# China mirror
bash scripts/prepare_all_data.sh --mirror

# Skip download, reprocess existing files
bash scripts/prepare_all_data.sh --skip-download

# SFT data only
bash scripts/prepare_all_data.sh --skip-rl
```

Options: `--skip-download`, `--skip-sft`, `--skip-rl`, `--num-workers N` (default 8).

### `download_datasets.sh`

Download raw datasets without preprocessing.

```bash
bash scripts/download_datasets.sh             # all
bash scripts/download_datasets.sh --dataset sft   # SFT only
bash scripts/download_datasets.sh --dataset rl    # RL only
bash scripts/download_datasets.sh --dataset eval  # 16 evaluation benchmarks
bash scripts/download_datasets.sh --mirror        # use hf-mirror.com
```

**Files downloaded:**
- SFT: `e2c-sft.parquet` (77.7 MB) → `data/raw/sft/`
- RL: `e2c-rl.parquet` (19.4 MB) + `e2c-rl-valid.parquet` (706 KB) → `data/raw/rl/`
- Eval (16 datasets) → `data/evaluation/`
  - Math: aime24, aime25, amc23, gsm8k, math-algebra, math500, minerva, olympiad_bench
  - Medical: anatomy, clinical_knowledge, college_biology, college_medicine, medical_genetics, medmcqa, medqa, professional_medicine

## Training

### `e2c_sft.sh`

E²C supervised fine-tuning. Trains the model on exploration-execution pairs.

```bash
bash scripts/e2c_sft.sh

# Override model or data
export MODEL_PATH="Qwen/Qwen3-8B"
export TRAIN_DATA="data/processed/sft/e2c-sft-train.parquet"
bash scripts/e2c_sft.sh
```

**Key env vars:** `MODEL_PATH`, `TRAIN_DATA`, `VAL_DATA`, `OUTPUT_DIR`, `TOTAL_TRAINING_STEPS`, `CUDA_VISIBLE_DEVICES`. GPUs are auto-detected if `CUDA_VISIBLE_DEVICES` is not set.

Checkpoint output: `models/checkpoints/sft/`

### `e2c_rl.sh`

Two-stage GRPO training. Stage 1 warms up with diverse rollouts (32 samples, temp=1.3). Stage 2 sharpens execution determinism with a higher advantage coefficient on exploration tokens (8 samples, temp=1.0, adv_coeff=2.0).

```bash
export MODEL_PATH="models/checkpoints/sft/final"
export N_GPUS=8
bash scripts/e2c_rl.sh           # both stages
bash scripts/e2c_rl.sh --stage 1 # warm-up only
bash scripts/e2c_rl.sh --stage 2 # main stage only
```

Checkpoint output: `models/checkpoints/rl/stage1-warmup/`, `models/checkpoints/rl/stage2-main/`

### `e2c_dapo.sh`

Alternative RL training using DAPO instead of GRPO. Same two-stage structure.

```bash
export MODEL_PATH="models/checkpoints/sft/final"
bash scripts/e2c_dapo.sh
bash scripts/e2c_dapo.sh --stage 2
```

### `ef-sft.sh`

Exploration-Focused SFT for domain adaptation. Fine-tunes only the exploration segments, mixed with a small fraction of base E²C data as regularization. Uses ~3.5% of the tokens required by standard SFT.

```bash
export MODEL_PATH="models/checkpoints/rl/stage2-main/final"
export TRAIN_DATA="data/processed/ef_sft/medical-train.parquet"
bash scripts/ef-sft.sh
```

## Evaluation

### `eval.sh`

Generate and evaluate model responses on math and medical benchmarks.

```bash
# Quick test on GSM8K (uses TingheOliver/Explore-Execute-Chain-Qwen by default)
bash scripts/eval.sh

# Specific dataset, 4 samples per question
bash scripts/eval.sh --dataset math --sample 4

# All math benchmarks, 8 samples per question
bash scripts/eval.sh --dataset all --sample 8

# Local checkpoint
bash scripts/eval.sh --model path/to/checkpoint --dataset gsm8k
```

Options: `--model PATH`, `--subfolder NAME`, `--dataset NAME`, `--sample N`, `--gpus N`, `--temp T`, `--save-path PATH`.

Available datasets: `gsm8k`, `math`, `aime24`, `aime25`, `amc23`, `all` (all math), `medqa`, `medmcqa`, `med` (all medical).

## Complete workflow

```bash
# 1. Prepare data
bash scripts/prepare_all_data.sh

# 2. SFT
bash scripts/e2c_sft.sh

# 3. RL
export MODEL_PATH="models/checkpoints/sft/final"
bash scripts/e2c_rl.sh

# 4. Evaluate
bash scripts/eval.sh --model models/checkpoints/rl/stage2-main/final --dataset all --sample 8

# 5. (Optional) Domain adaptation
export MODEL_PATH="models/checkpoints/rl/stage2-main/final"
bash scripts/ef-sft.sh
```

**Checkpoint locations:**
- SFT: `models/checkpoints/sft/`
- RL Stage 1: `models/checkpoints/rl/stage1-warmup/`
- RL Stage 2: `models/checkpoints/rl/stage2-main/`
- EF-SFT: `models/checkpoints/ef_sft/`
