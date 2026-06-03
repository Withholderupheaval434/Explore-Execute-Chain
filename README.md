# Explore-Execute Chain (E2C)

**Paper**: [Explore-Execute Chain: Towards an Efficient Structured Reasoning Paradigm](https://arxiv.org/abs/2509.23946)  
**Models**: [TingheOliver/Explore-Execute-Chain-Qwen](https://huggingface.co/TingheOliver/Explore-Execute-Chain-Qwen)  
**Datasets**: [TingheOliver/Explore-Execute-Chain-Datasets](https://huggingface.co/datasets/TingheOliver/Explore-Execute-Chain-Datasets)

---

E2C separates reasoning into two phases inside a single model:

1. **Exploration** (~1k tokens): sketch a high-level plan -- enumerate approaches, identify the most promising one.
2. **Execution** (~10k tokens): carry out the plan with full step-by-step reasoning.

Because test-time search targets only the short exploration phase, scaling compute is ~8x cheaper than searching over full reasoning chains. Because only exploration segments need domain-specific fine-tuning, adapting to a new domain (e.g., medical QA) uses ~3.5% of the tokens required by standard SFT.

## Results

### Mathematical reasoning

Qwen3-8B, Pass@1 averaged over 8 samples:

| Model | AIME'24 | AIME'25 | MATH500 | AMC23 | Avg |
|-------|---------|---------|---------|-------|-----|
| Qwen3-8B + GRPO | 36.9 | 34.4 | 88.2 | 79.3 | 59.6 |
| Qwen3-8B + E2C-(SFT+RL) | **40.6** | **33.8** | **87.7** | **80.3** | **61.5** |

### Test-time scaling

AIME 2024, K/N = 32:

| Method | Accuracy | Tokens (k) |
|--------|----------|------------|
| Self-Consistency | 50.0% | 86.2 |
| Tree-of-Thoughts | 50.0% | 71.3 |
| E2C-ReAct Loop | **53.3%** | **12.4** |

E2C-ReAct Loop matches or beats standard methods while using **7x fewer tokens**.

## Requirements

- Python 3.10+
- CUDA-capable GPU. Minimum VRAM:
  - Inference / evaluation: 16 GB (single GPU)
  - SFT training: 4x 40 GB GPUs recommended
  - RL training: 8x 40 GB GPUs recommended
- PyTorch 2.1+

## Setup

```bash
git clone https://github.com/OliverZ-dot/Explore-Execute-Chain-main.git
cd Explore-Execute-Chain-main
pip install -r verl/requirements.txt
```

## Quick start: inference

Run the released Qwen3-8B checkpoint on a single problem:

```bash
python example_inference.py \
    --model_path TingheOliver/Explore-Execute-Chain-Qwen \
    --subfolder  Qwen3-8B-E2C-SFT-RL \
    --problem    "Find all prime numbers p such that p^2 + 2 is also prime."
```

Use the 4B model instead:

```bash
python example_inference.py \
    --model_path TingheOliver/Explore-Execute-Chain-Qwen \
    --subfolder  Qwen3-4B-E2C-SFT-RL \
    --problem    "Your problem here"
```

All options:

```
python example_inference.py
    --model_path  TingheOliver/Explore-Execute-Chain-Qwen   # HF model ID or local path
    --subfolder   Qwen3-8B-E2C-SFT-RL                      # subfolder in HF repo (omit for local paths)
    --problem     "Your problem here"                        # omit to use built-in example
    --max_tokens  4096                                       # default: 2048
    --temperature 0.7                                        # default: 0.7
```

The model outputs two clearly delimited sections:

```
EXPLORATION PHASE:
  <high-level plan, ~1k tokens>

EXECUTION PHASE:
  <step-by-step solution, ~10k tokens>
```

### Interactive demo

Select from eight built-in problems (math, medical, code) or enter your own:

```bash
python example_interactive.py
```

> **Note**: `example_interactive.py` uses `TingheOliver/Explore-Execute-Chain-Qwen` (8B) by default. To switch models, edit `model_path_str` and `subfolder_str` at the top of `main()`. For a local path, set `subfolder_str = None`.

## Data

Download and preprocess all training and evaluation data in one step:

```bash
bash scripts/prepare_all_data.sh           # standard
bash scripts/prepare_all_data.sh --mirror  # use hf-mirror.com (recommended in China)
```

Partial downloads:

```bash
bash scripts/prepare_all_data.sh --skip-rl          # SFT data only
bash scripts/prepare_all_data.sh --skip-download    # reprocess already-downloaded files
bash scripts/download_datasets.sh --dataset eval    # 16 evaluation benchmarks only
```

**What gets downloaded** from `TingheOliver/Explore-Execute-Chain-Datasets`:

| Split | File | Size | Description |
|-------|------|------|-------------|
| SFT train | `e2c-sft.parquet` | 77.7 MB | 58k exploration-execution pairs |
| RL train | `e2c-rl.parquet` | 19.4 MB | 14k GRPO rollout prompts |
| RL valid | `e2c-rl-valid.parquet` | 706 KB | validation split |
| Eval (math) | 8 datasets | varies | AIME'24/25, AMC23, GSM8K, MATH500, Minerva, OlympiadBench, MATH-Algebra |
| Eval (medical) | 8 datasets | varies | MedQA, MedMCQA, Anatomy, Clinical Knowledge, College Biology, College Medicine, Medical Genetics, Professional Medicine |

Processed files land in `data/processed/`.

## Training

### Step 1 - E2C-SFT

Supervised fine-tuning on exploration-execution pairs.

```bash
bash scripts/e2c_sft.sh
```

Override defaults with environment variables:

```bash
export MODEL_PATH="Qwen/Qwen3-8B"
export TRAIN_DATA="data/processed/sft/e2c-sft-train.parquet"
export OUTPUT_DIR="models/checkpoints/sft"
bash scripts/e2c_sft.sh
```

Key env vars: `MODEL_PATH`, `TRAIN_DATA`, `VAL_DATA`, `OUTPUT_DIR`, `TOTAL_TRAINING_STEPS`, `CUDA_VISIBLE_DEVICES`. GPUs are auto-detected when `CUDA_VISIBLE_DEVICES` is unset.

Checkpoint: `models/checkpoints/sft/`

### Step 2 - E2C-RL

Two-stage GRPO. Stage 1 warms up with high diversity (rollout=32, temp=1.3, 1 epoch). Stage 2 sharpens execution with elevated advantage weight on exploration tokens (rollout=8, temp=1.0, adv_coeff=2.0, 2 epochs).

```bash
export MODEL_PATH="models/checkpoints/sft/final"
export N_GPUS=8
bash scripts/e2c_rl.sh            # both stages
bash scripts/e2c_rl.sh --stage 1  # warm-up only
bash scripts/e2c_rl.sh --stage 2  # main stage only
```

Checkpoints: `models/checkpoints/rl/stage1-warmup/`, `models/checkpoints/rl/stage2-main/`

### Step 3 (optional) - EF-SFT: domain adaptation

Fine-tunes only the exploration segments mixed with a small E2C regularization fraction. Uses ~3.5% of the tokens required by full SFT.

```bash
export MODEL_PATH="models/checkpoints/rl/stage2-main/final"
export TRAIN_DATA="data/processed/ef_sft/medical-train.parquet"
bash scripts/ef-sft.sh
```

Checkpoint: `models/checkpoints/ef_sft/`

### Alternative RL: DAPO

```bash
export MODEL_PATH="models/checkpoints/sft/final"
bash scripts/e2c_dapo.sh            # both stages
bash scripts/e2c_dapo.sh --stage 2  # main stage only
```

### Full workflow

```bash
bash scripts/prepare_all_data.sh
bash scripts/e2c_sft.sh
export MODEL_PATH="models/checkpoints/sft/final" && bash scripts/e2c_rl.sh
bash scripts/eval.sh --model models/checkpoints/rl/stage2-main/final --dataset all --sample 8
```

## Evaluation

```bash
bash scripts/eval.sh                                                         # GSM8K, released 8B model
bash scripts/eval.sh --dataset all --sample 8                                # all math benchmarks
bash scripts/eval.sh --dataset med --sample 4                                # all medical benchmarks
bash scripts/eval.sh --model path/to/ckpt --dataset aime24                  # local checkpoint (no subfolder needed)
bash scripts/eval.sh --subfolder Qwen3-4B-E2C-SFT-RL --dataset all          # use 4B model instead
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | `TingheOliver/Explore-Execute-Chain-Qwen` | HF ID or local path |
| `--subfolder NAME` | `Qwen3-8B-E2C-SFT-RL` | subfolder in HF repo; omit or leave empty for local paths |
| `--dataset NAME` | `gsm8k` | `gsm8k`, `math`, `aime24`, `aime25`, `amc23`, `all`, `medqa`, `medmcqa`, `med` |
| `--sample N` | `1` | samples per question (use 8 for reported results) |
| `--gpus N` | auto | number of GPUs |
| `--temp T` | `1.0` | sampling temperature |
| `--save-path PATH` | `evaluation/e2c-eval` | where to write predictions |

## Test-time scaling (AIME 2024)

Reproduces Table 3 of the paper. See [`tts/README.md`](tts/README.md) for full details.

```bash
cd tts
pip install -r requirements.txt
python scripts/download_aime2024.py

# Run all methods at all budgets
python run_tts.py

# Specific methods and budgets
python run_tts.py --methods e2c_react_loop e2c_tot --budgets 4 8 16 32

# Quick debug run (first 5 problems only)
python run_tts.py --limit 5
```

Set `model_path` (and `subfolder` if using the HF repo) in `tts/config/tts.yaml` before running. Results and plots are written to `tts/outputs/`.

**Available methods**: `greedy_cot`, `self_consistency`, `e2c_sc`, `e2c_rp`, `e2c_react_loop`, `e2c_tot`, `e2c_select_lm_judge`, `e2c_select_semantic_cluster`, `tree_of_thoughts`, `forest_of_thought` -- see `tts/config/tts.yaml` for the full list.

## Repository structure

```
.
+-- e2c/
|   +-- inference/          generation and evaluation scripts
|   +-- util/               reward, dataset, and model utilities
|   +-- config/             YAML configs for generation and evaluation
+-- scripts/
|   +-- prepare_all_data.sh
|   +-- download_datasets.sh
|   +-- e2c_sft.sh
|   +-- e2c_rl.sh
|   +-- e2c_dapo.sh
|   +-- ef-sft.sh
|   +-- eval.sh
|   +-- README.md           per-script documentation
+-- tts/                    test-time scaling experiments
|   +-- run_tts.py
|   +-- tts_methods.py
|   +-- config/tts.yaml
|   +-- README.md
+-- data/                   data preparation scripts
+-- verl/                   training framework (volcengine/verl, vendored)
+-- example_inference.py    single-problem inference demo
+-- example_interactive.py  interactive multi-problem demo
+-- example_problems.json   built-in example problems
```

## Citation

```bibtex
@misc{yang2025e2c,
  title={Explore-Execute Chain: Towards an Efficient Structured Reasoning Paradigm},
  author={Kaisen Yang and Tinghe Zhang and Rushi Shah and Kaicheng Yang and
          Qinwei Ma and Dianbo Liu and Alex Lamb},
  year={2025},
  eprint={2509.23946},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.23946}
}
```

## Acknowledgements

The RL training pipeline is built on [verl](https://github.com/volcengine/verl) (vendored with minor additions).

## License

MIT. See [verl/LICENSE](verl/LICENSE).
