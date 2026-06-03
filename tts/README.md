# Test-Time Scaling Experiments

This directory reproduces Table 3 of the E2C paper: applying E2C to various test-time compute strategies on AIME 2024 and comparing accuracy-vs-token-budget tradeoffs.

## Methods

| Method | Key | Description |
|--------|-----|-------------|
| Greedy CoT | `greedy_cot` | Single greedy decoding pass, no sampling |
| Self-Consistency | `self_consistency` | Sample N full CoT chains, majority-vote answer |
| E2C-SC | `e2c_sc` | Sample N E2C responses (exploration + execution), majority-vote |
| E2C-Select (LM judge) | `e2c_select_lm_judge` | Generate N explorations, pick best with LM judge, then execute |
| E2C-Select (semantic) | `e2c_select_semantic_cluster` | Generate N explorations, cluster semantically, pick centroid, execute |
| E2C-RP | `e2c_rp` | E2C with reward-based plan selection |
| E2C-ReAct Loop | `e2c_react_loop` | Iteratively revise the exploration plan until execution succeeds |
| E2C-ToT | `e2c_tot` | Tree search over exploration plans using semantic clustering |
| E2C-ToT (LM judge) | `e2c_tot_lm_judge` | Tree search over exploration plans using LM judge |
| E2C-ToT Layered | `e2c_tot_layered` | Multi-layer exploration tree, execute the best leaf once |
| Tree-of-Thoughts | `tree_of_thoughts` | Standard ToT over full reasoning chains |
| Forest-of-Thought | `forest_of_thought` | Forest-of-Thought over full reasoning chains |

E2C variants search over the short exploration phase (~1k tokens) rather than full chains (~10k tokens), yielding dramatically better accuracy-per-token efficiency.

## Setup

```bash
pip install -r requirements.txt
```

Download AIME 2024 problems:

```bash
python scripts/download_aime2024.py
```

## Configuration

All settings live in `config/tts.yaml`. Key fields:

```yaml
model:
  model_path: "TingheOliver/Explore-Execute-Chain-Qwen"
  subfolder: "Qwen3-8B-E2C-SFT-RL"   # Qwen3-4B-E2C-SFT-RL for 4B; empty for local paths
  device: "cuda"

tts:
  budgets: [1, 4, 8, 16, 32]    # K or N values to sweep
  temperature: 0.9
  max_explore_tokens: 512        # token budget for the exploration phase
  max_exec_tokens: 16384         # token budget for the execution phase
  max_full_tokens: 16384         # token budget for full-chain methods (SC, ToT, FoT)
  max_refine_rounds: 3           # e2c_react_loop: max exploration revision rounds
  max_refine_tokens: 1024        # e2c_react_loop: token budget per revision call
  tot_max_depth: 5               # e2c_tot_layered: max exploration tree depth

methods:
  - greedy_cot
  - self_consistency
  - e2c_sc
  - e2c_react_loop
  - tree_of_thoughts
  - forest_of_thought
  # Additional methods (require embedding backend):
  # - e2c_select_semantic_cluster
  # - e2c_tot

embedding:
  backend: "modelscope"                                       # or "huggingface"
  modelscope_model: "damo/nlp_gte_sentence-embedding_english-base"
  huggingface_model: "all-mpnet-base-v2"
```

> **Note on embedding**: `e2c_select_semantic_cluster` and `e2c_tot` require a sentence embedding model. The default uses ModelScope; set `backend: huggingface` if ModelScope is unavailable.

To use a local checkpoint, set `model_path` to the directory containing `config.json` and leave `subfolder` empty (or remove it).

## Running

```bash
# All methods in config/tts.yaml, all budgets
python run_tts.py

# Override methods and budgets from command line
python run_tts.py --methods e2c_react_loop e2c_sc self_consistency --budgets 4 8 16 32

# Quick debug run (first 5 problems only)
python run_tts.py --limit 5

# Use a different config file
python run_tts.py --config path/to/my_config.yaml
```

Results are written to `outputs/aime2024/tts_results.json` and plots to `outputs/`.

## Results (Qwen3-8B + E2C-RL, AIME 2024)

At K/N = 32:

| Method | Accuracy | Tokens (k) |
|--------|----------|------------|
| Self-Consistency | 50.0% | 86.2 |
| Tree-of-Thoughts | 50.0% | 71.3 |
| E2C-ReAct Loop | **53.3%** | **12.4** |

E2C-ReAct Loop achieves higher accuracy at 7x lower token cost. See `outputs/tradeoff_budget32.png` for the full accuracy-token tradeoff curve across all budgets.

## File overview

```
tts/
+-- run_tts.py              entry point; orchestrates all experiments
+-- tts_methods.py          all method implementations
+-- prompts.py              system and user prompt templates
+-- config/
|   +-- tts.yaml            experiment configuration
+-- scripts/
|   +-- download_aime2024.py
|   +-- plot_accuracy_token_tradeoff.py
|   +-- plot_budget32.py
+-- util/
|   +-- model.py            model loading
|   +-- dataset.py          AIME 2024 data loading
|   +-- embedding.py        sentence embedding (for semantic cluster methods)
|   +-- reward.py           answer matching and scoring
+-- data/                   AIME 2024 problems (populated after download)
+-- outputs/                results and plots
```
