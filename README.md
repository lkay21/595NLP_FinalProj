## 595NLP Final Project - RLP Reimplementation Starter

This repository is a **two-day scoped starter** for reimplementing the paper:
`Fine-Tuning Language Models with Reward Learning on Policy (RLP)`.

The design intentionally prioritizes:
- fast iteration over full-scale fidelity,
- small-model + LoRA readiness,
- an SPG-first path (usually stronger than UML in the paper),
- clear stage boundaries so you can finish an end-to-end demo quickly.

## Two-Day Execution Plan (Recommended)

### Day 1
1. Prepare/convert data into JSONL.
2. Run tiny SFT warm-up.
3. Train reward model on pairwise preferences.
4. Run short PPO baseline.

### Day 2
1. Sample policy outputs (`n=6` by default for speed).
2. Generate synthetic preferences with confidence filtering.
3. Retrain reward model with human + synthetic preferences.
4. Retrain policy (short PPO) and run final evaluation script.

## Quick Start

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Place data files in `data/raw`:
   - `sft_train.jsonl`
   - `pref_train.jsonl`
   - `unlabeled_train.jsonl`
4. Copy and adjust configs in `configs/`.
5. Run stage scripts in `scripts/` in order.

## Data Format

- `sft_train.jsonl`
  - `{"prompt": "...", "response": "..."}`
- `pref_train.jsonl`
  - `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- `unlabeled_train.jsonl`
  - `{"prompt": "..."}`

## Important Scope Notes

- This starter is intentionally lightweight and safe for a 2-day project timeline.
- Default configs favor feasibility on limited compute.
- The UML and SPG modules are implemented as practical approximations intended for coursework reproduction, not exact industrial parity.

## LiMem Demo (Knights and Knaves)

This repo also includes a small, standalone script:

- `limem_demo.py`

The script demonstrates a minimal version of the LiMem-style memorization check from the paper *On Memorization of Large Language Models in Logical Reasoning* using Knights and Knaves puzzles.

### What the script does

1. Defines 5 puzzle pairs:
   - an **original** puzzle
   - a **perturbed** version with only surface-level wording/name changes
2. Uses a `MockLLM` class to produce answers.
3. Evaluates per-pair correctness for original and perturbed versions.
4. Computes LiMem-related metrics:
   - **Accuracy (Acc)** = `# Correct Original / Total`
   - **Consistency Ratio (CR)** = `# Consistently Correct / # Correct Original`
   - **LiMem Score** = `Acc * (1 - CR)`

### Why this is useful

If a model performs well on original items but drops on logically equivalent perturbations, that indicates potential memorization of surface forms rather than robust reasoning. The LiMem score increases when this gap is larger.

### How to run

From the project root:

- `python limem_demo.py`

Example output includes:

- per-pair correctness (`original_correct`, `perturbed_correct`)
- aggregate metrics (`Acc`, `CR`, `LiMem Score`)

### Sweeping different “models”

`limem_demo.py` supports a lightweight sweep mode so you can compare multiple model configs side-by-side (and average over repeated runs if the model is stochastic).

- **Run the original single demo**:
  - `python limem_demo.py --mode single`

- **Run a sweep (defaults)**:
  - `python limem_demo.py --mode sweep`

- **Choose models + repeats**:
  - `python limem_demo.py --mode sweep --models mock_baseline,mock_robust --repeats 10`

#### Built-in model config names

These are *mock* model configs (no external API calls yet):

- `mock_baseline`: deterministic baseline
- `mock_robust`: more consistent on perturbed variants (lower LiMem)
- `mock_noisy_t02`: introduces random errors (use `--repeats > 1`)
- `mock_noisy_t04`: more random errors (use `--repeats > 1`)

### Does this work with a real API?

**Not yet** — the current script uses mock “models” implemented in Python, so `--models ...` does *not* call any external LLM API.

To use real models via an API, you’d add a new API-backed model class (e.g., `OpenAIAPILLM`) that implements the same interface as `MockLLM`:

- a `solve(puzzle: str, is_original: bool) -> str` method that:
  - sends the puzzle to your LLM endpoint
  - parses the response into the same answer format used by the evaluator (e.g., `A=Knave, B=Knight`)

Then you’d register it in `build_model(...)` with a new config name (e.g., `openai_gpt4o_mini_t0`), and include it in `--models`.
