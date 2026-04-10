"""
Minimal LiMem score demo for Knights and Knaves puzzles.

Generates 200 puzzle pairs per run (2 blocks of 100). Each pair shares one logic;
the paraphrase is a small leaf-level surface edit (synonym / function word), not a full re-lettering.

API mode uses 4 batched prompts per model (100 originals + 100 paraphrases per block).
"""

import argparse
import os
import random
import re
import time
import math
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import requests

from puzzle_gen import generate_n_pairs

# OVERALL TODOs
# - DONE: Update Markdown for understanding of project and how to run
# - DONE: Finalize game generation, ability to increase complexity, and ability to save results to a file for future displaying
# - TODO: Implement ability to display MODEL CONFIDENCE across complexity to be compared with limem score


@dataclass
class PuzzlePair:
    pair_id: int 
    original: str
    perturbed: str
    gold: str #correct answer


GAMES_PER_PROMPT = 20
N_PROMPT_BATCHES = 4  # block0 orig, block0 pert, block1 orig, block1 pert
DEMO_PAIR_COUNT = 8  # first N pairs for `demo` output readability


with open("prompts.json", "r") as f:
    prompts = json.load(f)

PROMPT_ENGINEERING_TECH_LOOKUP = [
    "zero-shot",
    "few-shot",
    "chain-of-thought",
    "self-consistency",
    "tree-of-thought",
    "persona",
    "self-ask"
]




# Complexity control: set MAX_PEOPLE_PUZZLE to control number of people in puzzles (default 2, can be increased for more complex puzzles).
MAX_PEOPLE_PUZZLE = 2  # You can increase this for more complex puzzles (requires puzzle_gen.py support)


def _invert_gold(gold: str) -> str:
    m = re.fullmatch(r"A=(Knight|Knave),\s*B=(Knight|Knave)", gold)
    if not m:
        return "A=Knight, B=Knight"

    def inv(s: str) -> str:
        return "Knave" if s == "Knight" else "Knight"

    return f"A={inv(m.group(1))}, B={inv(m.group(2))}"


# function to calculate the limem score
# how accurate the model is on the original 
# cr is how often the perturbed is correct when the original is correct (consistency ratio)
# limem = acc * (1 - cr) 
# a model with a higher limem score is more likely to have been memorizing
# a model with a low limem score but high accuracy is likely to have been solving by reasoning
# a model with low accuracy and low limem is likely to be guessing or not understanding the task
def calculate_limem(results: List[Dict[str, bool]]) -> Dict[str, float]:
    total = len(results)
    correct_original = sum(1 for r in results if r["original_correct"])
    consistently_correct = sum(
        1
        for r in results
        if r["original_correct"] and r["perturbed_correct"]
    )

    acc = correct_original / total if total else 0.0
    cr = consistently_correct / correct_original if correct_original else 0.0
    limem = acc * (1 - cr)

    return {
        "total": float(total),
        "correct_original": float(correct_original),
        "consistently_correct": float(consistently_correct),
        "accuracy": acc,
        "consistency_ratio": cr,
        "limem_score": limem,
    }


# utilizes puzzle_gen.py to create puzzle pairs for evaluation
def get_puzzle_pairs(seed: int, max_people: int = MAX_PEOPLE_PUZZLE) -> List[PuzzlePair]:
    rng = random.Random(seed)

    # og and peturbed, n is equal to the number of games needed to be generated 
    n = GAMES_PER_PROMPT * 2
    gen = generate_n_pairs(n, rng, max_people=max_people)  
    return [
        PuzzlePair(
            pair_id=i,
            original=g.original_text,
            perturbed=g.perturbed_text,
            gold=g.gold,
        )
        for i, g in enumerate(gen, start=1)
    ]


# overviews the task at hand and shows puzzle sample given to models
def print_benchmark_overview(pairs: List[PuzzlePair]) -> None:
    """Summary + sample games (listing all gold texts would be hundreds of lines)."""
    n = len(pairs)
    print()
    print("-" * 72)
    print(f"BENCHMARK: {n} puzzle pairs (same logic per pair; second text is a small surface tweak).")
    print(
        f"API calls per model run: {N_PROMPT_BATCHES} batched prompts "
        f"({GAMES_PER_PROMPT} puzzles each), not {2 * n} separate calls."
    )
    print("Puzzles are generated in puzzle_gen.py (AST + brute-force solve).")
    print("Rules: Knight = truthful, Knave = lies. Graded answer format: A=Knight|Knave, B=Knight|Knave")
    print("-" * 72)
    sample = pairs[: min(2, n)]
    for p in sample:
        print()
        print(f"  Example pair_id={p.pair_id}  gold={p.gold}")
        print(f"    Original:   {_puzzle_preview(p.original, 200)}")
        print(f"    Paraphrase: {_puzzle_preview(p.perturbed, 200)}")
    print()
    rest = max(0, n - len(sample))
    if rest:
        print(f"  ... ({rest} more pairs omitted from this summary)")
    print("-" * 72)
    print()


# Purdue GenAI chat API (OpenAI-compatible). Key: GENAI_RCAC_API_KEY in .env or environment.
DEFAULT_GENAI_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"
DEFAULT_API_KEY_ENV = "GENAI_RCAC_API_KEY"

# Models for API sweep (20 req/min per model enforced below).
API_MODEL_IDS = [
    "llama4:latest",
    "gpt-oss:120b"
]

API_RATE_LIMIT_PER_MINUTE = 20

MOCK_SWEEP_MODELS = [
    "mock_baseline",
    "mock_robust",
    "mock_noisy_t02",
    "mock_noisy_t04",
]


def _load_env_file(path: Path, *, announce: bool = False) -> None:
    """Parse KEY=VALUE lines; do not override variables already set in the process env."""
    if not path.is_file():
        return
    if announce:
        print(f"[limem] .env {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def load_dotenv_files() -> None:
    """Load `.env` from this script's directory, then the current working directory."""
    seen: set[Path] = set()
    for path in (
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_env_file(resolved, announce=True)


def build_batch_user_prompt(num_puzzles: int, numbered_lines: str) -> str:
    return (
        f"There are {num_puzzles} numbered puzzles below. "
        f"Output exactly {num_puzzles} lines, format as specified in the system message.\n\n"
        + numbered_lines.strip()
    )


def _puzzle_preview(text: str, max_len: int = 120) -> str:
    """Single-line snippet for logs (not the full prompt)."""
    one = " ".join(text.split())
    if len(one) <= max_len:
        return one
    return one[: max_len - 3] + "..."


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_model_answer(raw: str) -> Optional[str]:
    """Single-line A=/B= answer (kept for small tests)."""
    if not raw:
        return None
    text = _strip_markdown_fences(raw)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    pattern = re.compile(
        r"^([A-Za-z]+)\s*=\s*(Knight|Knave)\s*,\s*([A-Za-z]+)\s*=\s*(Knight|Knave)\s*$",
        re.IGNORECASE,
    )
    for line in reversed(lines):
        m = pattern.match(line)
        if not m:
            continue
        a, ra, b, rb = m.groups()
        return f"{a}={ra.capitalize()}, {b}={rb.capitalize()}"
    return None


# parses repsonses into answers, if an unknown is encountered, it is counted as incorrect
def parse_batch_answers(raw: str, n: int) -> List[str]:
    out: List[str] = ["UNKNOWN"] * n
    if not raw:
        return out
    text = _strip_markdown_fences(raw)
    line_pat = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.IGNORECASE)
    assign_pat = re.compile(r"([A-Z])\s*=\s*(Knight|Knave)", re.IGNORECASE)
    for line in text.splitlines():
        line_m = line_pat.match(line.strip())
        if not line_m:
            continue
        k = int(line_m.group(1))
        rest = line_m.group(2)
        assignments = assign_pat.findall(rest)
        if not assignments:
            continue
        if 1 <= k <= n:
            answer = ", ".join(f"{letter.upper()}={role.capitalize()}" for letter, role in assignments)
            out[k - 1] = answer
    return out


# Purdue GENAI limits reuests to 20 per minute. This requires rate limiting to avoid hitting errors during program run
class PerModelRateLimiter:
    """Sliding window: at most `max_calls` per `window_sec` per model name."""

    def __init__(self, max_calls: int, window_sec: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_sec = window_sec
        self._timestamps: Dict[str, Deque[float]] = defaultdict(deque)

    def acquire(self, model: str) -> None:
        now = time.monotonic()
        q = self._timestamps[model]
        cutoff = now - self.window_sec
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) < self.max_calls:
            q.append(now)
            return
        sleep_for = self.window_sec - (now - q[0]) + 0.05
        if sleep_for > 0:
            print(f"[limem] wait {sleep_for:.0f}s ({model}, {self.max_calls}/min)")
            time.sleep(sleep_for)
        self.acquire(model)


def chat_completion_text(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    rate_limiter: PerModelRateLimiter,
    timeout_sec: float = 180.0,
) -> str:
    rate_limiter.acquire(model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.0,
        "logprobs":True
    }

    for attempt in range(3):
        response = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
        if response.status_code != 200:
            print(f"[limem] ERROR: API request failed for model '{model}' (attempt {attempt + 1}/3): ")
            time.sleep(5)
            continue
        try:
            data = response.json()
        except Exception:
            print(f"[limem] ERROR: API response is not valid JSON for model '{model}': {response.text}")
            time.sleep(5)
            continue
        if data is None or data == "null":
            print(f"[limem] ERROR: API returned null/None for model '{model}'. Full response: {response.text}")
            time.sleep(5)
            continue
        break
    else:
        print(f"[limem] ERROR: API request failed after 3 attempts for model '{model}'. Last response: {response.text}")
        return "", None
    choices = data.get("choices") or []
    if not choices:
        print(f"[limem] ERROR: API response missing choices for model '{model}': {data!r}")
        raise RuntimeError(f"API response missing choices: {data!r}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        print(f"[limem] ERROR: API response missing message.content for model '{model}': {data!r}")
        raise RuntimeError(f"API response missing message.content: {data!r}")
    if not isinstance(content, str):
        content = str(content)

    logprobs_data = choices[0].get("logprobs")

    return content, logprobs_data


def evaluate_model_api(
    model_name: str,
    pairs: List[PuzzlePair],
    *,
    api_key: str,
    url: str = DEFAULT_GENAI_API_URL,
    rate_limiter: Optional[PerModelRateLimiter] = None,
    prompt_type: Optional[str] = None,
) -> List[Dict[str, bool]]:
    """
    Four batched chat calls: two blocks of 100 puzzles, each block with originals then paraphrases.
    """

    batch_conf = []

    if rate_limiter is None:
        rate_limiter = PerModelRateLimiter(API_RATE_LIMIT_PER_MINUTE, window_sec=60.0)
    expected = GAMES_PER_PROMPT * (N_PROMPT_BATCHES // 2)
    if len(pairs) != expected:
        raise ValueError(f"Need {expected} puzzle pairs, got {len(pairs)}")

    blocks = [
        pairs[0:GAMES_PER_PROMPT],
        pairs[GAMES_PER_PROMPT : 2 * GAMES_PER_PROMPT],
    ]
    batch_preds: List[List[str]] = []

    for b_idx, block in enumerate(blocks):
        for use_original in (True, False):
            label = f"b{b_idx + 1}_{'orig' if use_original else 'pert'}"
            numbered = "\n".join(
                f"{j}. {block[j - 1].original if use_original else block[j - 1].perturbed}"
                for j in range(1, GAMES_PER_PROMPT + 1)
            )
            user_msg = build_batch_user_prompt(GAMES_PER_PROMPT, numbered)

            prompt = prompts[prompt_type]
            # print(f"PROMPT IS:\n{prompt}\n")

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ]
            raw, logprobs_data = chat_completion_text(
                url=url,
                api_key=api_key,
                model=model_name,
                messages=messages,
                rate_limiter=rate_limiter,
                timeout_sec=600.0,
            )
            print(f"[DEBUG RAW RESPONSE]\n{raw[:500]}\n[END DEBUG]") 
            preds = parse_batch_answers(raw, GAMES_PER_PROMPT)
            ok = sum(1 for x in preds if x != "UNKNOWN")
            print(f"[limem] pi {model_name}  {label}  parsed {ok}/{GAMES_PER_PROMPT}")
            batch_preds.append(preds)

            avg_confidence = None
            if logprobs_data and "content" in logprobs_data:
                token_logprobs = [t["logprob"] for t in logprobs_data["content"] if "logprob" in t]
                if token_logprobs:
                    avg_confidence = sum(math.exp(lp) for lp in token_logprobs) / len(token_logprobs)

            if avg_confidence is not None:
                # print(f"[limem] pi {model_name}  {label}  avg token confidence: {avg_confidence:.4f}")
                batch_conf.append(avg_confidence)

    p0_o, p0_p, p1_o, p1_p = batch_preds
    results: List[Dict[str, bool]] = []

    for i in range(GAMES_PER_PROMPT):
        pair = pairs[i]
        results.append(
            {
                "pair_id": pair.pair_id,
                "original_correct": p0_o[i] == pair.gold,
                "perturbed_correct": p0_p[i] == pair.gold,
            }
        )
    for i in range(GAMES_PER_PROMPT):
        pair = pairs[GAMES_PER_PROMPT + i]
        results.append(
            {
                "pair_id": pair.pair_id,
                "original_correct": p1_o[i] == pair.gold,
                "perturbed_correct": p1_p[i] == pair.gold,
            }
        )

    overall_conf = sum(batch_conf) / len(batch_conf) if batch_conf else None

    return results, overall_conf


def evaluate_model(model: object, pairs: List[PuzzlePair]) -> List[Dict[str, bool]]:
    results: List[Dict[str, bool]] = []
    for pair in pairs:
        pred_o = model.solve(pair.original, pair, True)
        pred_p = model.solve(pair.perturbed, pair, False)
        results.append(
            {
                "pair_id": pair.pair_id,
                "original_correct": pred_o == pair.gold,
                "perturbed_correct": pred_p == pair.gold,
            }
        )
    return results



def run_demo(seed: int, max_people: int = MAX_PEOPLE_PUZZLE, save_path: Optional[str] = None) -> None:
    print("[limem] demo (mock)")
    pairs = get_puzzle_pairs(seed, max_people=max_people)[:DEMO_PAIR_COUNT]
    model = MockLLM()
    results = evaluate_model(model, pairs)

    metrics = calculate_limem(results)

    # Save results to file if requested
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("pair_id,original_correct,perturbed_correct\n")
            for r in results:
                f.write(f"{r['pair_id']},{r['original_correct']},{r['perturbed_correct']}\n")
            f.write("\n")
            for k, v in metrics.items():
                f.write(f"{k},{v}\n")
        print(f"[limem] Results saved to {save_path}")

    print("Per-pair correctness:")
    for r in results:
        print(
            f"Pair {r['pair_id']}: "
            f"original_correct={r['original_correct']}, "
            f"perturbed_correct={r['perturbed_correct']}"
        )

    print("\nLiMem metrics:")
    print(f"Total pairs: {int(metrics['total'])}")
    print(f"Correct Original: {int(metrics['correct_original'])}")
    print(f"Consistently Correct: {int(metrics['consistently_correct'])}")
    print(f"Accuracy (Acc): {metrics['accuracy']:.3f}")
    print(f"Consistency Ratio (CR): {metrics['consistency_ratio']:.3f}")
    print(f"LiMem Score: {metrics['limem_score']:.3f}")


def build_model(name: str, seed: int | None = None) -> object:
    if name == "mock_baseline":
        return MockLLM()
    if name == "mock_robust":
        return BetterGeneralizationMockLLM()
    if name == "mock_noisy_t02":
        return NoisyMockLLM(error_rate=0.2, seed=seed)
    if name == "mock_noisy_t04":
        return NoisyMockLLM(error_rate=0.4, seed=seed)
    raise ValueError(f"Unknown model config: {name}")



def run_experiments(
    model_names: List[str],
    repeats: int,
    seed: int,
    *,
    use_api: bool = False,
    api_key: Optional[str] = None,
    api_url: str = DEFAULT_GENAI_API_URL,
    max_people: int = MAX_PEOPLE_PUZZLE,
    save_path: Optional[str] = None,
    prompt_type: Optional[str] = None
) -> None:
    rows: List[Dict[str, float | str]] = []
    mode = "api" if use_api else "mock"
    print(
        f"[limem] sweep {mode}  models={len(model_names)}  repeats={repeats}  base_seed={seed}  max_people={max_people}"
    )

    for model_name in model_names:
        acc_vals: List[float] = []
        cr_vals: List[float] = []
        limem_vals: List[float] = []
        latencies_ms: List[float] = []
        confidence_vals: List[float] = []

        if use_api:
            api_limiter = PerModelRateLimiter(API_RATE_LIMIT_PER_MINUTE, window_sec=60.0)

        print(f"[limem] == {model_name} ==")
        for run_idx in range(repeats):
            run_seed = seed + run_idx
            pairs = get_puzzle_pairs(run_seed, max_people=max_people)
            print(f"[limem] run {run_idx + 1}/{repeats}  puzzle_seed={run_seed}")

            start = time.perf_counter()
            if use_api:
                if not api_key:
                    raise ValueError("API mode requires an API key (set env or pass api_key).")
                results, avg_confidence = evaluate_model_api(
                    model_name,
                    pairs,
                    api_key=api_key,
                    url=api_url,
                    rate_limiter=api_limiter,
                    prompt_type=prompt_type
                )
            else:
                model = build_model(model_name, seed=run_seed)
                results = evaluate_model(model, pairs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics = calculate_limem(results)

            avg_confidence_str = f"{avg_confidence:.4f}" if avg_confidence is not None else "N/A"
            print(
                f"[limem]    Acc={metrics['accuracy']:.3f} CR={metrics['consistency_ratio']:.3f} "
                f"LiMem={metrics['limem_score']:.3f}  Conf = {avg_confidence_str}  Latency = {elapsed_ms:.0f}ms"
            )
            acc_vals.append(metrics["accuracy"])
            cr_vals.append(metrics["consistency_ratio"])
            limem_vals.append(metrics["limem_score"])
            if avg_confidence is not None:
                confidence_vals.append(avg_confidence)
            latencies_ms.append(elapsed_ms)

        rows.append(
            {
                "model": model_name,
                "acc": sum(acc_vals) / len(acc_vals),
                "cr": sum(cr_vals) / len(cr_vals),
                "limem": sum(limem_vals) / len(limem_vals),
                "confidence": sum(confidence_vals) / len(confidence_vals) if confidence_vals else None,
                "latency_ms": sum(latencies_ms) / len(latencies_ms),
            }
        )

    rows.sort(key=lambda r: float(r["limem"]), reverse=True)
    col_w = max(20, max(len(str(r["model"])) for r in rows) + 1) if rows else 24
    print("LiMem sweep results (averaged):")
    print(f"{'Model':<{col_w}} {'Acc':>7} {'CR':>7} {'LiMem':>9} {'Confidence':>12} {'Avg ms':>10}")
    print("-" * (col_w + 47))
    for row in rows:
        conf_str = f"{float(row['confidence']):12.4f}" if row['confidence'] is not None else "        N/A "
        print(
            f"{str(row['model']):<{col_w}} "
            f"{float(row['acc']):7.3f} "
            f"{float(row['cr']):7.3f} "
            f"{float(row['limem']):9.3f} "
            f"{conf_str} "
            f"{float(row['latency_ms']):10.3f}"
        )

    # Save results to file if requested
    if save_path:
        file_exists = Path(save_path).is_file()
        with open(save_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write(f"Model,PromptType,Acc,CR,LiMem,Confidence, Avg_ms\n")
            for row in rows:
                f.write(f"{row['model']},{prompt_type},{row['acc']},{row['cr']},{row['limem']},{row['confidence']},{row['latency_ms']}\n")
        print(f"[limem] Sweep results saved to {save_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LiMem Knights/Knaves evaluation. "
            "Default: API sweep (models in API_MODEL_IDS); use 'mock' or 'demo' for offline mocks."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("mock", "demo", "full"),
        default=None,
        help="mock: sweep mock LLMs; demo: one quick mock run. Omit: API sweep (needs API key in .env).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model ids (API) or mock names (with mock). Default lists apply when empty.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Runs per model (default: 1 for API sweep, 5 for mock sweep).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for noisy mock variants",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=MAX_PEOPLE_PUZZLE,
        help="Maximum number of people in a puzzle (complexity). Default: 2",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save results as CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv_files()
    args = parse_args()

    if args.command == "demo":
        print_benchmark_overview(get_puzzle_pairs(args.seed, max_people=args.max_people))
        print(f"[limem] start  demo  seed={args.seed}  max_people={args.max_people}")
        run_demo(args.seed, max_people=args.max_people, save_path=args.save)
    else:
        if args.command == "full":
            use_api = True
            model_list = list(API_MODEL_IDS)
            repeats = 1

            print_benchmark_overview(get_puzzle_pairs(args.seed, max_people=args.max_people))
            print(f"[limem] start  full demo eval  repeats={repeats}  seed={args.seed}  max_people={args.max_people}")

            api_key: Optional[str] = None
            api_key = os.environ.get(DEFAULT_API_KEY_ENV, "").strip()

            if not api_key:
                raise ValueError(
                    f"Set {DEFAULT_API_KEY_ENV} in .env or the environment (project folder .env is loaded automatically)."
                )
        else:
            use_api = args.command != "mock"
            raw_models = args.models.strip()
            if raw_models:
                model_list = [m.strip() for m in raw_models.split(",") if m.strip()]
            elif use_api:
                model_list = list(API_MODEL_IDS)
            else:
                model_list = list(MOCK_SWEEP_MODELS)

            if not model_list:
                raise ValueError("No models in list. Use --models a,b if overriding defaults.")

            repeats = args.repeats
            if repeats is None:
                repeats = 5 if args.command == "mock" else 1
            if repeats <= 0:
                raise ValueError("--repeats must be a positive integer")

            cmd = args.command or "api"
            print_benchmark_overview(get_puzzle_pairs(args.seed, max_people=args.max_people))
            print(f"[limem] start  cmd={cmd}  repeats={repeats}  seed={args.seed}  max_people={args.max_people}")

            api_key: Optional[str] = None
            if use_api:
                api_key = os.environ.get(DEFAULT_API_KEY_ENV, "").strip()
                if not api_key:
                    raise ValueError(
                        f"Set {DEFAULT_API_KEY_ENV} in .env or the environment (project folder .env is loaded automatically)."
                    )


        start_time = time.perf_counter()
        for prompt_tech in PROMPT_ENGINEERING_TECH_LOOKUP:
            print(f"\n=== Running with prompt engineering technique: {prompt_tech} ===\n")

            run_experiments(
                model_list,
                repeats=repeats,
                seed=args.seed,
                use_api=use_api,
                api_key=api_key,
                api_url=DEFAULT_GENAI_API_URL,
                max_people=args.max_people,
                save_path=args.save,
                prompt_type=prompt_tech
            )
        elapsed = time.perf_counter() - start_time
        print(f"\n[limem] All experiments completed in {elapsed:.2f} seconds.")
