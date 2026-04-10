# LiMem Knights & Knaves Evaluation

This project provides a miniaturized version of the demo and evaluations suite LiMem scoring, using Knights and Knaves logic puzzles. It is run via batch prompting API-based LLMs, and aims to follow the main aspects of the designs outlined in the paper "On Memorization of Large Language Models in Logical Reasoning" by Chulin Xie, Yangsibo Huang, Chiyuan  Zhang, Da Yu, Xinyun Chen, Bill Yuchen Lin, Bo Li, Badih Ghazi, and Ravi Kumar. 

## Features
- Generates pairs of logic puzzles (original and paraphrased) with shared logic.
- Supports increasing puzzle complexity (number of people in the puzzle).
- Evaluates models for accuracy, consistency, and LiMem score.
- Can run with mock models or real LLMs via OpenAI-compatible API.
- Results can be saved to CSV for later analysis or plotting.

## Requirements
- Python 3.8+
- `requests` (for API mode)
- `puzzle_gen.py` in the same directory (must support `max_people` argument in `generate_n_pairs`)

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run a quick demo (mock model)
```
python limem_demo.py demo
```

### 3. Run a sweep with mock models
```
python limem_demo.py mock --repeats 5 --max-people 2 --save results_mock.csv
```

### 4. Run with API models (requires API key)
- Place your API key in a `.env` file as `GENAI_RCAC_API_KEY=your_key_here`.
- Example:
```
python limem_demo.py --models llama4:latest,gpt-oss:120b --repeats 1 --max-people 2 --save results_api.csv
```

### 5. Increase puzzle complexity
- Use `--max-people N` to generate puzzles with N people (if supported by `puzzle_gen.py`).

### 6. Save results
- Use `--save path.csv` to save per-run or sweep results as CSV for later plotting or analysis.

## Arguments
- `demo`: Run a single quick mock demo.
- `mock`: Run a sweep with mock models.
- (default): Run with API models (needs API key).
- `--models`: Comma-separated list of model names/IDs.
- `--repeats`: Number of runs per model.
- `--seed`: Base random seed.
- `--max-people`: Maximum number of people in a puzzle (complexity).
- `--save`: Path to save results as CSV.

## Example .env file
```
GENAI_RCAC_API_KEY=your_api_key_here
```

## Notes
- Increasing `--max-people` increases puzzle complexity, but requires `puzzle_gen.py` to support the `max_people` argument.
- Results are saved as CSV for easy plotting or further analysis.
- For API mode, ensure you do not exceed rate limits (20 requests/min/model).

## TODO
- Add model confidence display across complexity (future work).
