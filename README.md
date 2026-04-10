pip install -r requirements.txt

# LiMem Knights & Knaves Evaluation

This project evaluates large language models (LLMs) on logical reasoning using Knights and Knaves puzzles, following the LiMem scoring approach. It supports both mock and real LLMs (via OpenAI-compatible API), measures accuracy, consistency, and LiMem score, and saves results as CSV for analysis.

## Quick Start

1. **Install dependencies:**
	```
	pip install -r requirements.txt
	```

2. **Run a quick demo (mock model):**
	```
	python limem_demo.py demo
	```

3. **Run with API models:**
	- Add your API key to a `.env` file as `GENAI_RCAC_API_KEY=your_key_here`
	- Example:
	  ```
	  python limem_demo.py --models llama4:latest,gpt-oss:120b --repeats 1 --max-people 2 --save results.csv
	  ```

4. **Increase puzzle complexity:**
	- Use `--max-people N` (e.g., `--max-people 4`)

5. **Save results:**
	- Use `--save results.csv`

## Main Arguments
- `demo` — Run a quick mock demo
- `mock` — Run a sweep with mock models
- (default) — Run with API models (needs API key)
- `--models` — Comma-separated model names/IDs
- `--repeats` — Number of runs per model
- `--max-people` — Puzzle complexity
- `--save` — Path to save results as CSV

## Example .env file
```
GENAI_RCAC_API_KEY=your_api_key_here
```

## Notes
- Results are saved as CSV for easy analysis.
- For API mode, do not exceed rate limits (20 requests/min/model).

## Notes
- Increasing `--max-people` increases puzzle complexity, but requires `puzzle_gen.py` to support the `max_people` argument.
- Results are saved as CSV for easy plotting or further analysis.
- For API mode, ensure you do not exceed rate limits (20 requests/min/model).

## TODO
- Add model confidence display across complexity (future work).
