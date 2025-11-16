## Evaluation process for Math Prof responses

This document describes how automated evaluation of initial and refined math responses is implemented in this repository. It explains the data inputs, preprocessing, LLM evaluation pipeline, prompt design, output schema, running instructions, and troubleshooting pointers.

### 1) Overview

- Purpose: compare `initial_response` and `refined_response` against a `golden_answer` and score them on multiple metrics (Correctness, Clarity, Simplicity, Overall). The pipeline writes per-row JSONL records and an aggregated summary plus a CSV export.
- Primary evaluation engine: Groq LLM `llama-3.3-70b-versatile` (configured in the scripts). The evaluation harness prefers Groq-only for determinism.

### 2) Data input

- Source: CSV files under `tests/` such as `EvalSet.csv` or `u_math_eval_subset (version 1).csv`.
- Required columns (case-insensitive): `uuid`, `golden_answer`, `initial_response`, `refined_response`.
- Encoding: the script attempts multiple encodings (utf-8, latin-1, iso-8859-1, cp1252) to robustly load files. If your CSV uses a different encoding, re-save as UTF-8 or specify an encoding when reading.

### 3) Preprocessing and column handling

- Column names are normalized to lower-case and stripped of whitespace. If a column is named `Unnamed: 5` or similar, rename it to a meaningful name (e.g., `question`) either in the CSV or with a small preprocessing step.
- Example quick rename (Python/pandas):

```python
import pandas as pd
df = pd.read_csv('tests/EvalSet.csv', encoding='latin-1')
if 'unnamed: 5' in df.columns.str.lower():
    df = df.rename(columns={col: 'question' for col in df.columns if col.lower() == 'unnamed: 5'})
```

### 4) Prompt template and response format

- The evaluator constructs a prompt containing:
  - A concise scoring rubric (Correctness, Clarity, Simplicity, Overall) and instructions to return a JSON object only.
  - The `golden_answer`, `initial_response`, and `refined_response` content.
- Expected LLM output: a single JSON object containing numeric scores (0–10) and short rationales. The script extracts the first JSON object found in the LLM response (it also strips common markdown fences like ```json ... ```).

### 5) LLM call and safety

- The Groq API key is expected in environment variable `GROQ_API_KEY`. The client is created at runtime with helpful error messaging if missing.
- Calls use a conservative temperature (e.g., 0.0–0.1) for deterministic scoring and include exponential backoff for transient rate-limit errors.

### 6) Batching, delays, and resumability

- CLI flags: `--batch-size` (rows per batch) and `--batch-delay` (seconds between batches). Use small batch sizes (1–5) and a delay (5–30s) if you have tight rate limits.
- A `--resume-from` index supports crash recovery: the script writes per-row JSONL as it goes and can skip already-processed rows when resuming.

### 7) Output artifacts

- Per-row JSONL at `results/<run>.jsonl` containing input fields plus an `eval_raw` or `eval_error` entry when parsing fails.
- CSV summary `results/<run>.csv` containing original fields plus the evaluator outputs (or error details).
- Aggregated summary JSON `results/<run>.summary.json` containing means, standard deviations, and deltas across the dataset.

### 8) Error handling and debugging

- If JSON parsing fails, the record stores `error: true`, `error_message`, and a truncated `raw_response` to inspect what the LLM returned.
- Typical causes:
  - LLM returned prose or code fences instead of pure JSON (the extractor strips common fences but inspect `raw_response` to confirm).
  - Quote-escaping or markup in the cell content interfering with JSON production. The script sends raw text for the answer blocks and relies on the model to produce valid JSON.

### 9) How to run (examples)

Test (first 3 rows):

```cmd
set GROQ_API_KEY=your_key_here
python scripts/evaluate_responses.py --input ".\tests\EvalSet.csv" --output ".\results\eval_results.csv" --batch-size 1 --batch-delay 10 --limit 3
```

Full run:

```cmd
set GROQ_API_KEY=your_key_here
python scripts/evaluate_responses.py --input ".\tests\EvalSet.csv" --output ".\results\eval_results.csv" --batch-size 5 --batch-delay 15
```

PowerShell (set env var):

```powershell
$env:GROQ_API_KEY='your_key_here'
python .\scripts\evaluate_responses.py --input ".\tests\EvalSet.csv" --output ".\results\eval_results.csv" --batch-size 5 --batch-delay 15
```

### 10) Troubleshooting checklist

- If you get `UnicodeDecodeError` when loading CSV: try `encoding='latin-1'` or re-save the CSV as UTF-8.
- If you get `PermissionError` writing the CSV: ensure output file is not open in another program and you have write permissions to `results/`.
- If `eval_raw` contains unexpected text: paste the `raw_response` into a text file and inspect whether the LLM returned JSON, code fences, or an explanatory paragraph — then tweak the prompt or extractor.

### 11) Next steps / improvements

- Add small unit tests that mock the LLM output to validate JSON extraction and aggregation behavior.
- Add optional schema validation for the returned JSON to ensure required keys/scores are present before aggregation.
- Add support for alternate LLM providers behind a single client interface to make provider switching testable.

---

If you'd like, I can also: (a) add a small notebook cell that evaluates one row (we already added a todo for this), or (b) add a schema example and sample valid JSON response to the docs.
