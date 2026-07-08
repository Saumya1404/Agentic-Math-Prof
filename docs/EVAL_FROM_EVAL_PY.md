## Overview

This document is a concise, self-contained summary of the behavior, usage, and implementation details of the script `Scripts/Eval.py`. The content below is derived only from that file's source (the structure, functions, and strings in the script).

## Purpose

`Scripts/Eval.py` runs a batched evaluation of math responses using an LLM. For each dataset row it compares a "golden" (ground truth) answer, an "initial" model response, and a "refined" response. It uses an LLM (Groq client) to produce a detailed JSON evaluation per entry, derives additional metrics, and writes progress and final reports to disk.

## Requirements (inferred from script)

- Python packages: pandas, json (stdlib), groq (Groq client), dotenv, time, logging
- Environment variable or .env with: `GROQ_API_KEY` (the script calls `load_dotenv()` and instantiates `Groq(api_key=os.environ.get("GROQ_API_KEY"))`).

## High-level flow

1. Load environment variables using `dotenv.load_dotenv()`.
2. Instantiate a Groq client at module import using `Groq(api_key=os.environ.get("GROQ_API_KEY"))`.
3. Read an evaluation CSV (the `__main__` block reads `D:\programming\python\Math_prof\tests\EvalSet.csv` with `encoding='latin-1'`).
4. Run `evaluate_all_entries()` which:
   - validates required columns: `['question', 'golden_answer', 'initial_response', 'refined_response']`;
   - runs entries in batches (default batch size and delay parameters configurable in function calls);
   - for each entry calls `evaluate_math_response()` to get an LLM-produced JSON evaluation;
   - computes derived metrics and summary metrics per entry;
   - persists intermediate progress after every batch using `save_progress()`;
   - generates a final report and saves full JSON results to disk.

## Key functions and responsibilities

- `evaluate_math_response(golden_answer, initial_response, refined_response, query)`
  - Constructs an evaluation prompt containing the `query`, `GOLDEN ANSWER`, `INITIAL RESPONSE`, and `REFINED RESPONSE`.
  - Requests the LLM (Groq) via `client.chat.completions.create` with system and user messages.
  - Uses parameters: model `llama-3.3-70b-versatile`, temperature `0.1`, `max_tokens=4000`, `top_p=1`, `stream=False`, `response_format={"type": "json_object"}`.
  - Expects the LLM to return a JSON object matching a structured schema (see "Expected LLM output schema" below).
  - Parses the returned `completion.choices[0].message.content` as JSON and returns it. On failure prints an error and returns `None`.

- `calculate_additional_metrics(evaluation_result)`
  - Computes differences between refined and initial scores (e.g., correctness_improvement, simplicity_improvement).
  - Computes percent improvements and combined scores like `refinement_success_score` and simple quality indices.

- `extract_summary_metrics(evaluation_result, additional_metrics)`
  - Reduces the rich LLM output to essential fields for quick reporting: `initial_quality`, `refined_quality`, `improvement_score`, `human_feedback_importance`, and a few derived numbers.

- `print_summary(result_entry, entry_number)`
  - Console-friendly per-entry brief summary used while processing.

- `process_batch(batch, starting_index)`
  - Iterates rows in a batch, calls `evaluate_math_response` for each, computes additional metrics, constructs a result entry, appends to `batch_results`.
  - Skips entries where both `initial_response` and `refined_response` are empty.
  - On exceptions, logs and appends an error entry to maintain order.

- `save_progress(results, batch_start)`
  - Writes a compact JSON file named `evaluation_progress_batch_{batch_start}_{timestamp}.json` containing partial results after each batch.

- `evaluate_batch_entries(df, batch_size=5, delay_between_batches=2)`
  - Orchestrates batching: iterates dataset in windows of `batch_size`, calls `process_batch`, saves progress, and sleeps `delay_between_batches` seconds between batches to avoid rate limits.

- `generate_final_report(results)`
  - Gathers successful results and prints aggregated stats: averages and a quality distribution.
  - Saves a full detailed JSON file called `final_evaluation_results_{timestamp}.json`.

- `evaluate_all_entries(df, batch_size=5, delay_between_batches=2)`
  - High-level wrapper that validates columns and calls batching + reporting. Returns `results`.

## Expected LLM output schema (as requested in the script prompt)

The script instructs the LLM to return a JSON object with this structure (shown verbatim from the evaluation prompt):

{
  "initial_response_evaluation": {
    "correctness_score": 0-10,
    "simplicity_score": 0-10,
    "clarity_score": 0-10,
    "completeness_score": 0-10,
    "step_by_step_quality": 0-10,
    "overall_quality": 0-10,
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"]
  },
  "refined_response_evaluation": { ... same fields ... },
  "comparison_metrics": {
    "improvement_score": 0-10,
    "refinement_effectiveness": 0-10,
    "human_feedback_importance": 0-10,
    "key_improvements": ["list", "of", "specific", "improvements"],
    "remaining_issues": ["list", "of", "remaining", "issues"]
  },
  "quality_metrics": {
    "mathematical_accuracy": 0-10,
    "explanation_quality": 0-10,
    "pedagogical_value": 0-10,
    "error_reduction": 0-10,
    "conceptual_clarity": 0-10
  }
}

Note: the script performs `json.loads()` of the LLM response and relies on these numeric fields to derive additional metrics. If the LLM response does not parse as JSON, the script prints an error and returns `None` for that entry.

## Files produced by the script

- Intermediate per-batch progress files: `evaluation_progress_batch_{batch_start}_{timestamp}.json` (written by `save_progress`).
- Final detailed results: `final_evaluation_results_{timestamp}.json` (written by `generate_final_report`).
- When run from `__main__`, the script also writes `evaluation_results.json` (a final JSON with all results) if any results are present.

## Important behavior notes and known pitfalls (derived from the code)

- Groq client instantiation at import: `client = Groq(api_key=os.environ.get("GROQ_API_KEY"))` is executed at module import time. If `GROQ_API_KEY` is not set, this may raise an error at import time. The script calls `load_dotenv()` before that line, so it will pick up a `.env` if present in the current working directory—however, running the module without the env var set will likely fail early.

- LLM response parsing: The code expects a strict JSON object returned by the LLM and does `json.loads(response_content)`. Any non-JSON commentary or formatting will cause the evaluation for that entry to fail and result in `None` for the LLM output.

- Rate-limiting considerations: The script has `delay_between_batches` to avoid API rate limits; defaults are 2 seconds in the function signature and 120 seconds in the `__main__` invocation.

- Data validation: The script requires the exact columns `['question', 'golden_answer', 'initial_response', 'refined_response']` present in the dataframe. Missing columns will abort with an error message and print available columns.

- Empty responses: Entries where both `initial_response` and `refined_response` are empty are skipped.

## How to run (inferred from `__main__`)

1. Ensure dependencies are installed and `GROQ_API_KEY` is available (in the environment or a `.env` file).
2. Run the script directly (example used in `__main__`):

```powershell
python Scripts/Eval.py
```

The `__main__` block in the script reads `D:\programming\python\Math_prof\tests\EvalSet.csv` with `encoding='latin-1'`, prints the first row preview, and then calls `evaluate_all_entries(eval_df, batch_size=4, delay_between_batches=120)`.

## Recommendations (safe changes suggested by reading the code)

- Defer Groq client creation until runtime (lazy init) so importing the module or importing helper functions doesn't require the API key. E.g., provide a `get_groq_client()` helper that reads `GROQ_API_KEY` and returns a client or raises a clear error.
- Harden LLM response parsing: strip code fences, remove non-JSON commentary, and try a small set of recovery heuristics before failing a row. Alternatively, require the LLM to return exactly one JSON object with a clear wrapper phrase and validate returned schema.
- Add robust logging of raw LLM output on parse failures (already prints an error; persisting the raw response to disk will help debugging).
- Provide CLI flags or `argparse` for input CSV path, batch size, delay, and output directory rather than hard-coding values in `__main__`.

## Quick reference: important strings and parameters from the script

- Model: `llama-3.3-70b-versatile` (passed to `client.chat.completions.create`).
- Temperature: `0.1` (low temperature for deterministic results).
- `max_tokens`: `4000`.
- Response format requested: `{"type": "json_object"}` (script expects a JSON object from the LLM).

## Conclusion

This document summarizes the structure, runtime expectations, outputs, and potential runtime pitfalls of `Scripts/Eval.py`, using only the content present in that file. If you want, I can (a) create a follow-up patch that moves the Groq client to lazy initialization, (b) add basic `argparse` flags for input and output paths, or (c) implement more robust LLM output recovery heuristics.
