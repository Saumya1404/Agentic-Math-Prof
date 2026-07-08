r"""Evaluate initial and refined model responses against a golden answer using Groq.

Usage (PowerShell):
  $env:GROQ_API_KEY = "gsk_..."
  python .\scripts\evaluate_responses.py --input tests/u_math_eval_subset.csv --output results/eval_results.csv

Usage (cmd):
  set GROQ_API_KEY=gsk_...
  python .\scripts\evaluate_responses.py --input tests\u_math_eval_subset.csv --output results\eval_results.csv

Requirements: pandas, groq

Outputs:
  - CSV at --output with per-row numeric scores and rationales
  - JSONL at the same path with per-row JSON (optional)
  - JSON summary with aggregated statistics

Design: For each row, the script asks Groq's llama-3.3-70b-versatile to score the `initial_response` 
and `refined_response` on metrics (Correctness, Clarity, Simplicity, Overall accuracy) on a 0-10 
integer scale and provide short rationales for each score. The LLM is asked to respond with strict 
JSON to simplify parsing.

Features:
- Intelligent batching to avoid rate limits (--batch-size, --batch-delay)
- Crash recovery (--resume-from)
- Automatic rate limit detection and extended backoff
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, Tuple

import pandas as pd
from groq import Groq


PROMPT_TEMPLATE = r"""
You are an exacting evaluator of short math responses. For the example below, produce a JSON object with this exact schema:

{
  "initial": {
    "Correctness": int,      # 0-10
    "Clarity": int,          # 0-10
    "Simplicity": int,       # 0-10
    "Overall": int,          # 0-10
    "rationale": str         # 1-2 sentence explanation for the scores
  },
  "refined": { ... same fields ... },
  "meta": {
    "interpretation": str    # optional short note if the model changed interpretation
  }
}

Rules:
- Use integers 0 through 10 only for scores.
- Do NOT add extra top-level keys.
- Keep rationales short (max 2 sentences each).
- Score "Correctness" relative to the GOLDEN_ANSWER provided below.

Now evaluate the pair below.

GOLDEN_ANSWER:
""" + "{golden_answer}" + r"""

INITIAL_RESPONSE:
""" + "{initial}" + r"""

REFINED_RESPONSE:
""" + "{refined}" + r"""

Return ONLY the JSON object (no extra commentary). If you cannot evaluate because the response lacks enough info, set Correctness=0 and explain in the rationale.
"""


def _extract_json_block(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks and extra text."""
    # Remove markdown code fences if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # Try to find a JSON object in the text (greedy to get the largest/outermost one)
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        return m.group(0)
    
    # Fallback: return the text as-is and let json.loads fail with a clear error
    return text.strip()


def _call_groq_chat(prompt: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> str:
    """Call Groq API with exponential backoff and rate limit handling."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set. Set it with: $env:GROQ_API_KEY='gsk_...'")
    
    client = Groq(api_key=groq_api_key)
    messages = [
        {"role": "system", "content": "You are an evaluator assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Exponential backoff with extended delays for rate limits
    for attempt in range(8):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1500  # Limit output tokens to conserve quota
            )
            return response.choices[0].message.content
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate_limit" in error_str or "rate limit" in error_str:
                # Extract wait time from error message if available
                wait_match = re.search(r"(\d+)m(\d+)", str(e))
                if wait_match:
                    wait_minutes = int(wait_match.group(1))
                    wait_seconds = int(wait_match.group(2).split('.')[0])
                    total_wait = (wait_minutes * 60) + wait_seconds + 5  # Add 5s buffer
                    print(f"Rate limit hit. Waiting {total_wait}s as suggested by API...", file=sys.stderr)
                    time.sleep(total_wait)
                else:
                    # Exponential backoff with longer delays for rate limits
                    base_wait = (2 ** attempt) * 10  # 10s, 20s, 40s, 80s, 160s...
                    jitter = time.time() % 5  # Add jitter to avoid thundering herd
                    total_wait = base_wait + jitter
                    print(f"Rate limit error (attempt {attempt+1}/8). Waiting {total_wait:.1f}s...", file=sys.stderr)
                    time.sleep(total_wait)
            else:
                # Other errors: shorter backoff
                sleep = (2 ** attempt) + (0.1 * attempt)
                print(f"Groq call failed (attempt {attempt+1}/8): {e}. Retrying in {sleep:.1f}s...", file=sys.stderr)
                time.sleep(sleep)
    
    raise RuntimeError("Groq API calls failed after retries. Consider using --batch-delay to space out requests.")


def call_evaluator(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq API to evaluate responses."""
    return _call_groq_chat(prompt, model=model)


def evaluate_row(golden: str, initial: str, refined: str, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    # Don't escape quotes - the prompt template uses raw strings and the LLM will handle them
    prompt = PROMPT_TEMPLATE.format(golden_answer=golden,
                                     initial=initial,
                                     refined=refined)
    raw = call_evaluator(prompt, model=model)
    # extract JSON
    try:
        json_text = _extract_json_block(raw)
        parsed = json.loads(json_text)
        # Add the raw response for debugging if needed
        parsed['_raw_llm_response'] = raw[:500]  # first 500 chars only
    except Exception as e:
        # fallback: return error structure with more debug info
        return {
            "error": True, 
            "raw_response": raw[:1000],  # show first 1000 chars
            "error_message": f"JSON parse failed: {str(e)}",
            "attempted_json": json_text[:500] if 'json_text' in locals() else "N/A"
        }
    return parsed


def aggregate(df_scores: pd.DataFrame) -> Dict[str, Any]:
    # produce simple averages and count
    metrics = ["Correctness", "Clarity", "Simplicity", "Overall"]
    res = {"n": len(df_scores)}
    for side in ("initial", "refined"):
        for m in metrics:
            col = f"{side}_{m}"
            if col in df_scores:
                vals = pd.to_numeric(df_scores[col], errors="coerce").dropna()
                res[f"{col}_mean"] = float(vals.mean()) if len(vals) else None
                res[f"{col}_std"] = float(vals.std()) if len(vals) else None
            else:
                res[f"{col}_mean"] = None
                res[f"{col}_std"] = None
    # improvement deltas
    for m in metrics:
        ini = f"initial_{m}"
        ref = f"refined_{m}"
        if ini in df_scores and ref in df_scores:
            res[f"delta_{m}_mean"] = float(pd.to_numeric(df_scores[ref], errors="coerce").mean() - pd.to_numeric(df_scores[ini], errors="coerce").mean())
        else:
            res[f"delta_{m}_mean"] = None
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV input with golden_answer, initial_response, refined_response columns")
    p.add_argument("--output", required=True, help="CSV output path to write scores (created if needed)")
    p.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model to use (default: llama-3.3-70b-versatile)")
    p.add_argument("--limit", type=int, help="Limit rows processed (for debugging)")
    p.add_argument("--batch-size", type=int, default=5, help="Process N rows then pause (default: 5)")
    p.add_argument("--batch-delay", type=float, default=10.0, help="Seconds to wait between batches (default: 10)")
    p.add_argument("--resume-from", type=int, default=0, help="Resume from row N (0-indexed, for crash recovery)")
    args = p.parse_args(argv)

    # Try reading CSV with different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(args.input, encoding=encoding)
            print(f"Successfully read CSV with {encoding} encoding", file=sys.stderr)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise ValueError(f"Could not read {args.input} with any of the tried encodings: {encodings}")
    
    # Normalize column names to lowercase for case-insensitive matching
    df.columns = df.columns.str.strip().str.lower()
    
    # Verify required columns exist
    required_cols = ['golden_answer', 'initial_response', 'refined_response']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    n = len(df)
    if args.limit:
        df = df.head(args.limit)
    
    # Resume support: skip already-processed rows
    if args.resume_from > 0:
        print(f"Resuming from row {args.resume_from}...", file=sys.stderr)
        df = df.iloc[args.resume_from:]

    rows_out = []
    jsonl_path = os.path.splitext(args.output)[0] + ".jsonl"
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    batch_count = 0
    for idx, row in df.iterrows():
        actual_idx = idx if args.resume_from == 0 else idx
        print(f"Evaluating row {actual_idx+1}/{n}...", file=sys.stderr)
        golden = str(row.get("golden_answer", ""))
        initial = str(row.get("initial_response", ""))
        refined = str(row.get("refined_response", ""))

        try:
            parsed = evaluate_row(golden, initial, refined, model=args.model)
        except Exception as e:
            parsed = {"error": True, "error_message": str(e)}

        # flatten into columns
        out = dict(row)
        out.update({"eval_raw": parsed})

        if not parsed.get("error"):
            for side in ("initial", "refined"):
                side_obj = parsed.get(side, {})
                for key in ("Correctness", "Clarity", "Simplicity", "Overall"):
                    out[f"{side}_{key}"] = side_obj.get(key)
                out[f"{side}_rationale"] = side_obj.get("rationale")
        else:
            out["eval_error"] = parsed.get("error_message")

        rows_out.append(out)

        # append to JSONL immediately (crash recovery)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        
        batch_count += 1
        
        # Batch delay to avoid rate limits
        if batch_count % args.batch_size == 0 and batch_count < len(df):
            print(f"Batch of {args.batch_size} complete. Waiting {args.batch_delay}s to avoid rate limits...", file=sys.stderr)
            time.sleep(args.batch_delay)

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(args.output, index=False)

    summary = aggregate(df_out)
    summary_path = os.path.splitext(args.output)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "="*60)
    print("✅ Evaluation Complete")
    print("="*60)
    print(f"Processed: {len(rows_out)} rows")
    print(f"Outputs:")
    print(f" - CSV: {args.output}")
    print(f" - JSONL: {jsonl_path}")
    print(f" - Summary: {summary_path}")
    print("\nSummary Stats:")
    
    # Handle None values in summary gracefully
    def fmt_stat(val, default="N/A"):
        return f"{val:.2f}" if val is not None else default
    
    init_corr = summary.get('initial_Correctness_mean')
    init_std = summary.get('initial_Correctness_std', 0)
    ref_corr = summary.get('refined_Correctness_mean')
    ref_std = summary.get('refined_Correctness_std', 0)
    delta = summary.get('delta_Correctness_mean')
    
    print(f" - Initial Correctness: {fmt_stat(init_corr)} ± {fmt_stat(init_std, '0')}")
    print(f" - Refined Correctness: {fmt_stat(ref_corr)} ± {fmt_stat(ref_std, '0')}")
    print(f" - Delta (improvement): {fmt_stat(delta)}")
    print("="*60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
