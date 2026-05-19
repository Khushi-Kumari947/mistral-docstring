"""
collect_data.py
---------------
Data collection script for fine-tuning Mistral 7B on Python docstring generation.
Source: CodeSearchNet (via HuggingFace datasets) — no API key or account needed.

Output files (in ./data/ folder):
  train.jsonl  — 8000 samples
  val.jsonl    — 1000 samples
  test.jsonl   —  500 samples  <- this is your eval set, keep it untouched

Usage:
  pip install datasets tqdm
  python collect_data.py
"""

import ast
import json
import os
import re
import random
from tqdm import tqdm
from datasets import load_dataset


# ── CONFIG ────────────────────────────────────────────────────────────────────

TRAIN_SIZE  = 8000
VAL_SIZE    = 1000
TEST_SIZE   = 500
SEED        = 42
OUTPUT_DIR  = "./data"

PROMPT_TEMPLATE = (
    "You are a Python documentation expert. "
    "Write a clear, concise NumPy-style docstring for the following Python function.\n\n"
    "### Function:\n{function_code}\n\n"
    "### Docstring:"
)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_function_and_docstring(code_string: str):
    """
    Parse a raw code string and return (function_without_docstring, docstring).
    Returns (None, None) if the function has no docstring or can't be parsed.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return None, None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Must have a docstring as the first statement
        if not (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            continue

        docstring = node.body[0].value.value.strip()

        # Need at least one more statement after the docstring
        if len(node.body) < 2:
            continue

        # Rebuild function without docstring.
        # We avoid deepcopy entirely (breaks on Python 3.12).
        # Instead we unparse each body statement individually.
        try:
            body_lines = [ast.unparse(stmt) for stmt in node.body[1:]]
            args       = ast.unparse(node.args)
            returns    = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            body_str   = "\n    ".join(body_lines)
            function_code = f"def {node.name}({args}){returns}:\n    {body_str}"
        except Exception:
            continue

        return function_code, docstring

    return None, None


def is_quality_sample(function_code: str, docstring: str) -> bool:
    """
    Filter out low-quality samples using simple heuristics.
    Returns True if the sample is worth keeping.
    """
    if len(docstring) < 20 or len(docstring) > 1500:
        return False

    if len(function_code) < 50 or len(function_code) > 3000:
        return False

    skip_phrases = [
        "todo", "fixme", "pass", "not implemented",
        "placeholder", "deprecated", ":nodoc:", "undocumented"
    ]
    if any(p in docstring.lower() for p in skip_phrases):
        return False

    if not re.search(r'[.:\n]', docstring):
        return False

    return True


def make_jsonl_record(function_code: str, docstring: str) -> dict:
    return {
        "prompt": PROMPT_TEMPLATE.format(function_code=function_code),
        "completion": docstring,
        "function_code": function_code,
        "ground_truth_docstring": docstring,
    }


def save_jsonl(records: list, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} samples -> {filepath}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    print("=" * 60)
    print("Step 1: Loading CodeSearchNet (Python) from HuggingFace")
    print("  ~500MB download on first run, cached after that.")
    print("=" * 60)

    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)

    all_samples = []
    for split_name in ["train", "validation", "test"]:
        all_samples.extend(dataset[split_name])

    print(f"  Total raw samples loaded: {len(all_samples):,}")

    print("\nStep 2: Parsing and filtering samples")

    clean_samples = []
    skipped = 0

    for raw in tqdm(all_samples, desc="  Processing"):
        code = raw.get("whole_func_string", "")
        if not code:
            skipped += 1
            continue

        function_code, docstring = extract_function_and_docstring(code)

        if function_code is None or docstring is None:
            skipped += 1
            continue

        if not is_quality_sample(function_code, docstring):
            skipped += 1
            continue

        clean_samples.append(make_jsonl_record(function_code, docstring))

    print(f"  Clean samples kept : {len(clean_samples):,}")
    print(f"  Samples skipped    : {skipped:,}")

    random.shuffle(clean_samples)

    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(clean_samples) < total_needed:
        print(f"\n  WARNING: only {len(clean_samples)} clean samples found, adjusting sizes.")
        ratio = len(clean_samples) / total_needed
        t = int(TRAIN_SIZE * ratio)
        v = int(VAL_SIZE   * ratio)
        e = len(clean_samples) - t - v
    else:
        t, v, e = TRAIN_SIZE, VAL_SIZE, TEST_SIZE

    train = clean_samples[:t]
    val   = clean_samples[t : t + v]
    test  = clean_samples[t + v : t + v + e]

    print("\nStep 3: Saving splits")
    save_jsonl(train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val,   os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test,  os.path.join(OUTPUT_DIR, "test.jsonl"))

    print("\n" + "=" * 60)
    print("Sample from train.jsonl:")
    print("=" * 60)
    print(f"\nPROMPT:\n{train[0]['prompt'][:400]}...")
    print(f"\nCOMPLETION:\n{train[0]['completion'][:300]}...")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("  1. Upload ./data/ folder to Kaggle as a dataset")
    print("  2. Open Kaggle -> New Notebook -> attach dataset")
    print("  3. Run train.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()