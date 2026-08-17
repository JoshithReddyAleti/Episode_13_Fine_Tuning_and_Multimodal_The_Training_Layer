"""
Dataset utilities for fine-tuning.

Provides:
- validate_dataset(): sanity checks (length distribution, dedup, format)
- estimate_tokens(): count tokens for cost estimation
- split_dataset(): train/val/test with source grouping

Usage:
    python -m src.utils.dataset_utils validate path/to/dataset.jsonl
    python -m src.utils.dataset_utils estimate path/to/dataset.jsonl --model llama-3.1
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_dataset(path: str) -> dict[str, Any]:
    """Run sanity checks on a dataset. Returns a report dict."""
    p = Path(path)
    records = _load_jsonl(p)
    total = len(records)

    # Length distribution
    input_lens = []
    output_lens = []
    dupes = Counter()

    for r in records:
        # Support alpaca (instruction/input/output) and messages format
        if "messages" in r:
            inp = " ".join(m.get("content", "") for m in r["messages"] if m.get("role") == "user")
            out = " ".join(m.get("content", "") for m in r["messages"] if m.get("role") == "assistant")
        else:
            inp = str(r.get("instruction", "")) + str(r.get("input", ""))
            out = str(r.get("output", ""))
        input_lens.append(len(inp))
        output_lens.append(len(out))
        h = hashlib.sha256((inp + out).encode()).hexdigest()
        dupes[h] += 1

    exact_dupes = sum(1 for c in dupes.values() if c > 1)

    def stats(vals):
        if not vals:
            return {}
        vals_sorted = sorted(vals)
        return {
            "min": vals_sorted[0],
            "p50": vals_sorted[len(vals_sorted) // 2],
            "p95": vals_sorted[int(len(vals_sorted) * 0.95)],
            "max": vals_sorted[-1],
            "mean": sum(vals_sorted) / len(vals_sorted),
        }

    report = {
        "file": str(p),
        "total_examples": total,
        "exact_duplicates": exact_dupes,
        "dedup_rate": exact_dupes / total if total else 0.0,
        "input_length_chars": stats(input_lens),
        "output_length_chars": stats(output_lens),
    }
    return report


def estimate_tokens(path: str, chars_per_token: float = 4.0) -> dict[str, Any]:
    """Estimate total token count (rough heuristic: 4 chars/token)."""
    records = _load_jsonl(Path(path))
    total_chars = 0
    for r in records:
        if "messages" in r:
            for m in r["messages"]:
                total_chars += len(m.get("content", ""))
        else:
            total_chars += len(str(r.get("instruction", "")))
            total_chars += len(str(r.get("input", "")))
            total_chars += len(str(r.get("output", "")))
    est_tokens = int(total_chars / chars_per_token)
    return {
        "file": path,
        "total_examples": len(records),
        "total_chars": total_chars,
        "estimated_tokens": est_tokens,
        "chars_per_token": chars_per_token,
    }


def split_dataset(path: str, train_frac: float = 0.9, val_frac: float = 0.05) -> dict[str, int]:
    """Split into train/val/test. Returns counts."""
    records = _load_jsonl(Path(path))
    n = len(records)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    base = Path(path).with_suffix("")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = Path(f"{base}.{name}.jsonl")
        with out.open("w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")

    return {"train": len(train), "val": len(val), "test": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset utilities for fine-tuning")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Sanity-check a dataset")
    p_val.add_argument("path")

    p_est = sub.add_parser("estimate", help="Estimate token count / cost")
    p_est.add_argument("path")
    p_est.add_argument("--chars-per-token", type=float, default=4.0)

    p_split = sub.add_parser("split", help="Split into train/val/test")
    p_split.add_argument("path")
    p_split.add_argument("--train-frac", type=float, default=0.9)
    p_split.add_argument("--val-frac", type=float, default=0.05)

    args = parser.parse_args()

    if args.cmd == "validate":
        report = validate_dataset(args.path)
    elif args.cmd == "estimate":
        report = estimate_tokens(args.path, args.chars_per_token)
    elif args.cmd == "split":
        report = split_dataset(args.path, args.train_frac, args.val_frac)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
