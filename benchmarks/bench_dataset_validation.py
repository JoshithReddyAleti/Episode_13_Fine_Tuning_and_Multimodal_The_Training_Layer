'''Benchmark dataset validation throughput.'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import time
from src.utils import dataset_utils


def main():
    for n in [1000, 10000]:
        records = [{"instruction": "x" * 100, "input": "", "output": "y" * 200}] * n
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name
        t0 = time.perf_counter()
        report = dataset_utils.validate_dataset(path)
        dt = time.perf_counter() - t0
        print(f"n={n}: {dt:.3f}s ({n/dt:.0f} records/sec)")


if __name__ == "__main__":
    main()

