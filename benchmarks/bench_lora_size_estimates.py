'''Benchmark LoRA size estimate across configurations.'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from src.utils import training_utils


def main():
    configs = [(m, r, t) for m in [7, 13, 70] for r in [8, 16, 32] for t in ["attn_only", "attn_full", "attn_mlp"]]
    t0 = time.perf_counter()
    for m, r, t in configs:
        training_utils.lora_size(m, rank=r, target=t)
    dt = time.perf_counter() - t0
    print(f"{len(configs)} configs computed in {dt*1000:.2f}ms")


if __name__ == "__main__":
    main()

