'''Estimate training compute cost for typical scenarios.'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import training_utils

if __name__ == "__main__":
    scenarios = [
        (7, 5e7, "h100", "qlora"),
        (13, 5e7, "h100", "qlora"),
        (70, 5e7, "h100", "qlora"),
        (7, 5e9, "h100", "full"),  # CPT scale
    ]
    for model_b, n_tokens, gpu, method in scenarios:
        r = training_utils.estimate_cost(model_b, n_tokens, gpu, method)
        print(f"{model_b}B {method} on {gpu} for {n_tokens:.0e} tokens: "
              f"{r.get('estimated_hours', '?')}h, ~${r.get('estimated_cost_usd', '?')}")
