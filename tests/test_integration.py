'''Integration tests that don't require external services.'''
import json
import tempfile
from src.utils import dataset_utils, training_utils, multimodal_utils


def test_end_to_end_estimate():
    # Create small dataset
    records = [
        {"messages": [{"role": "user", "content": "x" * 100},
                      {"role": "assistant", "content": "y" * 200}]}
        for _ in range(100)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name

    # Validate
    report = dataset_utils.validate_dataset(path)
    assert report["total_examples"] == 100

    # Estimate tokens
    tokens = dataset_utils.estimate_tokens(path)
    n_tokens = tokens["estimated_tokens"]

    # Estimate training cost
    cost = training_utils.estimate_cost(7, n_tokens * 3, "h100", "qlora")
    assert cost["estimated_cost_usd"] >= 0


def test_lora_and_serving_size_match_intuitively():
    # A 7B model with attn+mlp LoRA at rank 16 should be under 100 MB
    r = training_utils.lora_size(7, rank=16, target="attn_mlp")
    assert r["adapter_size_mb_bf16"] < 100

