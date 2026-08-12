from src.utils import training_utils


def test_cost_estimate_h100_7b():
    r = training_utils.estimate_cost(7, 1e7, "h100", "qlora")
    assert "estimated_cost_usd" in r
    assert r["estimated_cost_usd"] > 0


def test_lora_size_7b_r16():
    r = training_utils.lora_size(7, rank=16, target="attn_mlp")
    assert r["adapter_size_mb_bf16"] > 0
    assert r["total_trainable_params"] > 0


def test_lr_range_qlora():
    r = training_utils.lr_range(7, "qlora")
    assert r["suggested_lr_low"] < r["suggested_lr_high"]
    assert r["suggested_lr_low"] == 1e-4


def test_lora_size_scales_with_rank():
    small = training_utils.lora_size(7, rank=8)
    big = training_utils.lora_size(7, rank=64)
    assert big["adapter_size_mb_bf16"] > small["adapter_size_mb_bf16"]

