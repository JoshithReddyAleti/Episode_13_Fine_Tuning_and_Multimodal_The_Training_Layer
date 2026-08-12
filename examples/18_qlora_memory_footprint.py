'''Estimate GPU memory for QLoRA training.'''

def estimate_qlora_memory_gb(model_b: float, seq_len: int = 2048, batch: int = 4) -> dict:
    # Rough heuristics
    model_gb = model_b * 0.5  # 4-bit weights approximately 0.5 GB per B params
    activations_gb = seq_len * batch * model_b * 0.02 / 1000  # very rough
    optimizer_gb = 0.3  # 8-bit paged Adam is small
    lora_gb = 0.05 * model_b  # LoRA adapters + gradients
    total = model_gb + activations_gb + optimizer_gb + lora_gb
    return {
        "model_b": model_b,
        "seq_len": seq_len,
        "batch": batch,
        "model_gb": round(model_gb, 1),
        "activations_gb": round(activations_gb, 1),
        "optimizer_gb": optimizer_gb,
        "lora_gb": round(lora_gb, 1),
        "total_gb_estimate": round(total, 1),
    }


if __name__ == "__main__":
    for m in [7, 13, 34, 70]:
        r = estimate_qlora_memory_gb(m)
        print(f"{m}B QLoRA @ seq {r['seq_len']}, batch {r['batch']}: ~{r['total_gb_estimate']} GB")
