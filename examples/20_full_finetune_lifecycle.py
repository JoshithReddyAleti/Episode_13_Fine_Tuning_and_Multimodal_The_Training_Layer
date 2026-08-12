'''Show the 11-step fine-tuning lifecycle.'''

LIFECYCLE = [
    "1. Problem framing        - what is fine-tuning fixing?",
    "2. Baseline measurement   - prompt/RAG baseline on golden set",
    "3. Success criteria       - what number must move? by how much?",
    "4. Dataset construction   - biggest lever, biggest time sink (50-70%)",
    "5. Method selection       - SFT? DPO? CPT? PEFT variant?",
    "6. Infrastructure setup   - single GPU? multi-GPU? cloud?",
    "7. Training runs          - often 5-20 iterations",
    "8. Evaluation             - task metrics + regression + human review",
    "9. Serving decisions      - merge into base? serve as LoRA?",
    "10. Monitoring in prod    - does it perform as expected?",
    "11. Retraining pipeline   - this is not one-shot",
]

if __name__ == "__main__":
    print("Fine-tuning lifecycle:")
    for s in LIFECYCLE:
        print(f"  {s}")
    print("\nRule: teams spending 90% on training and 10% on the rest ship the worst fine-tunes.")
