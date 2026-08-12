'''Outline the 10-step synthetic data pipeline.'''

STEPS = [
    "1. Seed: 200-500 high-quality human-authored examples.",
    "2. Expand: Evol-Instruct-style transformations by strong teacher.",
    "3. Diversify: multiple teacher models, temperatures.",
    "4. Distill: teacher generates outputs for many inputs.",
    "5. Filter: 10-stage quality pipeline.",
    "6. Deduplicate: MinHash + semantic.",
    "7. Contamination check against golden eval.",
    "8. Human sample review: 100+ spot-checked.",
    "9. Version and freeze.",
    "10. Fine-tune on the resulting dataset.",
]

if __name__ == "__main__":
    for s in STEPS:
        print(s)
