'''Show why splits by example break for grouped data.'''

import random

# Simulated: 100 examples from 10 sources, 10 each
examples = [(f"src_{i//10}", f"ex_{i}") for i in range(100)]

random.seed(42)

# Wrong: random split by example — sources leak between train and val.
random.shuffle(examples)
train_wrong = examples[:80]
val_wrong = examples[80:]
sources_train = set(s for s, _ in train_wrong)
sources_val = set(s for s, _ in val_wrong)
print(f"Wrong split: overlap sources = {sources_train & sources_val} (LEAKAGE!)")

# Right: split by source
sources = list({s for s, _ in examples})
random.shuffle(sources)
train_sources = set(sources[:8])
val_sources = set(sources[8:])
train_right = [(s, e) for s, e in examples if s in train_sources]
val_right = [(s, e) for s, e in examples if s in val_sources]
print(f"Right split: train sources = {sorted(train_sources)}, val sources = {sorted(val_sources)}")
print(f"No overlap. train={len(train_right)}, val={len(val_right)}")
