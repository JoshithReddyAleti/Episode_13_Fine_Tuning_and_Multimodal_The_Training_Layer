'''Show a MergeKit TIES-DARE configuration.'''

CONFIG = '''
models:
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      density: 0.5
      weight: 0.5
  - model: mistralai/Mistral-7B-Instruct-v0.2
    parameters:
      density: 0.5
      weight: 0.5
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  int8_mask: true
dtype: bfloat16
'''

if __name__ == "__main__":
    print("MergeKit TIES-DARE config:")
    print(CONFIG)
