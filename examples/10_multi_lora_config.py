'''Show a multi-LoRA vLLM server configuration.'''

VLLM_ARGS = [
    "--model", "meta-llama/Llama-3.1-70B-Instruct",
    "--enable-lora",
    "--max-loras", "16",           # concurrent unique adapters
    "--max-lora-rank", "32",
    "--lora-modules",
    "customer_a=s3://adapters/customer_a",
    "customer_b=s3://adapters/customer_b",
    "customer_c=s3://adapters/customer_c",
    "--tensor-parallel-size", "4",
    "--max-model-len", "8192",
]

if __name__ == "__main__":
    print("vLLM multi-LoRA server args:")
    print("  vllm serve \\")
    print("    " + " \\\n    ".join(VLLM_ARGS))
    print("\nRequests specify adapter via the model field in the API payload.")
