'''Illustrate the chat template verification workflow. The #1 SFT bug.

In production, use tokenizer.apply_chat_template and decode back to text
to verify character-level match.'''

CHATML_TEMPLATE = '''<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>'''

LLAMA31_TEMPLATE = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>'''


def apply_template(template: str, system: str, user: str, assistant: str) -> str:
    return template.format(system=system, user=user, assistant=assistant)


if __name__ == "__main__":
    print("ChatML:\n" + apply_template(CHATML_TEMPLATE, "You help.", "Hi", "Hello!"))
    print("\nLlama-3.1:\n" + apply_template(LLAMA31_TEMPLATE, "You help.", "Hi", "Hello!"))
    print("\nRule: verify tokenize → decode matches character-for-character.")
