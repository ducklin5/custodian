import torch
from transformers import AutoTokenizer

# Test tokenization
model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "The quick brown"
inputs = tokenizer(prompt, return_tensors="pt")

print("Python tokenizer results:")
print(f"Token IDs: {inputs.input_ids[0].tolist()}")
print(f"Input shape: {inputs.input_ids.shape}")

# Test conversion to float for hashing
input_float = inputs.input_ids.float()
print(f"As float: {input_float[0].tolist()}")

# Print individual token details
for i, token_id in enumerate(inputs.input_ids[0]):
    token_text = tokenizer.decode([token_id])
    print(f"Token {i}: ID={token_id}, text='{token_text}'")