import torch
from transformers import AutoModelForCausalLM
import hashlib
import numpy as np

def tensor_hash(tensor):
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().flatten().astype(np.float32)
    else:
        data = np.array(tensor).flatten().astype(np.float32)
    
    rounded_elements = [f"{val:.3f}" for val in data]
    tensor_string = "".join(rounded_elements)
    hash_bytes = hashlib.md5(tensor_string.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="little")

# Compare first few elements to see if they match
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
mlp = model.model.layers[0].mlp

print("=== Python MLP Weight Details ===")
print(f"Gate proj weight hash: {tensor_hash(mlp.gate_proj.weight):016x}")
print(f"Gate proj first 10 elements: {mlp.gate_proj.weight.flatten()[:10].tolist()}")
print(f"Up proj weight hash: {tensor_hash(mlp.up_proj.weight):016x}")
print(f"Up proj first 10 elements: {mlp.up_proj.weight.flatten()[:10].tolist()}")
print(f"Down proj weight hash: {tensor_hash(mlp.down_proj.weight):016x}")
print(f"Down proj first 10 elements: {mlp.down_proj.weight.flatten()[:10].tolist()}")
