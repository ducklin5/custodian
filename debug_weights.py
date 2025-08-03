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

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
mlp = model.model.layers[0].mlp

print("=== Python MLP Weight Hashes ===")
print(f"Gate proj weight: hash={tensor_hash(mlp.gate_proj.weight):016x}")
print(f"Up proj weight: hash={tensor_hash(mlp.up_proj.weight):016x}")  
print(f"Down proj weight: hash={tensor_hash(mlp.down_proj.weight):016x}")

print(f"Gate proj weight shape: {mlp.gate_proj.weight.shape}")
print(f"Up proj weight shape: {mlp.up_proj.weight.shape}")
print(f"Down proj weight shape: {mlp.down_proj.weight.shape}")
