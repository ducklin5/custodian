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
attn = model.model.layers[0].self_attn

print("=== Python Attention Weight Hashes and Shapes ===")
print(f"Q proj weight: hash={tensor_hash(attn.q_proj.weight):016x}, shape={attn.q_proj.weight.shape}")
print(f"K proj weight: hash={tensor_hash(attn.k_proj.weight):016x}, shape={attn.k_proj.weight.shape}")  
print(f"V proj weight: hash={tensor_hash(attn.v_proj.weight):016x}, shape={attn.v_proj.weight.shape}")
print(f"O proj weight: hash={tensor_hash(attn.o_proj.weight):016x}, shape={attn.o_proj.weight.shape}")
