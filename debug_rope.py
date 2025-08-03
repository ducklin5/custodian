#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import hashlib

def tensor_hash(tensor):
    """Compute a hash of the tensor for debugging purposes - concatenate all elements as strings and MD5 hash"""
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().flatten().astype(np.float32)
    else:
        data = np.array(tensor).flatten().astype(np.float32)
    
    rounded_elements = [f"{val:.3f}" for val in data]
    tensor_string = "".join(rounded_elements)
    hash_bytes = hashlib.md5(tensor_string.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder='little')

def debug_rope_dimensions():
    # SmolLM-135M config
    head_dim = 64  # 576 / 9
    seq_len = 3
    theta = 10000.0
    
    print(f"Head dim: {head_dim}")
    print(f"Seq len: {seq_len}")
    
    # Compute inv_freq like our Rust implementation
    half_dim = head_dim // 2  # 32
    print(f"Half dim: {half_dim}")
    
    # Create inv_freq: 1.0 / (theta^(2i/head_dim)) for i in [0, half_dim)
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    print(f"inv_freq shape: {inv_freq.shape}")
    print(f"inv_freq: {inv_freq[:5]}")
    
    # Position IDs for seq_len=3: [0, 1, 2]
    position_ids = torch.arange(seq_len, dtype=torch.float32)
    print(f"position_ids shape: {position_ids.shape}")
    print(f"position_ids: {position_ids}")
    
    # Compute outer product
    angles = torch.outer(position_ids, inv_freq)  # [seq_len, half_dim] = [3, 32]
    print(f"angles shape: {angles.shape}")
    
    # Compute cos and sin
    cos_half = torch.cos(angles)  # [3, 32]
    sin_half = torch.sin(angles)  # [3, 32]
    print(f"cos_half shape: {cos_half.shape}")
    print(f"sin_half shape: {sin_half.shape}")
    
    # Expand to full head dimension by repeating
    cos_full = torch.cat([cos_half, cos_half], dim=-1)  # [3, 64]
    sin_full = torch.cat([sin_half, sin_half], dim=-1)  # [3, 64]
    print(f"cos_full shape: {cos_full.shape}")
    print(f"sin_full shape: {sin_full.shape}")
    
    print(f"cos_full hash: {tensor_hash(cos_full):016x}")
    print(f"sin_full hash: {tensor_hash(sin_full):016x}")

if __name__ == "__main__":
    debug_rope_dimensions()