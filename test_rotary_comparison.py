import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import hashlib

def tensor_hash(tensor):
    """Compute a hash of the tensor for debugging purposes - concatenate all elements as strings and MD5 hash"""
    # Convert to numpy and flatten all values
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().flatten().astype(np.float32)
    else:
        data = np.array(tensor).flatten().astype(np.float32)
    
    # Round each element to 3 decimal places and concatenate to string
    rounded_elements = [f"{val:.3f}" for val in data]
    tensor_string = "".join(rounded_elements)
    
    # MD5 hash the concatenated string
    hash_bytes = hashlib.md5(tensor_string.encode()).digest()
    # Convert to 64-bit integer for display (little-endian to match Rust)
    return int.from_bytes(hash_bytes[:8], byteorder='little')

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary positional embedding manually to match HuggingFace implementation"""
    def rotate_half(x):
        """Rotate half the hidden dims of the input tensor"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # cos/sin have shape [seq_len, head_dim//2], we need to repeat for full head_dim
    cos_full = torch.cat([cos, cos], dim=-1)  # [seq_len, head_dim]
    sin_full = torch.cat([sin, sin], dim=-1)  # [seq_len, head_dim]
    
    # Index by position and add dimensions for broadcasting: [1, 1, seq_len, head_dim]
    cos_full = cos_full[position_ids].unsqueeze(1)  
    sin_full = sin_full[position_ids].unsqueeze(1)  
    
    q_embed = (q * cos_full) + (rotate_half(q) * sin_full)
    k_embed = (k * cos_full) + (rotate_half(k) * sin_full)
    return q_embed, k_embed

# Test rotary encoding specifically
model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The quick brown"
inputs = tokenizer(prompt, return_tensors="pt")

print("=== PYTHON ROTARY ENCODING DEBUG ===")
print("Input tokens:", inputs.input_ids)

# Get the first layer to extract Q, K, V projections and rotary encoding components
first_layer = model.model.layers[0]
config = model.config

# Extract model components for manual computation
embeddings = model.model.embed_tokens(inputs.input_ids)
print(f"Embeddings: hash={tensor_hash(embeddings):016x}, shape={embeddings.shape}")

# Apply input layer norm
normalized_input = first_layer.input_layernorm(embeddings)
print(f"After input norm: hash={tensor_hash(normalized_input):016x}, shape={normalized_input.shape}")

# Get Q, K, V projections
q = first_layer.self_attn.q_proj(normalized_input)
k = first_layer.self_attn.k_proj(normalized_input)
v = first_layer.self_attn.v_proj(normalized_input)

print(f"Q projection: hash={tensor_hash(q):016x}, shape={q.shape}")
print(f"K projection: hash={tensor_hash(k):016x}, shape={k.shape}")
print(f"V projection: hash={tensor_hash(v):016x}, shape={v.shape}")

# Reshape for multi-head attention
batch_size, seq_len, _ = q.shape
num_heads = config.num_attention_heads
num_key_value_heads = config.num_key_value_heads
head_dim = config.hidden_size // num_heads

# Reshape and transpose: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k = k.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
v = v.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

print(f"Q reshaped: hash={tensor_hash(q):016x}, shape={q.shape}")
print(f"K reshaped: hash={tensor_hash(k):016x}, shape={k.shape}")
print(f"V reshaped: hash={tensor_hash(v):016x}, shape={v.shape}")

# Debug the attention layer structure
print(f"Attention layer attributes: {dir(first_layer.self_attn)}")

# Create rotary embedding manually to match the model's configuration
# From HuggingFace implementation
def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

# Create rotary embedding components
head_dim = config.hidden_size // config.num_attention_heads
cos, sin = precompute_freqs_cis(head_dim, seq_len, config.rope_theta)
print(f"Rotary cos: hash={tensor_hash(cos):016x}, shape={cos.shape}")
print(f"Rotary sin: hash={tensor_hash(sin):016x}, shape={sin.shape}")

# Apply rotary encoding
position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

print(f"Q after RoPE: hash={tensor_hash(q_rot):016x}, shape={q_rot.shape}")
print(f"K after RoPE: hash={tensor_hash(k_rot):016x}, shape={k_rot.shape}")

# Test individual tensor values for debugging
print(f"\nFirst few Q values before RoPE: {q[0, 0, 0, :5].tolist()}")
print(f"First few Q values after RoPE: {q_rot[0, 0, 0, :5].tolist()}")
print(f"First few K values before RoPE: {k[0, 0, 0, :5].tolist()}")
print(f"First few K values after RoPE: {k_rot[0, 0, 0, :5].tolist()}")

# Simulate KV cache behavior
print(f"\n=== SIMULATING KV CACHE AND REPEAT_KV ===")

# For the first forward pass, KV cache just returns the same K,V tensors
# In subsequent passes, it would concatenate with previous cached values
# Since this is the first token sequence, cache behavior is identity
k_cached = k_rot.clone()  # Simulate cache returning the same tensor
v_cached = v.clone()      # V doesn't go through RoPE

print(f"After KV cache: K hash={tensor_hash(k_cached):016x}, V hash={tensor_hash(v_cached):016x}")

# Simulate repeat_kv for grouped query attention
# num_heads=9, num_key_value_heads=3, so we need to repeat each KV head 3 times
n_rep = num_heads // num_key_value_heads  # 9 // 3 = 3

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    # Expand: [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_kv_heads, n_rep, seq_len, head_dim]
    x = x.unsqueeze(2).expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    # Reshape: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

k_repeated = repeat_kv(k_cached, n_rep)
v_repeated = repeat_kv(v_cached, n_rep)

print(f"After repeat_kv: K hash={tensor_hash(k_repeated):016x}, V hash={tensor_hash(v_repeated):016x}")
print(f"K shape after repeat_kv: {k_repeated.shape}")
print(f"V shape after repeat_kv: {v_repeated.shape}")

# Verify shapes match expectations
print(f"Expected K shape: [1, {num_heads}, {seq_len}, {head_dim}]")
print(f"Actual K shape: {k_repeated.shape}")
assert k_repeated.shape == (1, num_heads, seq_len, head_dim), f"K shape mismatch"
assert v_repeated.shape == (1, num_heads, seq_len, head_dim), f"V shape mismatch"

print(f"\nRotary cos values for pos=0: {cos[0, :5].tolist()}")
print(f"Rotary sin values for pos=0: {sin[0, :5].tolist()}")

# Also test the rotary computation step by step
print(f"\n=== MANUAL ROTARY COMPUTATION ===")
q_test = q[0, 0, 0, :]  # First head, first token
print(f"Original Q (first token, first head): {q_test[:8].tolist()}")

# Split into first and second half
q1 = q_test[:head_dim//2]  
q2 = q_test[head_dim//2:]
print(f"Q first half: {q1[:4].tolist()}")
print(f"Q second half: {q2[:4].tolist()}")

# Rotate half: (-q2, q1)
q_rotated = torch.cat([(-q2), q1], dim=0)
print(f"Rotated Q: {q_rotated[:8].tolist()}")

# Apply cos and sin with full dimension
cos_full = torch.cat([cos[0], cos[0]], dim=0)  # Repeat to full head_dim
sin_full = torch.cat([sin[0], sin[0]], dim=0)  # Repeat to full head_dim
q_final = q_test * cos_full + q_rotated * sin_full
print(f"Final Q after RoPE: {q_final[:8].tolist()}")

# Compare with actual rotary result
print(f"Actual RoPE result: {q_rot[0, 0, 0, :8].tolist()}")

# Verify they match
diff = torch.abs(q_final - q_rot[0, 0, 0, :])
print(f"Difference (should be ~0): max={diff.max().item():.10f}")