import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Load model to analyze its rotary encoding
model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The quick brown"
inputs = tokenizer(prompt, return_tensors="pt")

print("=== ANALYZING PYTHON ROTARY ENCODING IMPLEMENTATION ===")
print("Input tokens:", inputs.input_ids.tolist())

# Get first layer attention
first_layer = model.model.layers[0].self_attn
config = model.config

print(f"Config - rope_theta: {config.rope_theta}")
print(f"Config - hidden_size: {config.hidden_size}")
print(f"Config - num_attention_heads: {config.num_attention_heads}")
print(f"Config - max_position_embeddings: {config.max_position_embeddings}")

# Let's dive into the actual rotary embedding implementation
# Check if there's a rotary_emb attribute
if hasattr(first_layer, 'rotary_emb'):
    print("Model has rotary_emb attribute")
    rotary_emb = first_layer.rotary_emb
    print(f"Rotary embedding type: {type(rotary_emb)}")
    print(f"Rotary embedding attributes: {dir(rotary_emb)}")
else:
    print("No rotary_emb attribute found")

# Let's look at the actual forward pass and see how position_ids are handled
embeddings = model.model.embed_tokens(inputs.input_ids)
first_layer_block = model.model.layers[0]  # The full layer, not just attention
normalized_input = first_layer_block.input_layernorm(embeddings)

# Get Q, K, V from attention
q = first_layer.q_proj(normalized_input)
k = first_layer.k_proj(normalized_input)
v = first_layer.v_proj(normalized_input)

batch_size, seq_len, _ = q.shape
num_heads = config.num_attention_heads
num_key_value_heads = config.num_key_value_heads
head_dim = config.hidden_size // num_heads

# Reshape and transpose
q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k = k.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

print(f"\nBefore RoPE:")
print(f"Q shape: {q.shape}")
print(f"K shape: {k.shape}")
print(f"Q hash: {tensor_hash(q):016x}")
print(f"K hash: {tensor_hash(k):016x}")

# Now let's implement our own rotary encoding to match exactly
def precompute_freqs_cis(dim, seq_len, theta=10000.0, device=None):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rotary_pos_emb_custom(q, k, cos, sin, position_ids):
    """Apply rotary positional embedding to queries and keys."""
    def rotate_half(x):
        """Rotate half the hidden dims of the input tensor."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Select cos/sin values for the given positions and expand for broadcasting
    cos_selected = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim//2]
    sin_selected = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim//2]
    
    # Expand to full head dimension by repeating
    cos_full = torch.cat([cos_selected, cos_selected], dim=-1)  # [batch, 1, seq_len, head_dim]
    sin_full = torch.cat([sin_selected, sin_selected], dim=-1)  # [batch, 1, seq_len, head_dim]
    
    q_embed = (q * cos_full) + (rotate_half(q) * sin_full)
    k_embed = (k * cos_full) + (rotate_half(k) * sin_full)
    return q_embed, k_embed

# Test our custom implementation
device = q.device
cos, sin = precompute_freqs_cis(head_dim, seq_len, config.rope_theta, device)

print(f"\nCustom RoPE components:")
print(f"Cos shape: {cos.shape}")
print(f"Sin shape: {sin.shape}")
print(f"Cos hash: {tensor_hash(cos):016x}")
print(f"Sin hash: {tensor_hash(sin):016x}")

# Position IDs for first forward pass - should be [0, 1, 2] for "The quick brown"
position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
print(f"Position IDs: {position_ids}")

# Apply our custom rotary encoding
q_custom, k_custom = apply_rotary_pos_emb_custom(q, k, cos, sin, position_ids)

print(f"\nAfter custom RoPE:")
print(f"Q hash: {tensor_hash(q_custom):016x}")
print(f"K hash: {tensor_hash(k_custom):016x}")

# Compare with position 0 only (like in our test)
position_ids_zero = torch.zeros((1, seq_len), dtype=torch.long, device=device)
q_pos0, k_pos0 = apply_rotary_pos_emb_custom(q, k, cos, sin, position_ids_zero)

print(f"\nAfter RoPE with position 0 only:")
print(f"Q hash: {tensor_hash(q_pos0):016x}")
print(f"K hash: {tensor_hash(k_pos0):016x}")

# Check individual values
print(f"\nDetailed comparison:")
print(f"Original Q[0,0,0,:5]: {q[0,0,0,:5].tolist()}")
print(f"Custom RoPE Q[0,0,0,:5]: {q_custom[0,0,0,:5].tolist()}")
print(f"Position 0 Q[0,0,0,:5]: {q_pos0[0,0,0,:5].tolist()}")

print(f"\nCos values for different positions:")
for pos in range(seq_len):
    print(f"Position {pos} cos[:5]: {cos[pos, :5].tolist()}")
    print(f"Position {pos} sin[:5]: {sin[pos, :5].tolist()}")

# Test what happens if we use cache length as position
cache_len = 0  # First forward pass
position_ids_cache = torch.full((1, seq_len), cache_len, dtype=torch.long, device=device)
q_cache, k_cache = apply_rotary_pos_emb_custom(q, k, cos, sin, position_ids_cache)

print(f"\nUsing cache length ({cache_len}) as position:")
print(f"Q hash: {tensor_hash(q_cache):016x}")
print(f"K hash: {tensor_hash(k_cache):016x}")