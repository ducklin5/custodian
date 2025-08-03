import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import hashlib

def tensor_hash(tensor):
    """Compute a hash of the tensor for debugging purposes"""
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().flatten().astype(np.float32)
    else:
        data = np.array(tensor).flatten().astype(np.float32)
    
    rounded_elements = [f"{val:.3f}" for val in data]
    tensor_string = "".join(rounded_elements)
    hash_bytes = hashlib.md5(tensor_string.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="little")

# Load model
model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The quick brown"
inputs = tokenizer(prompt, return_tensors="pt")

model.eval()

# Get the first layer to debug it step by step
first_layer = model.model.layers[0]

with torch.no_grad():
    # Step 1: Get embeddings
    embeddings = model.model.embed_tokens(inputs.input_ids)
    print(f"Embeddings: hash={tensor_hash(embeddings):016x}")
    
    # Step 2: Input layer norm (attention norm)  
    attn_norm_output = first_layer.input_layernorm(embeddings)
    print(f"After attn norm: hash={tensor_hash(attn_norm_output):016x}")
    
    # Step 3: Self attention
    # We need to manually run the attention to get the output
    # Create position IDs for the input
    position_ids = torch.arange(0, inputs.input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
    
    attention_output = first_layer.self_attn(
        attn_norm_output,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False
    )[0]  # Get just the hidden states, not the cache
    print(f"After attention: hash={tensor_hash(attention_output):016x}")
    
    # Step 4: Attention residual connection
    attn_residual = embeddings + attention_output
    print(f"After attn residual: hash={tensor_hash(attn_residual):016x}")
    
    # Step 5: Post attention layer norm (FFN norm) - THIS IS THE FFN INPUT
    ffn_norm_output = first_layer.post_attention_layernorm(attn_residual)
    print(f"After ffn norm (FFN INPUT): hash={tensor_hash(ffn_norm_output):016x}")
    
    # Now debug MLP step by step with the CORRECT FFN input
    print("\n=== DETAILED PYTHON MLP DEBUG ===")
    print(f"FFN input: hash={tensor_hash(ffn_norm_output):016x}")
    print(f"FFN input tensor (first 20 values): {ffn_norm_output.flatten()[:20].tolist()}")
    
    # Get the MLP
    mlp = first_layer.mlp
    
    # Debug weight information
    print(f"Gate proj weight: hash={tensor_hash(mlp.gate_proj.weight):016x}, shape={list(mlp.gate_proj.weight.shape)}")
    print(f"Up proj weight: hash={tensor_hash(mlp.up_proj.weight):016x}, shape={list(mlp.up_proj.weight.shape)}")
    print(f"Down proj weight: hash={tensor_hash(mlp.down_proj.weight):016x}, shape={list(mlp.down_proj.weight.shape)}")
    print(f"Gate proj weight (first 10 values): {mlp.gate_proj.weight.flatten()[:10].tolist()}")
    
    # Step by step MLP computation
    gate_output = mlp.gate_proj(ffn_norm_output)
    print(f"After gate_proj: hash={tensor_hash(gate_output):016x}")
    print(f"Gate proj output (first 20 values): {gate_output.flatten()[:20].tolist()}")
    
    up_output = mlp.up_proj(ffn_norm_output)
    print(f"After up_proj: hash={tensor_hash(up_output):016x}")
    print(f"Up proj output (first 20 values): {up_output.flatten()[:20].tolist()}")
    
    silu_gate = F.silu(gate_output)
    print(f"After silu(gate): hash={tensor_hash(silu_gate):016x}")
    
    combined = silu_gate * up_output
    print(f"After silu*up: hash={tensor_hash(combined):016x}")
    
    final_output = mlp.down_proj(combined)
    print(f"After down_proj: hash={tensor_hash(final_output):016x}")
    print(f"Final MLP output (first 20 values): {final_output.flatten()[:20].tolist()}")
    print(f"Final MLP output (last 20 values): {final_output.flatten()[-20:].tolist()}")
    
    # Compare with direct MLP call
    direct_mlp_output = mlp(ffn_norm_output)
    print(f"Direct MLP output: hash={tensor_hash(direct_mlp_output):016x}")
    
    print(f"Outputs match: {torch.allclose(final_output, direct_mlp_output, atol=1e-6)}")
    
    # Step 6: FFN residual connection
    ffn_residual = attn_residual + final_output
    print(f"After ffn residual: hash={tensor_hash(ffn_residual):016x}")