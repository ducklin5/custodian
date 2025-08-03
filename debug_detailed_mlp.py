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

# Get the first layer MLP to debug it step by step
first_layer = model.model.layers[0]
mlp = first_layer.mlp

with torch.no_grad():
    # Run the full forward pass to get the intermediate states
    outputs = model(inputs.input_ids, return_dict=True, output_hidden_states=True)
    
    # Get the embeddings and first layer input
    embeddings = model.model.embed_tokens(inputs.input_ids)
    print(f"Embeddings: hash={tensor_hash(embeddings):016x}")
    
    # First layer attention norm
    attn_norm_output = first_layer.input_layernorm(embeddings)
    print(f"After attn norm: hash={tensor_hash(attn_norm_output):016x}")
    
    # First layer attention - get intermediate
    hidden_states = embeddings
    attention_mask = torch.ones_like(inputs.input_ids)
    
    # Apply attention - use the same approach as the working test_llama.py
    # We'll use the full model forward to get the intermediate states
    # Let's skip the manual attention computation and use the hook approach
    
    # Register hooks to capture the MLP input
    ffn_norm_input = None
    def capture_ffn_norm_input(module, input, output):
        global ffn_norm_input
        ffn_norm_input = input[0].clone()
        
    hook = first_layer.post_attention_layernorm.register_forward_hook(capture_ffn_norm_input)
    
    # Run forward pass to trigger the hook
    _ = model(inputs.input_ids)
    hook.remove()
    
    if ffn_norm_input is None:
        print("Could not capture FFN norm input, using simple approach")
        # Use a simpler approach - just get the intermediate state after first operations
        with torch.no_grad():
            attn_norm_output = first_layer.input_layernorm(embeddings)
            # Simplified - use zero attention output to isolate FFN behavior  
            ffn_norm_input = embeddings  # This is approximate
    
    
    # Now debug MLP step by step with detailed logging
    print("=== DETAILED PYTHON MLP DEBUG ===")
    print(f"FFN input: hash={tensor_hash(ffn_norm_input):016x}")
    print(f"FFN input tensor (first 20 values): {ffn_norm_input.flatten()[:20].tolist()}")
    
    # Debug weight information
    print(f"Gate proj weight: hash={tensor_hash(mlp.gate_proj.weight):016x}, shape={list(mlp.gate_proj.weight.shape)}")
    print(f"Up proj weight: hash={tensor_hash(mlp.up_proj.weight):016x}, shape={list(mlp.up_proj.weight.shape)}")
    print(f"Down proj weight: hash={tensor_hash(mlp.down_proj.weight):016x}, shape={list(mlp.down_proj.weight.shape)}")
    print(f"Gate proj weight (first 10 values): {mlp.gate_proj.weight.flatten()[:10].tolist()}")
    
    # Step by step MLP computation
    gate_output = mlp.gate_proj(ffn_norm_input)
    print(f"After gate_proj: hash={tensor_hash(gate_output):016x}")
    print(f"Gate proj output (first 20 values): {gate_output.flatten()[:20].tolist()}")
    
    up_output = mlp.up_proj(ffn_norm_input)
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
    direct_mlp_output = mlp(ffn_norm_input)
    print(f"Direct MLP output: hash={tensor_hash(direct_mlp_output):016x}")
    
    print(f"Outputs match: {torch.allclose(final_output, direct_mlp_output, atol=1e-6)}")
