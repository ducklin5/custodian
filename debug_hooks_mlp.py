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

# Storage for captured values
captured_values = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            # For attention, take first element (hidden states)
            captured_values[name] = output[0].clone()
        else:
            captured_values[name] = output.clone()
        print(f"{name}: hash={tensor_hash(captured_values[name]):016x}")
        if 'ffn_norm' in name:
            print(f"{name} tensor (first 20 values): {captured_values[name].flatten()[:20].tolist()}")
    return hook

with torch.no_grad():
    # Register hooks to capture intermediate values
    hooks = []
    
    # Hook on embeddings
    hooks.append(model.model.embed_tokens.register_forward_hook(make_hook('embeddings')))
    
    # Hook on first layer components  
    hooks.append(first_layer.input_layernorm.register_forward_hook(make_hook('attn_norm')))
    hooks.append(first_layer.self_attn.register_forward_hook(make_hook('attention')))
    hooks.append(first_layer.post_attention_layernorm.register_forward_hook(make_hook('ffn_norm')))
    hooks.append(first_layer.mlp.register_forward_hook(make_hook('mlp')))
    
    # Run the model to trigger hooks
    _ = model(inputs.input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print("\n=== DETAILED PYTHON MLP DEBUG ===")
    # Now we have the correct FFN input from the ffn_norm hook
    ffn_input = captured_values['ffn_norm']
    print(f"FFN input: hash={tensor_hash(ffn_input):016x}")
    print(f"FFN input tensor (first 20 values): {ffn_input.flatten()[:20].tolist()}")
    
    # Get the MLP
    mlp = first_layer.mlp
    
    # Debug weight information
    print(f"Gate proj weight: hash={tensor_hash(mlp.gate_proj.weight):016x}, shape={list(mlp.gate_proj.weight.shape)}")
    print(f"Up proj weight: hash={tensor_hash(mlp.up_proj.weight):016x}, shape={list(mlp.up_proj.weight.shape)}")
    print(f"Down proj weight: hash={tensor_hash(mlp.down_proj.weight):016x}, shape={list(mlp.down_proj.weight.shape)}")
    print(f"Gate proj weight (first 10 values): {mlp.gate_proj.weight.flatten()[:10].tolist()}")
    
    # Step by step MLP computation
    gate_output = mlp.gate_proj(ffn_input)
    print(f"After gate_proj: hash={tensor_hash(gate_output):016x}")
    print(f"Gate proj output (first 20 values): {gate_output.flatten()[:20].tolist()}")
    
    up_output = mlp.up_proj(ffn_input)
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
    direct_mlp_output = mlp(ffn_input)
    print(f"Direct MLP output: hash={tensor_hash(direct_mlp_output):016x}")
    
    print(f"Outputs match: {torch.allclose(final_output, direct_mlp_output, atol=1e-6)}")
    
    # Also verify against captured MLP output
    if 'mlp' in captured_values:
        captured_mlp = captured_values['mlp']
        print(f"Captured MLP output: hash={tensor_hash(captured_mlp):016x}")
        print(f"Direct vs Captured match: {torch.allclose(final_output, captured_mlp, atol=1e-6)}")