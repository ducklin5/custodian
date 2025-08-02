import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The quick brown"
inputs = tokenizer(prompt, return_tensors="pt")

print("=== PYTHON DEBUG INFO ===")
print("Input tokens:", inputs.input_ids)
print("Input text:", tokenizer.decode(inputs.input_ids[0]))
print("Model config:")
print(f"  vocab_size: {model.config.vocab_size}")
print(f"  hidden_size: {model.config.hidden_size}")
print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
print(f"  num_attention_heads: {model.config.num_attention_heads}")
print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
print(f"  intermediate_size: {model.config.intermediate_size}")
print(f"  max_position_embeddings: {model.config.max_position_embeddings}")
print(f"  rms_norm_eps: {model.config.rms_norm_eps}")
print(f"  rope_theta: {model.config.rope_theta}")
print(f"  tie_word_embeddings: {model.config.tie_word_embeddings}")

# Manual generation with debugging
model.eval()
with torch.no_grad():
    input_ids = inputs.input_ids
    print(f"\n=== FORWARD PASS DEBUG ===")
    print(f"Input shape: {input_ids.shape}")
    
    # Get model outputs
    outputs = model(input_ids, return_dict=True, output_hidden_states=True)
    logits = outputs.logits
    hidden_states = outputs.hidden_states
    
    print(f"Logits shape: {logits.shape}")
    print(f"Number of hidden state layers: {len(hidden_states)}")
    
    # Get logits for the last position
    last_logits = logits[0, -1, :]  # [vocab_size]
    print(f"Last position logits shape: {last_logits.shape}")
    print(f"Last position logits min/max: {last_logits.min().item():.6f} / {last_logits.max().item():.6f}")
    
    # Get top 10 tokens
    top_k = 10
    top_values, top_indices = torch.topk(last_logits, top_k)
    print(f"\nTop {top_k} tokens:")
    for i in range(top_k):
        token_id = top_indices[i].item()
        value = top_values[i].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ('{token_text}'): {value:.6f}")
    
    # Get the actual next token (greedy)
    next_token = torch.argmax(last_logits).item()
    next_token_text = tokenizer.decode([next_token])
    print(f"\nGreedy next token: {next_token} ('{next_token_text}')")
    
    # Print some embedding stats
    embeddings = model.model.embed_tokens.weight
    print(f"\nEmbedding weights shape: {embeddings.shape}")
    print(f"Embedding weights min/max: {embeddings.min().item():.6f} / {embeddings.max().item():.6f}")
    
    # Print first layer attention weights sample
    first_layer = model.model.layers[0]
    print(f"\nFirst layer attention weights shape: {first_layer.self_attn.q_proj.weight.shape}")
    print(f"First layer q_proj weights min/max: {first_layer.self_attn.q_proj.weight.min().item():.6f} / {first_layer.self_attn.q_proj.weight.max().item():.6f}")

print("\n=== GENERATION COMPARISON ===")
outputs = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False)
print("Generated with model.generate():", tokenizer.decode(outputs[0]))

# Manual generation to show first few tokens
print("\n=== MANUAL STEP-BY-STEP GENERATION ===")
current_ids = inputs.input_ids.clone()
for step in range(5):
    with torch.no_grad():
        outputs = model(current_ids)
        next_token_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        next_token_text = tokenizer.decode([next_token])
        print(f"Step {step + 1}: Token {next_token} ('{next_token_text}')")
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]])], dim=1)
