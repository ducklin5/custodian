import safetensors
from safetensors import safe_open
import os
import glob

model_path = os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/*/model.safetensors")
files = glob.glob(model_path)
if files:
    model_file = files[0]
    print(f"Found model file: {model_file}")
    
    with safe_open(model_file, framework="pt", device="cpu") as f:
        keys = f.keys()
        attn_keys = [key for key in keys if "self_attn" in key and "layers.0" in key]
        print("Layer 0 Attention keys:")
        for key in sorted(attn_keys):
            tensor = f.get_tensor(key)
            print(f"  {key}: shape={list(tensor.shape)}")
else:
    print("Model file not found")
