import safetensors
from safetensors import safe_open
import os

# Check if model file exists
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/*/model.safetensors")
import glob
files = glob.glob(model_path)
if files:
    model_file = files[0]
    print(f"Found model file: {model_file}")
    
    with safe_open(model_file, framework="pt", device="cpu") as f:
        keys = f.keys()
        mlp_keys = [key for key in keys if "mlp" in key and "layers.0" in key]
        print("Layer 0 MLP keys:")
        for key in sorted(mlp_keys):
            tensor = f.get_tensor(key)
            print(f"  {key}: shape={list(tensor.shape)}")
else:
    print("Model file not found")
