import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The quick brown"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(inputs.input_ids, max_new_tokens=30, do_sample=False)
print(tokenizer.decode(outputs[0]))
