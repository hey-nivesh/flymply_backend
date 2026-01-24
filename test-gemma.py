from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = "You are an aviation assistant. Give a turbulence advisory for probability 0.74."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(out[0], skip_special_tokens=True))
