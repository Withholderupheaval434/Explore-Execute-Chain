from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "TingheOliver/Explore-Execute-Chain-Qwen"
subfolder = "Qwen3-8B-E2C-SFT-RL"

tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(model_name, subfolder=subfolder)

# Test example: Fibonacci sequence
inputs = tokenizer("What is the 10th number in the Fibonacci sequence?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
