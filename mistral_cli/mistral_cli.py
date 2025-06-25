import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Paths
LORA_PATH = "./"
TOKENIZER_PATH = "./tokenizer"

# Load LoRA config
peft_config = PeftConfig.from_pretrained(LORA_PATH)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Load base model (no quantization)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map="cpu",  # Force to CPU
    trust_remote_code=True
)

# Apply LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.to("cpu")  # Move to CPU

# Generate response
def generate_response(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# CLI Entry
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mistral_cli.py \"<your prompt>\"")
        sys.exit(1)

    prompt = sys.argv[1]
    print(f"\nYour Prompt: {prompt}\n")
    print("Stacbot Response:\n")
    print(generate_response(prompt))
