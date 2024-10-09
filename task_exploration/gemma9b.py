# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# Path to the local model directory
model_path = "/datasets/ai/gemma/models--google--gemma-2-9b-it"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model and move it to the GPU (CUDA)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# Create a text generation pipeline, ensure device is set to GPU
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)  # device=0 for the first GPU

# Define the prompt and generation parameters
prompt = "Once upon a time"
max_tokens = 100  # Define the maximum number of tokens to generate

output = pipeline(prompt, max_new_tokens=max_tokens, do_sample=True)

# Print the generated text
print(output[0]['generated_text'])


# %%
