# %%
import json
import os 
with open("../../config.json", 'r') as file:
    config = json.load(file)
    token = config.get('huggingface_token', None)

os.environ["HF_TOKEN"] = token

import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload

import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import types
from transformer_lens.utils import Slice, SliceInput

import functools
import re
from collections import defaultdict
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# Imports for displaying vis in Colab / notebook

torch.set_grad_enabled(False)

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

import torch
from collections import defaultdict

# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

import random

# %%

model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device = device)


# %%

import random

# Expanded Value sets for clean and corrupted prompts
values_string_concat = {
    "string_value": ["age:", "height:", "value:", "name:", "score:", "distance:", "price:", "weight:", "rating:", "color:", "temperature:", "location:"],
    "int_value": [25, 100, 42, 7, 63, 150, 200, 10, 50, 120, 300, 500]
}

values_list_add = {
    "list_var": ["my_list", "numbers", "values", "data", "array", "collection", "dataset", "elements", "items", "sequence", "collection", "entries"],
    "int_value": [5, 10, 100, 2, 20, 50, 75, 1, 33, 44, 99, 150, 250]
}

values_int_str_op = {
    "int_value": [10, 42, 7, 5, 100, 123, 88, 9, 16, 28, 58, 91, 77],
    "string_value": ["20", "45", "30", "60", "10", "99", "33", "50", "19", "47", "28", "88", "65"]
}

values_func_arg_mismatch = {
    "str_value": ["three", "four", "five", "six", "ten", "eight", "twelve", "seventeen", "twenty", "zero", "nine", "fifteen", "eighteen"]
}

values_str_int_concat = {
    "str_value": ["Result", "Value", "Score", "Height", "Age", "Weight", "Price", "Distance", "Time", "Temperature", "Rating", "Speed", "Quantity"],
    "int_value": [10, 42, 7, 100, 25, 150, 75, 50, 200, 120, 300, 500, 90]
}

# Templates for clean and corrupted prompts (unchanged)
CLEAN_PROMPT_TEMPLATE_1 = """
print("{string_value} " + {int_value})
# When this code is executed, Python will raise a"""

CORRUPTED_PROMPT_TEMPLATE_1 = """
print("{string_value} " + "{int_value}"
# When this code is executed, Python will raise a"""

CLEAN_PROMPT_TEMPLATE_2 = """
{list_var} = [1, 2, 3]
{list_var} += {int_value}
# When this code is executed, Python will raise a"""

CORRUPTED_PROMPT_TEMPLATE_2 = """
{list_var} = [1, 2, 3
{list_var} += {int_value}
# When this code is executed, Python will raise a"""

CLEAN_PROMPT_TEMPLATE_3 = """
result = {int_value} + "{string_value}"
# When this code is executed, Python will raise a"""

CORRUPTED_PROMPT_TEMPLATE_3 = """
result = {int_value} + "{string_value}
# When this code is executed, Python will raise a"""

CLEAN_PROMPT_TEMPLATE_4 = """
sum([1, 2, "{str_value}"])
# When this code is executed, Python will raise a"""

CORRUPTED_PROMPT_TEMPLATE_4 = """
sum([1, 2, "{str_value}"]
# When this code is executed, Python will raise a"""

CLEAN_PROMPT_TEMPLATE_5 = """
result = "{str_value}: " + {int_value}
# When this code is executed, Python will raise a"""

CORRUPTED_PROMPT_TEMPLATE_5 = """
result = "{str_value}: " + {int_value}
# When this code is executed, Python will raise a"""

# Expanded Value sets (updated above)

# Templates mapped to corresponding value sets
templates_and_values = [
    {"clean_template": CLEAN_PROMPT_TEMPLATE_1, "corrupted_template": CORRUPTED_PROMPT_TEMPLATE_1, "values": values_string_concat},
    # {"clean_template": CLEAN_PROMPT_TEMPLATE_2, "corrupted_template": CORRUPTED_PROMPT_TEMPLATE_2, "values": values_list_add},
    # {"clean_template": CLEAN_PROMPT_TEMPLATE_3, "corrupted_template": CORRUPTED_PROMPT_TEMPLATE_3, "values": values_int_str_op},
    # {"clean_template": CLEAN_PROMPT_TEMPLATE_4, "corrupted_template": CORRUPTED_PROMPT_TEMPLATE_4, "values": values_func_arg_mismatch},
    # {"clean_template": CLEAN_PROMPT_TEMPLATE_5, "corrupted_template": CORRUPTED_PROMPT_TEMPLATE_5, "values": values_str_int_concat}
]

def generate_code_prompts(templates_and_values, N, seed=None):
    clean_prompts = []
    corrupted_prompts = []
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Loop over the templates and values to generate prompts
    for template_set in templates_and_values:
        clean_template = template_set["clean_template"]
        corrupted_template = template_set["corrupted_template"]
        values = template_set["values"]
        
        # Get the keys and lists of values
        keys = list(values.keys())
        value_sets = [dict(zip(keys, combination)) for combination in zip(*values.values())]
        
        # Shuffle value_sets to introduce randomness
        random.shuffle(value_sets)

        # For each combination of values, generate clean and corrupted prompts
        for val_set in value_sets:
            clean_prompt = clean_template.format(**val_set)
            corrupted_prompt = corrupted_template.format(**val_set)
            clean_prompts.append(clean_prompt)
            corrupted_prompts.append(corrupted_prompt)
    
    # Shuffle the final list of prompts
    combined_prompts = list(zip(clean_prompts, corrupted_prompts))
    random.shuffle(combined_prompts)
    
    # Return only N randomly selected prompts
    return combined_prompts[:N]

def get_end_positions(tokenized_prompts, eos_token_id, pad_token_id):
    """
    Function to find the END position of each prompt, based on EOS or padding tokens.
    Args:
    - tokenized_prompts: A list of tokenized prompts (with padding).
    - eos_token_id: The token ID representing the end-of-sequence (EOS) token.
    - pad_token_id: The token ID representing the padding token.
    
    Returns:
    - end_positions: A list of integers indicating the end position for each tokenized prompt.
    """
    end_positions = []
    
    for tokens in tokenized_prompts:
        if eos_token_id in tokens:
            # Find the position of the EOS token (if it exists)
            end_pos = tokens.index(eos_token_id)
        else:
            # If no EOS, find the last non-padding token
            end_pos = len(tokens) - 1
            while end_pos >= 0 and tokens[end_pos] == pad_token_id:
                end_pos -= 1
        
        # Add the END position to the list
        end_positions.append(end_pos)  # +1 because positions are 0-indexed
    
    return end_positions

# Example of using the above with your generated prompts

# Generate prompts
N = 10
seed = 42
generated_prompts = generate_code_prompts(templates_and_values, N, seed)

for i, (clean_prompt, corr_prompt) in enumerate(generated_prompts):
    print(f"Prompt {i+1} (clean): {clean_prompt}")
    print(f"Prompt {i+1} (corrupted): {corr_prompt}")
    print()

# Separate clean and corrupted prompts
clean_prompts = [clean_prompt for clean_prompt, _ in generated_prompts]
corr_prompts = [corr_prompt for _, corr_prompt in generated_prompts]

# %% testing 
    
from transformer_lens.utils import test_prompt

# Test the clean and corrupted prompts
test_prompt(clean_prompts[0], " Syntax", model)
# %% testing 
    
from transformer_lens.utils import test_prompt

# Test the clean and corrupted prompts
test_prompt(corr_prompts[0], " Syntax", model)

# %% 


# Tokenize the clean and corrupted prompts (assuming 'model' has a .to_tokens method)
clean_tokens = model.to_tokens(clean_prompts)
corr_tokens = model.to_tokens(corr_prompts)

# Assuming model.tokenizer provides eos_token_id and pad_token_id
eos_token_id = model.tokenizer.eos_token_id  # End-of-sequence token ID
pad_token_id = model.tokenizer.pad_token_id  # Padding token ID

# Get the END positions for clean and corrupted prompts
clean_end_positions = get_end_positions(clean_tokens, eos_token_id, pad_token_id)
corr_end_positions = get_end_positions(corr_tokens, eos_token_id, pad_token_id)

# Now you have clean_tokens, corr_tokens, and their corresponding END positions

print(clean_tokens.shape, corr_tokens.shape)
print(clean_end_positions, corr_end_positions)

# %%

# prompt = """print(price + "200")
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# """ 
# prompt = """print("age: " + "20"
#                        ^
# Python will raise a"""

prompt = """print("b")
# When this code is executed, Python will raise a"""

test_prompt(prompt, "TypeError", model)


# %%
print("price: " + "200")
# %%

saes = [
    SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=str(device),
    )[0]
    for layer in tqdm(range(model.cfg.n_layers))
]

