# %%
import json
import os 
with open("../../config.json", 'r') as file:
    config = json.load(file)
    token = config.get('huggingface_token', None)

os.environ["HF_TOKEN"] = token

# %% 
import os
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from neel_plotly import line, imshow, scatter
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
# %% Testing





# %% dataset definition

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

# Separate clean and corrupted prompts
clean_prompts = [clean_prompt for clean_prompt, _ in generated_prompts]
corr_prompts = [corr_prompt for _, corr_prompt in generated_prompts]

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

# %% logit diff test
model.to_str_tokens(clean_tokens[1])
# %%

clean_logits = model(clean_tokens)
corr_logits = model(corr_tokens)

print(clean_logits.shape, corr_logits.shape)

# %%

# Ensure clean_end_positions are valid
print(f"clean_end_positions: {clean_end_positions}")
print(f"Max end position: {max(clean_end_positions)}")
print(f"clean_logits shape: {clean_logits.shape}")  # Should be [batch_size, seq_len, vocab_size]

# %%

syntax_token_id = model.tokenizer.encode(" Syntax", add_special_tokens=False)[0]
print(f"Syntax token ID: {syntax_token_id}")
print(f"Vocab size: {clean_logits.size(-1)}")  # This should be equal to vocab_size


# %%

# syn_tok = model.tokenizer.encode(" Syntax")[-1]
# type_tok = model.tokenizer.encode(" TypeError")[-1]
# # print(model.tokenizer.decode(syn_tok[-1]))

# syn_tok_list = [syn_tok] * clean_logits.size(0)
# type_tok_list = [type_tok] * clean_logits.size(0)

# %%

syntax_token_id = model.tokenizer.encode(" Syntax", add_special_tokens=False)[0]
syntax_token_id

# %%

syn_logits = clean_logits[range(clean_logits.size(0)), clean_end_positions, :][:, syntax_token_id]
syn_logits


# %%
type_token_id = model.tokenizer.encode(" TypeError", add_special_tokens=False)[0]
# type_token_id
type_logits = clean_logits[range(clean_logits.size(0)), clean_end_positions, :][:, type_token_id]
type_logits

# %%

logit_diff = (type_logits - syn_logits).mean()
logit_diff


# %%

def logit_diff_error_type(logits, end_positions, err1_tok =type_token_id, err2_tok = syntax_token_id):
    err1_logits = logits[range(logits.size(0)), end_positions, :][:, err1_tok]
    err2_logits = logits[range(logits.size(0)), end_positions, :][:, err2_tok]
    logit_diff = (err1_logits - err2_logits).mean()
    return logit_diff

# %%

clean_logit_diff = logit_diff_error_type(clean_logits, clean_end_positions)
corr_logit_diff = logit_diff_error_type(corr_logits, clean_end_positions)
print(f"Clean logit diff: {clean_logit_diff}")
print(f"Corrupted logit diff: {corr_logit_diff}")

# %% patching metric

def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = logit_diff_error_type(logits, end_positions)
    return ((patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff))

err_metric_denoising = partial(
    _err_type_metric,
    clean_logit_diff=clean_logit_diff,
    corr_logit_diff=corr_logit_diff,
    end_positions=clean_end_positions,
) 

# %% 

print(f"Clean Baseline is 1: {err_metric_denoising(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {err_metric_denoising(corr_logits).item():.4f}")


# %%

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corr_tokens)

# %% residual stream patching
import transformer_lens.patching as patching
import sys 
sys.path.append("../../")
from utils import plot
resid_pre_act_patch_results = patching.get_act_patch_resid_pre(model, corr_tokens, clean_cache, err_metric_denoising)
plot.imshow(resid_pre_act_patch_results, 
       yaxis="Layer", 
       xaxis="Position", 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
       title="resid_pre Activation Patching")


# %%
attn_head_out_all_pos_act_patch_results = patching.get_act_patch_attn_head_out_all_pos(model, corr_tokens, clean_cache, err_metric_denoising)
plot.imshow(attn_head_out_all_pos_act_patch_results, 
       yaxis="Layer", 
       xaxis="Head", 
       title="attn_head_out Activation Patching (All Pos)")



# %%

ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
# if DO_SLOW_RUNS:
attn_head_out_act_patch_results = patching.get_act_patch_attn_head_out_by_pos(model, corr_tokens, clean_cache, err_metric_denoising)
attn_head_out_act_patch_results = einops.rearrange(attn_head_out_act_patch_results, "layer pos head -> (layer head) pos")
plot.imshow(attn_head_out_act_patch_results, 
    yaxis="Head Label", 
    xaxis="Pos", 
    x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
    y=ALL_HEAD_LABELS,
    title="attn_head_out Activation Patching By Pos")


# clean_end_positions 
# %%

print(corr_prompts[0])



# corr_end_positions

# def _ioi_metric_noising(
#         logits: Float[torch.Tensor, "batch seq d_vocab"],
#         clean_logit_diff: float,
#         corrupted_logit_diff: float,
#         ioi_dataset: IOIDataset,
#     ) -> float:
#     '''
#     We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
#     and -1 when performance has been destroyed (i.e. is same as ABC dataset).
#     '''
#     patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
#     return ((patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff))

# %%
# type(syn_tok)

# %%
# clean_logits[range(clean_logits.size(0)), clean_end_positions, syn_tok_list]



# %%



# def logits_to_ave_logit_diff_2(
#     logits: Float[Tensor, "batch seq d_vocab"],
#     ioi_dataset: IOIDataset = ioi_dataset,
#     per_prompt=False
# ) -> Float[Tensor, "*batch"]:
#     '''
#     Returns logit difference between the correct and incorrect answer.

#     If per_prompt=True, return the array of differences rather than the average.
#     '''

#     # Only the final logits are relevant for the answer
#     # Get the logits corresponding to the indirect object / subject tokens respectively
#     io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
#     s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
#     # Find logit difference
#     answer_logit_diff = io_logits - s_logits
#     return answer_logit_diff if per_prompt else answer_logit_diff.mean()