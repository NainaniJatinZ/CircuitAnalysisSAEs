# %%
import os 
import gc
import torch
os.chdir("/work/pi_jensen_umass_edu/jnainani_umass_edu/CircuitAnalysisSAEs")
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload
import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal
from transformer_lens.utils import Slice, SliceInput
import sys 
import functools
import re
from collections import defaultdict
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import json
from sae_lens import SAE, HookedSAETransformer
from utils import plot
from circ4latents import data_gen
# sys.path.append("../../utils/")
with open("config.json", 'r') as file:
    config = json.load(file)
    token = config.get('huggingface_token', None)
os.environ["HF_TOKEN"] = token
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

hf_cache = "/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub"
os.environ["HF_HOME"] = "/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub"


# %%

model = HookedSAETransformer.from_pretrained("google/gemma-2-9b", device=device, cache_dir=hf_cache)

# %%
sae, cfg_dict, sparsity = SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id="layer_21/width_16k/canonical", device=device)

# %%
from transformer_lens.utils import test_prompt

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> print("weight" + "20")
"""
# model.add_sae(sae)
# test_prompt(prompt, "Traceback", model, prepend_space_to_answer=False)
logits = model.run_with_saes(prompt, saes = [sae])
# model.reset_saes()

topk_stuff = torch.topk(logits[0, -1, :], 5)

print(model.tokenizer.decode(topk_stuff.indices))
print(topk_stuff.values)

# %%

# # define a random tensor of size [1, 1, d_model]
# d_model = model.cfg.d_model
# random_tensor = torch.randn(1, 1, d_model).to(device)

# sae(random_tensor).shape

# %%


def filtered_hook(act, hook, sae):
    print(f"act shape: {act.shape}")
    act = sae(act)
    print(f"act shape after SAE: {act.shape}")
    return act
hook_point = sae.cfg.hook_name
model.add_hook(hook_point, partial(filtered_hook, sae = sae), "fwd")
logits = model(prompt)
model.reset_hooks()

topk_stuff = torch.topk(logits[0, -1, :], 5)
print(model.tokenizer.decode(topk_stuff.indices))
print(topk_stuff.values)

# %%

model.tokenizer.pad_token_id
# %%
import torch
from functools import partial

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> print("weight" + 20)
"""

tokens = model.to_tokens(prompt) 

exempt_token_ids = [
    model.tokenizer.bos_token_id,
    model.tokenizer.eos_token_id,
    model.tokenizer.pad_token_id
]

mask = torch.ones_like(tokens, dtype=torch.bool)
for token_id in exempt_token_ids:
    mask &= tokens != token_id  # Update mask to False where token is in exempt list

# Define the modified hook function
def filtered_hook(act, hook, sae, mask):
    # act shape: [batch_size, seq_len, hidden_size]
    print(f"act shape: {act.shape}")
    mask_expanded = mask.unsqueeze(-1).expand_as(act)
    act = torch.where(mask_expanded, sae(act), act)
    print(f"act shape after SAE: {act.shape}")
    return act

# Add the hook to the model
hook_point = sae.cfg.hook_name
model.add_hook(hook_point, partial(filtered_hook, sae=sae, mask=mask), "fwd")

logits = model(tokens)
model.reset_hooks()

# Proceed with your analysis
topk_stuff = torch.topk(logits[0, -1, :], 5)
print(model.tokenizer.decode(topk_stuff.indices))
print(topk_stuff.values)

# %%


import torch
from functools import partial

def run_with_sae_filtered(tokens, filtered_ids, model, sae):
    # Ensure tokens are a torch.Tensor
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)

    # Create a mask where True indicates positions to modify
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in filtered_ids:
        mask &= tokens != token_id 

    # Define the modified hook function
    def filtered_hook(act, hook, sae, mask):
        # act shape: [batch_size, seq_len, hidden_size]
        # Expand mask to match the shape of act
        mask_expanded = mask.unsqueeze(-1).expand_as(act)
        # Apply sae only to positions where mask is True
        act = torch.where(mask_expanded, sae(act), act)
        return act

    hook_point = sae.cfg.hook_name
    model.add_hook(hook_point, partial(filtered_hook, sae=sae, mask=mask), "fwd")
    logits = model(tokens)
    model.reset_hooks()
    return logits

# Define your tokens and filtered token IDs
prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> print("weight" + 20)
"""
tokens = model.to_tokens(prompt)  # Convert prompt to token IDs

# Define the token IDs you want to exempt (e.g., bos, eos, pad tokens)
filtered_ids = [
    model.tokenizer.bos_token_id,
    model.tokenizer.eos_token_id,
    model.tokenizer.pad_token_id
]

# Run the function
logits = run_with_sae_filtered(tokens, filtered_ids, model, sae)

# Proceed with your analysis
topk_stuff = torch.topk(logits[0, -1, :], 3)

for tok_id in topk_stuff.indices:
    print(model.tokenizer.decode(tok_id))
# print(model.tokenizer.decode(topk_stuff.indices))
print(topk_stuff.values)

# %%

logits = model(prompt)
# Proceed with your analysis
topk_stuff = torch.topk(logits[0, -1, :], 3)

for tok_id in topk_stuff.indices:
    print(model.tokenizer.decode(tok_id))
# print(model.tokenizer.decode(topk_stuff.indices))
print(topk_stuff.values)

# %%


import torch
from functools import partial

def run_with_saes_filtered(tokens, filtered_ids, model, saes):
    # Ensure tokens are a torch.Tensor
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)

    # Create a mask where True indicates positions to modify
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in filtered_ids:
        mask &= tokens != token_id

    # For each SAE, add the appropriate hook
    for sae in saes:
        hook_point = sae.cfg.hook_name

        # Define the modified hook function
        def filtered_hook(act, hook, sae=sae, mask=mask):
            # act shape: [batch_size, seq_len, hidden_size]
            # Expand mask to match the shape of act
            mask_expanded = mask.unsqueeze(-1).expand_as(act)
            # Apply sae only to positions where mask is True
            act = torch.where(mask_expanded, sae(act), act)
            return act

        # Add the hook to the model
        model.add_hook(hook_point, filtered_hook, dir='fwd')

    # Run the model with the tokens
    logits = model(tokens)

    # Reset the hooks after computation
    model.reset_hooks()
    return logits

# %%

from tasks.error_detection.type.data import generate_samples

selected_templates = [1] #, 2, 3, 4, 5]
N = 50
samples = generate_samples(selected_templates, N)
for sample in samples[0]:
    prompt = sample
    print(prompt)

# %%
# Regular expression pattern to match the string inside print("...")
pattern = r'print\("([^"]+)"'

# Function to get the first match from each string in the list
def extract_first_match(input_list):
    first_matches = []
    for item in input_list:
        match = re.search(pattern, item)  # Use re.search to get only the first match
        if match:
            first_matches.append(match.group(1))  # Append the matched string to the list
    return first_matches

# Get the first match from each string in the list
first_matches = extract_first_match(samples[0])

# Print the results
for match in first_matches:
    print(match)


# %%

no_err_toks = model.to_tokens(first_matches, prepend_bos=False)[:, 0]
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]


selected_pos  = {
    # "s_start": [],
    # "s_end": [],
    "i_start": [],
    "i_end": [],
    "end": []
}

for i in range(N):
    str_tokens_clean = model.to_str_tokens(samples[0][i])
    str_tokens_corr = model.to_str_tokens(samples[1][i])
    # Find the positions with differences
    diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]
    pos_end = len(str_tokens_clean) - 1  # The last position
    # print(diff_positions, pos_end)
    selected_pos["i_start"].append(diff_positions[0])
    selected_pos["i_end"].append(diff_positions[-1])
    selected_pos["end"].append(pos_end)
selected_pos

# %%
# %%

for param in model.parameters():
    param.requires_grad_(False)

# %%
# model.add_sae(sae)
with torch.no_grad():
    logits = model(samples[1])
# tokens = model.to_tokens(samples[1])
# logits = run_with_saes_filtered(tokens, [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id], model, [sae])
err = logits[range(logits.size(0)), selected_pos["end"], :][:, traceback_token_id]
no_err = logits[range(logits.size(0)), selected_pos["end"], no_err_toks]
# no_err = logits[range(logits.size(0)), selected_pos["end"], :][:, trip_arrow_token_id]
print((err - no_err).mean())
# model.reset_saes()
# %%
tokens.shape
# %%
filtered_ids = [
    model.tokenizer.bos_token_id,
    model.tokenizer.eos_token_id,
    model.tokenizer.pad_token_id
]
mask = torch.ones_like(tokens, dtype=torch.bool)
for token_id in filtered_ids:
    mask &= tokens != token_id

mask.shape
# %%
mask[0, :]


# %% START

from tasks.error_detection.type.data import generate_samples

selected_templates = [2] #, 2, 3, 4, 5]
N = 50
samples = generate_samples(selected_templates, N)
for sample in samples[0]:
    prompt = sample
    print(prompt)
    break

selected_pos  = {
    # "s_start": [],
    # "s_end": [],
    "i_start": [],
    "i_end": [],
    "end": []
}

for i in range(N):
    str_tokens_clean = model.to_str_tokens(samples[0][i])
    str_tokens_corr = model.to_str_tokens(samples[1][i])
    # Find the positions with differences
    diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]
    pos_end = len(str_tokens_clean) - 1  # The last position
    # print(diff_positions, pos_end)
    selected_pos["i_start"].append(diff_positions[0])
    selected_pos["i_end"].append(diff_positions[-1])
    selected_pos["end"].append(pos_end)
selected_pos

# %%
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]

def logit_diff_fn(logits, selected_pos, traceback_token_id=traceback_token_id, trip_arrow_token_id=trip_arrow_token_id):
    err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
    no_err = logits[range(logits.size(0)), selected_pos, :][:, trip_arrow_token_id]
    return (err - no_err).mean()


# %%
import torch
from functools import partial

def run_with_saes_filtered(tokens, filtered_ids, model, saes):
    # Ensure tokens are a torch.Tensor
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)

    # Create a mask where True indicates positions to modify
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in filtered_ids:
        mask &= tokens != token_id

    # For each SAE, add the appropriate hook
    for sae in saes:
        hook_point = sae.cfg.hook_name

        # Define the modified hook function
        def filtered_hook(act, hook, sae=sae, mask=mask):
            # act shape: [batch_size, seq_len, hidden_size]
            # Expand mask to match the shape of act
            mask_expanded = mask.unsqueeze(-1).expand_as(act)
            # Apply sae only to positions where mask is True
            act = torch.where(mask_expanded, sae(act), act)
            return act

        # Add the hook to the model
        model.add_hook(hook_point, filtered_hook, dir='fwd')

    # Run the model with the tokens
    logits = model(tokens)

    # Reset the hooks after computation
    model.reset_hooks()
    return logits

# %%
for param in model.parameters():
    param.requires_grad_(False)

clean_tokens = model.to_tokens(samples[0])
corr_tokens = model.to_tokens(samples[1])

# clean 
logits = model(clean_tokens)
clean_diff = logit_diff_fn(logits, selected_pos['end'])

# corr
logits = model(corr_tokens)
corr_diff = logit_diff_fn(logits, selected_pos['end'])

print(f"clean_diff: {clean_diff}")
print(f"corr_diff: {corr_diff}")

# %%
import gc 

del logits 
gc.collect()

# %%

!nvidia-smi

# %%

# clean with sae 
logits = run_with_saes_filtered(clean_tokens, [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id], model, [sae])
clean_sae_diff = logit_diff_fn(logits, selected_pos['end'])

# corr with sae
logits = run_with_saes_filtered(corr_tokens, [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id], model, [sae])
corr_sae_diff = logit_diff_fn(logits, selected_pos['end'])

print(f"clean_sae_diff: {clean_sae_diff}")
print(f"corr_sae_diff: {corr_sae_diff}")

del logits 
gc.collect()

# %%

def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = logit_diff_fn(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=selected_pos['end'])

# print(f"err_metric_denoising: {err_metric_denoising(logits)}")

# %%
mask



# %%

from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint
# from torchtyping import TensorType as TT

def get_cache_fwd_and_bwd(
    model,
    tokens,
    metric,
    sae,
    error_term: bool = True,
    retain_graph: bool = True
):
    # torch.set_grad_enabled(True)
    model.reset_hooks()
    # model.reset_saes()
    cache = {}
    grad_cache = {}
    filter_base_acts = lambda name: "blocks.21.hook_resid_post" in name
    # filter_sae_acts = lambda name: "hook_sae_acts_post" in name

    def forward_cache_hook(act, hook):
        act.requires_grad_(True)
        # act.retain_graph()
        cache[hook.name] = act.detach()

    def backward_cache_hook(grad, hook):
        grad.requires_grad_(True)
        # grad.retain_graph()
        grad_cache[hook.name] = grad.detach()

    # sae.use_error_term = error_term
    # model.add_sae(sae)
    model.add_hook(filter_base_acts, forward_cache_hook, "fwd")
    model.add_hook(filter_base_acts, backward_cache_hook, "bwd")
    # logits = run_with_saes_filtered(tokens, [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id], model, [sae])
    value = metric(model(tokens)) #logits)
    value.backward() #retain_graph=retain_graph)

    model.reset_hooks()
    # model.reset_saes()
    # torch.set_grad_enabled(False)
    return (
        value,
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )

# %%

clean_value, clean_cache, _ = get_cache_fwd_and_bwd(model, clean_tokens, err_metric_denoising, sae)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
# print("Clean Gradients Cached:", len(clean_grad_cache))

corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, corr_tokens, err_metric_denoising, sae)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# %%

sae_acts = sae.encode(clean_cache['blocks.10.hook_resid_post'])
sae_acts_corr = sae.encode(corrupted_cache['blocks.10.hook_resid_post'])
print(sae_acts.shape, sae_acts_corr.shape)

sae_grad_cache = torch.einsum('bij,kj->bik', corrupted_grad_cache['blocks.10.hook_resid_post'], sae.W_dec)
print(sae_grad_cache.shape)