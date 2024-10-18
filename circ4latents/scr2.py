# %% Loading libraries and gemma 2 2 imports 
import os 
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

model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device = device, cache_dir = hf_cache)

# %% 
with open("sae_closest_strings.json", 'r') as file:
    closest_strings = json.load(file)

closest_strings
# %%

layers = [3, 5]
saes = [
    SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id=closest_strings[str(layer)],
        device=str(device)
    )[0]
    for layer in tqdm(layers)
]


# %%

import random
code_like_names = ["print", "sum", "return", "fetch", "process"]
english_phrases = ["hello", "hi", "ask", "tell", "goodbye"]

# Define list of variable names (with inverted commas)
variable_names = ["'age'", "'score'", "'temperature'", "'value'", "'location'", "'distance'"] #, "''", "''"]

# Function to generate N clean and corrupted pairs based on the two templates
def generate_two_templates_pairs(n):
    pairs = []
    
    for _ in range(n):
        # Randomly pick a code-like function and an English phrase
        code_func = random.choice(code_like_names)
        english_phrase = random.choice(english_phrases)
        
        # Randomly pick a variable name with inverted commas
        var_name = random.choice(variable_names)
        
        # Template 1: code-like vs. non-code-like
        clean_1 = f"{code_func} {var_name}"
        corrupted_1 = f"{english_phrase} {var_name}"
        
        # Template 2: function call-like vs. plain
        clean_2 = f"{english_phrase}({var_name})"
        corrupted_2 = f"{english_phrase} {var_name}"
        
        # Append both pairs to the list
        pairs.append((clean_1, corrupted_1))
        pairs.append((clean_2, corrupted_2))
    
    return pairs
n = 10
pairs = generate_two_templates_pairs(n)
for clean, corrupted in pairs:
    print(f"Clean: {clean}\nCorrupted: {corrupted}\n")

clean_prompts = [pair[0] for pair in pairs]
corrupted_prompts = [pair[1] for pair in pairs]

# %%
lat_ind = 8566
layer_ind = 5

def latent_patch_metric(cache):
    # layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
    result = cache[:, :, 8566].sum()
    # print(result.requires_grad)
    return result 
    # return cache[layer_name][:, :, lat_ind].sum()

_, clean_cache = model.run_with_cache_with_saes(clean_prompts, saes=saes[1])
_, corrupted_cache = model.run_with_cache_with_saes(corrupted_prompts, saes=saes[1])
clean_patch = latent_patch_metric(clean_cache[f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'])
corrupted_patch = latent_patch_metric(corrupted_cache[f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'])
print(f"Clean Patch: {clean_patch}, Corrupted Patch: {corrupted_patch}")
def latent_patch_metric_denoising(cache, clean_diff, corr_diff):
    # Make sure the result has requires_grad enabled
    result = (latent_patch_metric(cache) - corr_diff) / (clean_diff - corr_diff)
    # assert result.requires_grad, "Result tensor must require gradients"
    return result

latent_metric_denoising = partial(latent_patch_metric_denoising, lat_ind=lat_ind, clean_diff=clean_patch, corr_diff=corrupted_patch)

# %%

from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint

def get_cache_fwd_and_bwd(
    model,
    tokens,
    metric,
    # layer_name,
    saes,
    error_term: bool = True,
    retain_graph: bool = True
):
    model.reset_hooks()
    cache = {}
    grad_cache = {}
    value_container = {}
    filter_sae_acts = lambda name: "hook_sae_acts_post" in name

    def forward_cache_hook(act, hook):
        cache[hook.name] = act

    def backward_cache_hook(grad, hook):
        grad_cache[hook.name] = grad

    def custom_metric_hook(act, hook):
        value = metric(act)
        value.backward(retain_graph=retain_graph)
        value_container['value'] = value.item()
        return act

    model.add_sae(saes[0])
    model.add_sae(saes[1])
    hook_point = saes[1].cfg.hook_name + ".hook_sae_acts_post"
    model.add_hook(filter_sae_acts, forward_cache_hook, "fwd")
    model.add_hook(filter_sae_acts, backward_cache_hook, "bwd")
    model.add_hook(hook_point, custom_metric_hook, 'fwd')

    model(tokens)

    model.reset_hooks()
    model.reset_saes()

    value = value_container['value']

    return (
        value,
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )

# Usage example
layer_name = 'blocks.5.hook_resid_post.hook_sae_acts_post'
receiver_layer = 'blocks.3.hook_resid_post.hook_sae_acts_post'
clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, clean_prompts, latent_patch_metric, saes
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))

corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, corrupted_prompts, latent_patch_metric, saes
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# %%

corrupted_grad_cache['blocks.3.hook_resid_post.hook_sae_acts_post'].shape
# %%
clean_prompts
# %%
clean_cache[f'blocks.3.hook_resid_post.hook_sae_acts_post'].shape
# %%
corrupted_cache[f'blocks.3.hook_resid_post.hook_sae_acts_post'].shape
# %%
index_layer = 3
resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
clean_residual = clean_cache[resid_point]
corr_residual = corrupted_cache[resid_point]
corr_grad_residual = corrupted_grad_cache[resid_point]

residual_attr_final = einops.reduce(
            corr_grad_residual * (clean_residual - corr_residual),
            "batch pos n_features -> n_features",
            "sum",
        )