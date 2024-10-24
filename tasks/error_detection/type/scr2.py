
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

# Model and configuration
model = HookedSAETransformer.from_pretrained("google/gemma-2-9b", device=device, cache_dir=hf_cache)
sae, cfg_dict, sparsity = SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id="layer_10/width_16k/canonical", device=device)

from transformer_lens.utils import test_prompt
from tasks.error_detection.type.data import generate_samples

selected_templates = [1] #, 2, 3, 4, 5]
N = 20
samples = generate_samples(selected_templates, N)
for sample in samples[0]:
    prompt = sample
    print(prompt)

# Token ID for "Traceback"
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]

def type_error_patch_metric(logits, end_positions, err1_tok=traceback_token_id):
    err1_logits = logits[range(logits.size(0)), end_positions, :][:, err1_tok]
    return err1_logits.mean()

# Function to tokenize samples (cached and reused)
def tokenize_samples(samples):
    tokenized = model.tokenizer(samples, return_tensors="pt", padding=True, truncation=True)
    return tokenized.input_ids.to(device), tokenized.attention_mask.to(device)

# Memory-efficient metric calculation
def calculate_metric(samples):
    input_ids, attention_mask = tokenize_samples(samples)
    with torch.no_grad():  # Disable gradients for inference
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    end_positions = [mask.nonzero()[-1].item() for mask in attention_mask]  # Efficient extraction of end positions
    return type_error_patch_metric(logits, end_positions)

# Efficient metric difference calculation
def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = type_error_patch_metric(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

# %% 
attention_mask = model.tokenizer(samples[1]).attention_mask
end_positions = [len(mask) - mask[::-1].index(1) - 1 for mask in attention_mask]
# err_metric_denoising = partial(_err_type_metric, clean_logit_diff=1, corr_logit_diff=0, end_positions=end_positions)

# %% 
# with torch.no_grad():
#     # sae.use_error_term = False
#     # model.add_sae(sae)
#     logits_clean = model(samples[1])
#     # model.reset_saes()
# probs = logits_clean.softmax(dim=-1)
# print(probs.shape)
# err1_logits = logits_clean[range(logits_clean.size(0)), end_positions, :] #[:, err1_tok]
# probs[range(logits_clean.size(0)), end_positions, :][:, traceback_token_id].mean()

# %%
for param in model.parameters():
    param.requires_grad_(False)

# %%

def type_error_patch_metric_prob(logits, end_positions, err1_tok=traceback_token_id):
    probs = logits.softmax(dim=-1)
    err1_logits = probs[range(logits.size(0)), end_positions, :][:, err1_tok]
    return err1_logits.mean()

with torch.no_grad():
    logits = model(samples[0])
clean_diff = type_error_patch_metric_prob(logits, end_positions)
print(clean_diff)
with torch.no_grad():
    logits = model(samples[1])
corr_diff = type_error_patch_metric_prob(logits, end_positions)
print(corr_diff)

# %%

def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = type_error_patch_metric_prob(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=end_positions)

# %%

import gc 
del logits
gc.collect()

# %%

from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint

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
    filter_base_acts = lambda name: "blocks.10.hook_resid_post" in name
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
    value = metric(model(tokens))
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
err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=end_positions)
clean_value, clean_cache, clean_grad = get_cache_fwd_and_bwd(model, samples[0], err_metric_denoising, sae)


# %%

print(len(clean_cache), len(clean_grad))
# %%

clean_grad['blocks.10.hook_resid_post']

# %%
clean_grad['blocks.10.hook_resid_post.hook_sae_acts_post'].shape

# %%

corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, samples[1], err_metric_denoising, sae)


# %%

sae_acts = sae.encode(clean_cache['blocks.10.hook_resid_post'])
sae_acts.shape
# %%
sae_acts_corr = sae.encode(corrupted_cache['blocks.10.hook_resid_post'])
sae_acts_corr.shape


# %%

sae.W_dec.shape


# %%
corrupted_grad_cache['blocks.10.hook_resid_post'].shape

# %%

# Using einsum to perform the matrix multiplication
sae_grad_cache = torch.einsum('bij,kj->bik', corrupted_grad_cache['blocks.10.hook_resid_post'], sae.W_dec)

# The result will have the shape [20, 33, 16384]
sae_grad_cache.shape

# %%

sae_clean_cache = sae.decode(sae_acts)
sae_corr_cache = sae.decode(sae_acts_corr)
print(sae_clean_cache.shape, sae_corr_cache.shape)

# %%
residual_attr_final = einops.reduce(
    sae_grad_cache * (sae_acts - sae_acts_corr),
    "batch pos n_features -> n_features",
    "sum",
)
residual_attr_final.shape

# %%
torch.topk(residual_attr_final, 10).values


# %%

indices = torch.topk(residual_attr_final, 10).indices
values = torch.topk(residual_attr_final, 10).values


# %%
from IPython.display import IFrame
def get_dashboard_html(sae_release="gemma-2-9b", sae_id="10-gemmascope-res-16k", feature_idx=0):
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)


for idx, val in zip(indices, values):
    print(f"Feature Index: {idx}, Value: {val}")
    html = get_dashboard_html(sae_id="10-gemmascope-res-16k", feature_idx=idx)
    display(IFrame(html, width=1200, height=300))
    
# html = get_dashboard_html(sae_id="10-gemmascope-res-16k", feature_idx=indices[0])
# display(IFrame(html, width=1200, height=300))





# %% 
samples[0][1]

# %%
hook_point = "blocks.10.hook_resid_post.hook_sae_acts_post"
clean_residual = clean_cache[hook_point]
corr_residual = corrupted_cache[hook_point]
corr_grad_residual = corrupted_grad_cache[hook_point]
residual_attr_final = einops.reduce(
    corr_grad_residual * (clean_residual - corr_residual),
    "batch pos n_features -> pos n_features",
    "sum",
)
residual_attr_final.shape

# %%

torch.topk(corr_grad_residual, 10).values





# %%

test_prompt(samples[0][-1], "Traceback", model)

# %%

# [range(logits_clean.size(0)), end_positions-1]

# %%
# Compute clean baseline
with torch.no_grad():
    logits_clean = model(samples[0])
print(f"Clean Baseline is 1: {err_metric_denoising(logits_clean).item():.4f}")
del logits_clean
gc.collect()
torch.cuda.empty_cache()

# %%

# Compute corrupted baseline
with torch.no_grad():
    logits_corr = model(samples[1])
print(f"Corrupted Baseline is 0: {err_metric_denoising(logits_corr).item():.4f}")
del logits_corr
gc.collect()
torch.cuda.empty_cache()

# %%

with torch.no_grad():
    sae.use_error_term = False
    model.add_sae(sae)
    logits_clean = model(samples[0])
    model.reset_saes()
print(f"Clean Baseline with sae, no error: {type_error_patch_metric(logits_clean, end_positions).item():.4f}")
del logits_clean
gc.collect()
torch.cuda.empty_cache()

# %%
# Compute corrupted baseline
with torch.no_grad():
    sae.use_error_term = False
    model.add_sae(sae)
    logits_corr = model(samples[1])
    model.reset_saes()
print(f"Corrupted Baseline is 0: {type_error_patch_metric(logits_corr, end_positions).item():.4f}")
del logits_corr
gc.collect()
torch.cuda.empty_cache()



# %% attribution 

from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint
# from torchtyping import TensorType as TT

def get_cache_fwd_and_bwd(model: HookedSAETransformer, saes: list[SAE], input, metric, error_term: bool = True, retain_graph: bool = True):
    """
    Get forward and backward caches for a model, including hooks for 'hook_sae_acts_post' and 'hook_sae_error'.
    """
    # Filters to identify relevant hooks
    filter_sae_acts = lambda name: "hook_sae_acts_post" in name
    filter_sae_error = lambda name: "hook_sae_error" in name

    # This hook function will store activations in the appropriate cache
    cache_dict = {"fwd": {}, "bwd": {}}

    def cache_hook(act, hook, dir: Literal["fwd", "bwd"]):
        cache_dict[dir][hook.name] = act.detach()

    with model.saes(saes=saes, use_error_term=error_term):
        # Adding hooks for both 'hook_sae_acts_post' and 'hook_sae_error'
        with model.hooks(
            fwd_hooks=[
                (filter_sae_acts, partial(cache_hook, dir="fwd")),
                (filter_sae_error, partial(cache_hook, dir="fwd"))
            ],
            bwd_hooks=[
                (filter_sae_acts, partial(cache_hook, dir="bwd")),
                (filter_sae_error, partial(cache_hook, dir="bwd"))
            ],
        ):
            # Forward pass fills the fwd cache, then backward pass fills the bwd cache (we don't care about metric value)
            _ = metric(model(input)).backward(retain_graph=retain_graph)
            # value.backward(retain_graph=retain_graph)

    return (
        0,
        ActivationCache(cache_dict["fwd"], model),
        ActivationCache(cache_dict["bwd"], model),
    )

# %%

clean_value, clean_cache, _ = get_cache_fwd_and_bwd(
    model, [sae], samples[0], err_metric_denoising
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
# print("Clean Gradients Cached:", len(clean_grad_cache))

# %%
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, [sae], samples[1], err_metric_denoising
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# %%

selected_pos  = {
    "s_start": [],
    "s_end": [],
    "i_start": [],
    "i_end": [],
    "end": []
}

for i in range(N):

    str_tokens_clean = model.to_str_tokens(samples[0][i])
    str_tokens_corr = model.to_str_tokens(samples[1][i])
    # Find the positions with differences
    diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]

    # Find positions of the first '("', the first '"' after '("', and the end position
    pos_open_paren_quote = str_tokens_clean.index('("')
    pos_first_quote_after_open = pos_open_paren_quote + str_tokens_clean[pos_open_paren_quote:].index('"') 
    pos_end = len(str_tokens_clean) - 1  # The last position

    # Return the positions with differences, and the positions found
    # print(diff_positions, pos_open_paren_quote, pos_first_quote_after_open, pos_end)
    # print(str_tokens_clean[pos_first_quote_after_open])
    selected_pos["s_start"].append(pos_open_paren_quote)
    selected_pos["s_end"].append(pos_first_quote_after_open)
    selected_pos["i_start"].append(diff_positions[0])
    selected_pos["i_end"].append(diff_positions[-1])
    selected_pos["end"].append(pos_end)

# %%
    
top_feats_per_pos = {}
K = 10
for idx, val in selected_pos.items():
    # print(idx, val)
    clean_residual_selected = sae_acts[torch.arange(sae_acts.shape[0]), val, :]
    corr_residual_selected = sae_acts_corr[torch.arange(sae_acts_corr.shape[0]), val, :]
    corr_grad_residual_selected = sae_grad_cache[torch.arange(sae_grad_cache.shape[0]), val, :]

    # Residual attribution calculation only for the selected positions
    residual_attr_final = einops.reduce(
        corr_grad_residual_selected * (clean_residual_selected - corr_residual_selected),
        "batch n_features -> n_features",
        "sum",
    )

    # Get the top N features for this layer
    top_feats = torch.topk(residual_attr_final, K)
    top_indices = top_feats.indices
    top_values = top_feats.values

    top_feats_per_pos[idx] = (top_indices, top_values)


# %%
top_feats_per_pos

# %%
sae_grad_cache

# %%

residual_attr_final = einops.reduce(
    sae_grad_cache * (sae_acts - sae_acts_corr),
    "batch pos n_features -> n_features",
    "sum",
)
residual_attr_final.shape

# %%

import torch
import einops
import requests
from bs4 import BeautifulSoup
import re
import json

# Function to get HTML for a specific feature
def get_dashboard_html(sae_release="gemma-2-9b", sae_id="10-gemmascope-res-16k", feature_idx=0):
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)

# Function to scrape the description for a feature
def scrape_description(layer, feature_idx):
    url = get_dashboard_html(sae_release="gemma-2-2b", sae_id=f"{layer}-gemmascope-res-16k", feature_idx=feature_idx)
    response = requests.get(url)
    
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        soup_str = str(soup)

        # Use regex to find the "description" field in the JSON structure
        all_descriptions = re.findall(r'description\\":\\"(.*?)",', soup_str)
        
        if all_descriptions:
            return all_descriptions[-1]  # Return the last description
        else:
            return "No description found."
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"

# %%
layer = 10
top_10_features_for_rel_pos = {}
interesting_keys = list(top_feats_per_pos.keys())[2:]
# print(interesting_keys)
for key in interesting_keys:
    print(f"Position: {key}")
    indices, values = top_feats_per_pos[key]
    top_10_features_for_rel_pos[key] = []
    for idx, val in zip(indices, values):
        print(f"Feature Index: {idx}, Value: {val}")
        description = scrape_description(layer, idx)
        print(description)
        top_10_features_for_rel_pos[key].append((idx.item(), val.item(), description))

# Save the results to a JSON file
with open('tasks/error_detection/type/out/layer10_top_10_features_for_rel_pos.json', 'w') as json_file:
    json.dump(top_10_features_for_rel_pos, json_file, indent=4)


# %%
    
# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_and_save_position_heatmap(data_idx: int, position_name: str, top_features: list, str_tokens: list, clean_cache: dict, corr_cache: dict):
    """
    Function to plot and save a heatmap for the top features in a given position.

    Args:
    - data_idx: Index of the specific prompt to visualize.
    - position_name: The name of the position being analyzed (e.g., d1, d2, END).
    - top_features: List of top features for the current position.
    - str_tokens: List of input tokens (strings) to display on the x-axis.
    - clean_cache: Dictionary of activations from the clean cache.
    - corr_cache: Dictionary of activations from the corrupted cache.

    Returns:
    - Saves the heatmap as a PNG file in the 'features' directory.
    """
    activations_matrix = []

    # Iterate over the top features (each feature has two rows: clean and corrupted)
    for feature in top_features:
        # layer = feature['layer']
        feature_idx = feature[0]

        # Get activations from clean and corrupted caches
        # layer_name = f'blocks.{layer}.hook_resid_post.hook_sae_acts_post'
        clean_activations = clean_cache[data_idx, :, feature_idx].cpu().detach().numpy()
        corr_activations = corr_cache[data_idx, :, feature_idx].cpu().detach().numpy()

        # Append clean and corrupted activations to the matrix
        activations_matrix.append(clean_activations)
        activations_matrix.append(corr_activations)

    # Convert the activations matrix to a numpy array for plotting
    activations_matrix = np.array(activations_matrix)

    # Create a heatmap for the current position
    plt.figure(figsize=(10, 6)) #len(activations_matrix) * 0.5))  # Adjust the figure size based on the number of rows

    plt.imshow(activations_matrix, aspect='auto', cmap='coolwarm')

    # Set x-axis to display the input tokens
    plt.xticks(ticks=np.arange(len(str_tokens)), labels=str_tokens, rotation=90)

    # Set y-axis labels to show clean and corrupted rows for each feature
    y_ticks = []
    for feature in top_features:
        feature_idx = feature[0]
        y_ticks.append(f'{feature_idx} (clean)')
        y_ticks.append(f'{feature_idx} (corr)')

    plt.yticks(ticks=np.arange(len(y_ticks)), labels=y_ticks)

    # Add a color bar to the side
    plt.colorbar(label='Activation Value')

    # Set axis labels and title
    plt.xlabel("Tokens")
    plt.ylabel("Features (Clean and Corrupted)")
    plt.title(f"Feature Activations for Position {position_name} (Top 5 Features)")
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.2)  

    # Create the 'features' directory if it doesn't exist
    if not os.path.exists("features"):
        os.makedirs("features")

    # Save the heatmap as a PNG file
    filename = f"tasks/error_detection/type/out/layer10_features/heatmap_position_{position_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()  # Close the plot to avoid display issues in loops

    print(f"Heatmap saved: {filename}")

# %%
# Example loop to generate heatmaps for each position
for position_name, top_10_features in top_10_features_for_rel_pos.items():
    print(f"Top 25 Features for Position: {position_name}")

    # # Take the top 5 features for the current position
    # top_5_features = top_25_features[:5]

    # Replace 'str_tokens' with the actual list of input tokens (strings)
    # Example: str_tokens = ["This", "is", "an", "example", "input", "sentence", "."]
    str_tokens = model.to_str_tokens(samples[0][0])  # Example input tokens

    # Generate and save the heatmap for the current position
    plot_and_save_position_heatmap(data_idx=0, position_name=position_name, top_features=top_10_features, str_tokens=str_tokens, clean_cache=sae_acts, corr_cache=sae_acts_corr)



# %%
sae_acts_corr.shape
# convert the 
# top_10_features_for_rel_pos
# val = 0.0007687670877203345
# val
# %%
values
# %%
