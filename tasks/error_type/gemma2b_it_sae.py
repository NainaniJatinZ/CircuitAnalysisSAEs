# %% imports 

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
import os 
from sae_lens import SAE, HookedSAETransformer
import error_data
sys.path.append("../../utils/")
import plot

with open("../../config.json", 'r') as file:
    config = json.load(file)
    token = config.get('huggingface_token', None)
os.environ["HF_TOKEN"] = token
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %%
# !nvidia-smi

# %% model loading 

model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device = device)

# %% Loading data 

clean_prompts, corr_prompts, clean_tokens, corr_tokens, clean_end_positions, corr_end_positions = error_data.create_dataset(N=20, template_numbers=[1], tokenizer=model.tokenizer)
print("Clean Prompts:", clean_prompts[0])
print("Corrupted Prompts:", corr_prompts[0])
print("Clean Tokens:", clean_tokens[0])
print("Corrupted Tokens:", corr_tokens[0])
print("Clean End Positions:", clean_end_positions)
print("Corrupted End Positions:", corr_end_positions)

# %% helper functions
syntax_token_id = model.tokenizer.encode(" Syntax", add_special_tokens=False)[0]
type_token_id = model.tokenizer.encode(" TypeError", add_special_tokens=False)[0]

print(f"Syntax token id: {syntax_token_id}, Type token id: {type_token_id}")

def logit_diff_error_type(logits, end_positions, err1_tok =type_token_id, err2_tok = syntax_token_id):
    err1_logits = logits[range(logits.size(0)), end_positions, :][:, err1_tok]
    err2_logits = logits[range(logits.size(0)), end_positions, :][:, err2_tok]
    logit_diff = (err1_logits - err2_logits).mean()
    return logit_diff

# %%

clean_logits = model(clean_tokens)
corr_logits = model(corr_tokens)
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
print(f"Clean Baseline is 1: {err_metric_denoising(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {err_metric_denoising(corr_logits).item():.4f}")

# %% 
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# TODO: Make this nicer.
df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
df[df['model']=='gemma-2-2b'] # Each row is a "release" which has multiple SAEs which may have different configs / match different hook points in a model. 
import re
from collections import defaultdict

sae_keys = list(df.loc['gemma-scope-2b-pt-res']['saes_map'].keys())
# Dictionary to store the closest string for each layer
closest_strings = {}

# Regular expression to extract the layer number and l0 value
pattern = re.compile(r'layer_(\d+)/width_16k/average_l0_(\d+)')

# Organize strings by layer
layer_dict = defaultdict(list)

for s in sae_keys:
    match = pattern.search(s)
    if match:
        layer = int(match.group(1))
        l0_value = int(match.group(2))
        layer_dict[layer].append((s, l0_value))

# Find the string with l0 value closest to 100 for each layer
for layer, items in layer_dict.items():
    closest_string = min(items, key=lambda x: abs(x[1] - 100))
    closest_strings[layer] = closest_string[0]

# Output the closest string for each layer
for layer in sorted(closest_strings):
    print(f"Layer {layer}: {closest_strings[layer]}")


# %% loading the saes 
layers = [3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25]
saes = [
    SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id=closest_strings[layer],
        device=str(device),
    )[0]
    for layer in tqdm(layers)
]

# %%
sae_layer = -3
error_term = False
clean_logits = model.run_with_saes(clean_tokens, saes=saes[sae_layer], use_error_term=error_term)
corr_logits = model.run_with_saes(corr_tokens, saes=saes[sae_layer], use_error_term=error_term)
clean_logit_diff = logit_diff_error_type(clean_logits, clean_end_positions)
corr_logit_diff = logit_diff_error_type(corr_logits, clean_end_positions)
print(f"Clean logit diff with sae error term {error_term}: {clean_logit_diff}")
print(f"Corrupted logit diff with sae error term {error_term}: {corr_logit_diff}")

error_term = True
clean_logits = model.run_with_saes(clean_tokens, saes=saes[sae_layer], use_error_term=error_term)
corr_logits = model.run_with_saes(corr_tokens, saes=saes[sae_layer], use_error_term=error_term)
clean_logit_diff = logit_diff_error_type(clean_logits, clean_end_positions)
corr_logit_diff = logit_diff_error_type(corr_logits, clean_end_positions)
print(f"Clean logit diff with sae error term {error_term}: {clean_logit_diff}")
print(f"Corrupted logit diff with sae error term {error_term}: {corr_logit_diff}")
# %%

from transformer_lens.utils import test_prompt
print(clean_prompts[0])
test_prompt(clean_prompts[0], " Syntax", model)

# prompt_dash = """print("score: " + 63)
# # When this code is executed, Python will raise a"""
# test_prompt(prompt_dash, " Syntax" ,model)

# %%
use_error_term = True
saes[0]
# %% Attribution 
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
            value = metric(model(input))
            value.backward(retain_graph=retain_graph)

    return (
        value,
        ActivationCache(cache_dict["fwd"], model),
        ActivationCache(cache_dict["bwd"], model),
    )
# model.set_use_attn_in(True)
# model.set_use_attn_result(True)
# model.set_use_hook_mlp_in(True)
# model.set_use_split_qkv_input(True)
# def get_cache_fwd_and_bwd_sae(model, sae, tokens, metric, use_error_term=True):
#     model.reset_hooks(including_permanent=True)
#     model.reset_saes()
#     sae.reset_hooks()
#     cache = {}
#     model.add_sae(sae)
#     sae.use_error_term = use_error_term
#     filter_sae_acts = lambda name: "hook_sae_acts_post"
#     filter_sae_error = lambda name: "hook_sae_error"
    
#     def forward_cache_hook(act, hook):
#         cache[hook.name] = act.detach()

#     model.add_hook(filter_sae_acts, forward_cache_hook, "fwd")
#     model.add_hook(filter_sae_error, forward_cache_hook, "fwd")

#     grad_cache = {}

#     def backward_cache_hook(act, hook):
#         grad_cache[hook.name] = act.detach()

#     model.add_hook(filter_sae_acts, backward_cache_hook, "bwd")
#     model.add_hook(filter_sae_error, backward_cache_hook, "bwd")

#     value = metric(model(tokens))
#     value.backward()
#     model.reset_hooks()
#     return (
#         value.item(),
#         ActivationCache(cache, model),
#         ActivationCache(grad_cache, model),
#     )

# %%

clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, saes, clean_tokens, err_metric_denoising
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))

# %%
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, saes, corr_tokens, err_metric_denoising
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))


# %%

import torch
import einops

# Assume end_positions is a list or tensor containing the position for each batch
# Example: end_positions = [2, 5, 1, 3]  # One end position for each batch
index_layer = 25
# Convert end_positions to a tensor if it's not already one
end_positions = torch.tensor(clean_end_positions)

# Fetch the residuals and errors for this layer
resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
error_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_error'

clean_residual = clean_cache[resid_point]
corr_residual = corrupted_cache[resid_point]
corr_grad_residual = corrupted_grad_cache[resid_point]

# Use advanced indexing to select the appropriate position for each batch
# This selects the values corresponding to the end_positions in the 'pos' dimension for each batch
clean_residual_selected = clean_residual[torch.arange(clean_residual.shape[0]), end_positions, :]
corr_residual_selected = corr_residual[torch.arange(corr_residual.shape[0]), end_positions, :]
corr_grad_residual_selected = corr_grad_residual[torch.arange(corr_grad_residual.shape[0]), end_positions, :]

# Residual attribution calculation only for the selected positions
residual_attr_final = einops.reduce(
    corr_grad_residual_selected * (clean_residual_selected - corr_residual_selected),
    "batch n_features -> n_features",
    "sum",
)

print(residual_attr_final.shape)  # This will now be (n_features,)

# %%

import torch
import einops
import requests
from bs4 import BeautifulSoup
import re
import json

# Function to get HTML for a specific feature
def get_dashboard_html(sae_release="gemma-2-2b", sae_id="8-gemmascope-res-16k", feature_idx=0):
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

# Initialize an empty list to store (layer, feature index, value) tuples
top_features = []

# Define the layers to process
layers = [3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25]

# Iterate through each layer and calculate residuals
for index_layer in layers:
    # Fetch the residuals and errors for this layer
    resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
    error_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_error'

    clean_residual = clean_cache[resid_point]
    corr_residual = corrupted_cache[resid_point]
    corr_grad_residual = corrupted_grad_cache[resid_point]

    # Use advanced indexing to select the appropriate position for each batch
    clean_residual_selected = clean_residual[torch.arange(clean_residual.shape[0]), end_positions, :]
    corr_residual_selected = corr_residual[torch.arange(corr_residual.shape[0]), end_positions, :]
    corr_grad_residual_selected = corr_grad_residual[torch.arange(corr_grad_residual.shape[0]), end_positions, :]

    # Residual attribution calculation only for the selected positions
    residual_attr_final = einops.reduce(
        corr_grad_residual_selected * (clean_residual_selected - corr_residual_selected),
        "batch n_features -> n_features",
        "sum",
    )

    # Get the top 20 features for this layer
    top_feats = torch.topk(residual_attr_final, 20)

    top_indices = top_feats.indices
    top_values = top_feats.values

    # Store the (layer, feature index, value) tuples
    for i in range(20):
        feature_idx = top_indices[i].item()  # Get the feature index
        value = top_values[i].item()         # Get the feature value
        top_features.append((index_layer, feature_idx, value))

# Sort all features by value and select the top 15 overall
top_features.sort(key=lambda x: x[2], reverse=True)
top_15_features = top_features[:15]

# Initialize a list to store the top 15 features with their descriptions
top_15_with_descriptions = []

# Scrape the descriptions for the top 15 features
for layer, feature_idx, value in top_15_features:
    description = scrape_description(layer, feature_idx)
    
    top_15_with_descriptions.append({
        "layer": layer,
        "feature_idx": feature_idx,
        "value": value,
        "description": description
    })

# Save the results to a JSON file
with open('top_15_features_with_descriptions.json', 'w') as json_file:
    json.dump(top_15_with_descriptions, json_file, indent=4)

print("Results saved to top_15_features_with_descriptions.json")


# %%
layer = 25
feature_idx = 3985
url = get_dashboard_html(sae_release="gemma-2-2b", sae_id=f"{layer}-gemmascope-res-16k", feature_idx=feature_idx)
response = requests.get(url)

if response.status_code == 200:
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    soup_str = str(soup)
    print(soup_str)
    # Use regex to find the "description" field in the JSON structure
    # all_descriptions = re.findall(r'description\\":\\"(.*?)",', soup_str)


# %%

import torch
import einops

# Initialize an empty list to store (layer, feature index, value) tuples
top_50_tuples = []

# Define the layers to process
layers = [3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25]

for index_layer in layers:
    # Fetch the residuals and errors for this layer
    resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
    error_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_error'

    clean_residual = clean_cache[resid_point]
    corr_residual = corrupted_cache[resid_point]
    corr_grad_residual = corrupted_grad_cache[resid_point]

    # Residual attribution calculation
    residual_attr_final = einops.reduce(
        corr_grad_residual * (clean_residual - corr_residual),
        "batch pos n_features -> pos n_features",
        "sum",
    )

    # Sum over residuals to get feature importances
    summed_tensor = torch.sum(residual_attr_final, dim=0)

    # Get the top 50 features (indices and values) for this layer
    top_feats = torch.topk(summed_tensor, 50)

    top_indices = top_feats.indices
    top_values = top_feats.values

    # Store the (layer, feature index, value) tuples
    for i in range(50):
        top_50_tuples.append((index_layer, top_indices[i].item(), top_values[i].item()))

# Sort the tuples by the value in descending order
top_50_tuples.sort(key=lambda x: x[2], reverse=True)

# Optionally, print the results
for layer, feature_idx, value in top_50_tuples[:50]:
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Value: {value}")
# %%
    



# %%

import requests
from bs4 import BeautifulSoup
import re
import json

# Assume top_50_tuples is already defined as a list of tuples: (layer, feature index, value)
# Example format of top_50_tuples: [(3, 1234, 0.789), (4, 5678, 0.654), ...]

# Function to get HTML for a specific feature
def get_dashboard_html(sae_release="gemma-2-2b", sae_id="8-gemmascope-res-16k", feature_idx=0):
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

# Initialize a list to store the (layer, feature index, value, description) tuples
top_50_with_descriptions = []

# Iterate through the top_50_tuples
for layer, feature_idx, value in top_50_tuples:
    # Scrape the description for this feature
    description = scrape_description(layer, feature_idx)
    
    # Append the tuple with description to the list
    top_50_with_descriptions.append({
        "layer": layer,
        "feature_idx": feature_idx,
        "value": value,
        "description": description
    })
    if value < 0.02:
        break

# Save the results to a JSON file
with open('top_50_features_with_descriptions.json', 'w') as json_file:
    json.dump(top_50_with_descriptions, json_file, indent=4)

print("Results saved to top_50_features_with_descriptions.json")

# %%

import torch
import einops
import requests
from bs4 import BeautifulSoup
import re
import json

# Function to get HTML for a specific feature
def get_dashboard_html(sae_release="gemma-2-2b", sae_id="8-gemmascope-res-16k", feature_idx=0):
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

# Initialize an empty list to store (layer, feature index, value, description) tuples
top_50_with_descriptions = []

# Define the layers to process
layers = [3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25]

for index_layer in layers:
    # Fetch the residuals and errors for this layer
    resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
    error_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_error'

    clean_residual = clean_cache[resid_point]
    corr_residual = corrupted_cache[resid_point]
    corr_grad_residual = corrupted_grad_cache[resid_point]

    # Residual attribution calculation
    residual_attr_final = einops.reduce(
        corr_grad_residual * (clean_residual - corr_residual),
        "batch pos n_features -> pos n_features",
        "sum",
    )

    # Sum over residuals to get feature importances
    summed_tensor = torch.sum(residual_attr_final, dim=0)

    # Get the top 50 features (indices and values) for this layer
    top_feats = torch.topk(summed_tensor, 50)

    top_indices = top_feats.indices
    top_values = top_feats.values

    # Store the (layer, feature index, value) tuples and scrape descriptions
    for i in range(50):
        feature_idx = top_indices[i].item()  # Get the feature index
        value = top_values[i].item()         # Get the feature value

        # Scrape the description for this feature
        description = scrape_description(index_layer, feature_idx)
        
        # Append the tuple with description to the list
        top_50_with_descriptions.append({
            "layer": index_layer,
            "feature_idx": feature_idx,
            "value": value,
            "description": description
        })

# Sort the tuples by the value in descending order (optional if needed)
top_50_with_descriptions.sort(key=lambda x: x["value"], reverse=True)

# Save the results to a JSON file
with open('top_50_features_with_descriptions.json', 'w') as json_file:
    json.dump(top_50_with_descriptions, json_file, indent=4)

print("Results saved to top_50_features_with_descriptions.json")




# %%

clean_cache.keys()




# %% Layer 3 analysis 

index_layer = 3
resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
error_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_error'
clean_cache[resid_point].shape, clean_cache[error_point].shape



# %% residual attribution
clean_residual = clean_cache[resid_point]
corr_residual = corrupted_cache[resid_point]
corr_grad_residual = corrupted_grad_cache[resid_point]
residual_attr_final = einops.reduce(
    corr_grad_residual * (clean_residual - corr_residual),
    "batch pos n_features -> pos n_features",
    "sum",
)
residual_attr_final.shape

# %% error attribution
clean_error = clean_cache[error_point]
corr_error = corrupted_cache[error_point]
corr_grad_error = corrupted_grad_cache[error_point]
error_attr_final = einops.reduce(
    corr_grad_error * (clean_error - corr_error),
    "batch pos n_features -> pos",
    "sum",
)
error_attr_final.shape

# %%

# %%
summed_tensor = torch.sum(residual_attr_final, dim=0)

# Find the top 50 elements along the second dimension (size 16384)
top_feats = torch.topk(summed_tensor, 50)

# Get the indices of the top 50 elements
top_indices = top_feats.indices

# Optionally, you can also access the values with
top_values = top_feats.values

# Print the results
print("Top 50 feature indices:", top_indices)
print("Top 50 feature values:", top_values)

# %%

from IPython.display import IFrame
html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(sae_release = "gemma-2-2b", sae_id="8-gemmascope-res-16k", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)

html = get_dashboard_html(sae_release = "gemma-2-2b", sae_id=f"{index_layer}-gemmascope-res-16k", feature_idx=top_indices[0].item())
display(IFrame(html, width=1200, height=300))

# %%

html = get_dashboard_html(sae_release = "gemma-2-2b", sae_id=f"{index_layer}-gemmascope-res-16k", feature_idx=top_indices[1].item())
display(IFrame(html, width=1200, height=300))

# %%


import requests
from bs4 import BeautifulSoup

# Construct the URL
def get_dashboard_html(sae_release = "gemma-2-2b", sae_id="8-gemmascope-res-16k", feature_idx=0):
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)

# URL for the dashboard
url = get_dashboard_html(sae_release = "gemma-2-2b", sae_id=f"{index_layer}-gemmascope-res-16k", feature_idx=top_indices[0].item())

# Fetch the page content
response = requests.get(url)

# If the page was fetched successfully
if response.status_code == 200:
    html = response.text

    # Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')
    # print(soup)
    soup_str = str(soup)

    # Use a regular expression to find the "description" field in the JSON structure
    all_descriptions = re.findall(r'description\\":\\"(.*?)",', soup_str)

    if all_descriptions:
        # Get the last description
        last_description = all_descriptions[-1]

        # Print the last description
        print("Last Description:", last_description)
    else:
        print("No descriptions found.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

# %%
    
# print(soup_str)
re.findall(r'description\\":\\"(.*?)",', soup_str)


# %%
residual_attr_final[0, top_feats.indices]



# %% overall iteration

import torch
import einops
import requests
from bs4 import BeautifulSoup
import re
import json

# Function to get HTML for a specific feature
def get_dashboard_html(sae_release="gemma-2-2b", sae_id="8-gemmascope-res-16k", feature_idx=0):
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
# Initialize a dictionary to store results for each layer
results = {}

# Loop through the index layers 3 to 6
for index_layer in layers:
    # Fetch the residuals and errors for this layer
    resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
    error_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_error'

    clean_residual = clean_cache[resid_point]
    corr_residual = corrupted_cache[resid_point]
    corr_grad_residual = corrupted_grad_cache[resid_point]

    # Residual attribution calculation
    residual_attr_final = einops.reduce(
        corr_grad_residual * (clean_residual - corr_residual),
        "batch pos n_features -> pos n_features",
        "sum",
    )

    # Sum over residuals to find the top 50 features
    summed_tensor = torch.sum(residual_attr_final, dim=0)
    top_feats = torch.topk(summed_tensor, 10)  # Top 10 features
    top_indices = top_feats.indices

    # Prepare the data structure for this layer
    layer_data = {
        "layer": index_layer,
        "top_features": []
    }

    # Loop through the top 10 features and scrape the description
    for i in range(10):
        feature_idx = top_indices[i].item()  # Convert tensor to integer
        description = scrape_description(index_layer, feature_idx)
        
        # Add the feature index and its description to the layer data
        layer_data["top_features"].append({
            "feature_idx": feature_idx,
            "description": description
        })

    # Add this layer's data to the results dictionary
    results[f"layer_{index_layer}"] = layer_data

# Save the results to a JSON file
with open('layer_features_descriptions.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to layer_features_descriptions.json")




# %%
import matplotlib.pyplot as plt
import networkx as nx

# Initialize a graph
G = nx.Graph()

# Positioning of nodes for layout
pos = {}
y_pos = 0  # Initial vertical position for the first layer
layer_spacing = 5  # Vertical spacing between layers
node_spacing = 2   # Horizontal spacing between nodes in a layer

# Loop over the layers and their top features
for layer_key, layer_data in results.items():
    layer_index = layer_data['layer']
    features = layer_data['top_features'][:5]  # Take only the first 5 features
    
    x_pos = 0  # Horizontal position for nodes in the current layer

    for feature in features:
        feature_idx = feature['feature_idx']
        description = feature['description']

        # Add the feature index as the node, and its description as the label
        G.add_node(f"{layer_key}_feat_{feature_idx}", label=f"F{feature_idx}: {description}")

        # Set the position of the node in the layout
        pos[f"{layer_key}_feat_{feature_idx}"] = (x_pos, y_pos)

        x_pos += node_spacing  # Move horizontally for the next node
    
    y_pos -= layer_spacing  # Move vertically for the next layer

# Create the layout with nodes positioned at (x_pos, y_pos)
fig, ax = plt.subplots(figsize=(10, 8))

# Draw the nodes with their positions
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=8, ax=ax)

# Extract labels and set them
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

plt.title('Top 5 Features per Layer')
plt.show()



# %%
clean_end_positions

# %%
# clean_residual = sae_clean_cache[hook_point]
# corr_residual = sae_corrupted_cache[hook_point]
# corr_grad_residual = sae_corrupted_grad_cache[hook_point]
# residual_attr_final = einops.reduce(
#     corr_grad_residual * (clean_residual - corr_residual),
#     "batch pos n_features -> pos n_features",
#     "sum",
# )
# residual_attr_final.shape



# %%
