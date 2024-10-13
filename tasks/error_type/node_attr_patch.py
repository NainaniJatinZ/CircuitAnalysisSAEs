# %% imports 
import os 
os.chdir("/home/jnainani_umass_edu/CircuitAnalysisSAEs/tasks/error_type")
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

# %% model loading 

model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device = device)

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
clean_prompts, corr_prompts, clean_tokens, corr_tokens, clean_end_positions, corr_end_positions, differing_positions = error_data.create_dataset(N=30, template_numbers=[1], tokenizer=model.tokenizer)
print("Clean Prompts:", clean_prompts[0])
print("Corrupted Prompts:", corr_prompts[0])
print("Clean Tokens:", clean_tokens[0])
print("Corrupted Tokens:", corr_tokens[0])
print("Clean End Positions:", clean_end_positions)
print("Corrupted End Positions:", corr_end_positions)
print("Differing Positions:", differing_positions)

# %%
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
import gc
del clean_logits, corr_logits
gc.collect()
# Empty the CUDA cache
torch.cuda.empty_cache()

# %%

clean_value, clean_cache, _ = get_cache_fwd_and_bwd(
    model, saes, clean_tokens, err_metric_denoising
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
# print("Clean Gradients Cached:", len(clean_grad_cache))

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

# Function to process and store the top features for a given position (d1, d2, or END)
def process_and_store_top_features(position_name, selected_positions, layers, clean_cache, corrupted_cache, corrupted_grad_cache, N=25):
    # Initialize an empty list to store (layer, feature index, value) tuples for the current position
    top_features_for_position = []

    # Iterate through each layer and calculate residuals
    for index_layer in layers:
        # Fetch the residuals and errors for this layer
        resid_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_acts_post'
        # error_point = f'blocks.{index_layer}.hook_resid_post.hook_sae_error'

        clean_residual = clean_cache[resid_point]
        corr_residual = corrupted_cache[resid_point]
        corr_grad_residual = corrupted_grad_cache[resid_point]

        # Use advanced indexing to select the appropriate position (d1, d2, or END) for each batch
        clean_residual_selected = clean_residual[torch.arange(clean_residual.shape[0]), selected_positions, :]
        corr_residual_selected = corr_residual[torch.arange(corr_residual.shape[0]), selected_positions, :]
        corr_grad_residual_selected = corr_grad_residual[torch.arange(corr_grad_residual.shape[0]), selected_positions, :]

        # Residual attribution calculation only for the selected positions
        residual_attr_final = einops.reduce(
            corr_grad_residual_selected * (clean_residual_selected - corr_residual_selected),
            "batch n_features -> n_features",
            "sum",
        )

        # Get the top N features for this layer
        top_feats = torch.topk(residual_attr_final, N)

        top_indices = top_feats.indices
        top_values = top_feats.values

        # Store the (layer, feature index, value) tuples
        for i in range(N):
            feature_idx = top_indices[i].item()  # Get the feature index
            value = top_values[i].item()         # Get the feature value
            top_features_for_position.append((index_layer, feature_idx, value))

    # Sort all features by value and select the top N overall
    top_features_for_position.sort(key=lambda x: x[2], reverse=True)
    top_n_features = top_features_for_position[:N]

    # Initialize a list to store the top N features with their descriptions
    top_n_with_descriptions = []

    # Scrape the descriptions for the top N features
    for layer, feature_idx, value in top_n_features:
        description = scrape_description(layer, feature_idx)
        print(f"Layer {layer}, Feature {feature_idx}, Value: {value}, Description: {description}")
        top_n_with_descriptions.append({
            "layer": layer,
            "feature_idx": feature_idx,
            "value": value,
            "description": description
        })

    return top_n_with_descriptions

# Main logic
layers = [3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25]
positions_to_evaluate = ["d1", "d2", "END"]

d1_positions = [pos[0] for pos in differing_positions]
d2_positions = [pos[1] for pos in differing_positions]

# Initialize a dictionary to store results for each position type
top_25_features_for_all_positions = {}

# Process for each position (d1, d2, and END)
for position_name, selected_positions in zip(positions_to_evaluate, [d1_positions, d2_positions, clean_end_positions]):
    top_25_features_for_position = process_and_store_top_features(position_name, selected_positions, layers, clean_cache, corrupted_cache, corrupted_grad_cache, N=25)
    top_25_features_for_all_positions[position_name] = top_25_features_for_position

# Save the results to a JSON file
with open('top_25_latents_direct_from_relevant_pos.json', 'w') as json_file:
    json.dump(top_25_features_for_all_positions, json_file, indent=4)

print("Results saved to top_25_latents_direct_from_relevant_pos.json")


# %% Analysis
from IPython.display import IFrame
for position_name, top_25_features in top_25_features_for_all_positions.items():
    print(f"Top 25 Features for Position: {position_name}")
    for feature in top_25_features[:5]:
        print(f"Layer {feature['layer']}, Feature {feature['feature_idx']}, Value: {feature['value']}")
        print(f"Description: {feature['description']}")
        # Display the html for the feature
        html = get_dashboard_html(sae_id=f"{feature['layer']}-gemmascope-res-16k", feature_idx=feature["feature_idx"])
        display(IFrame(html, width=1200, height=300))

# %%
        
model.to_str_tokens(clean_prompts[0])

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
        layer = feature['layer']
        feature_idx = feature['feature_idx']

        # Get activations from clean and corrupted caches
        layer_name = f'blocks.{layer}.hook_resid_post.hook_sae_acts_post'
        clean_activations = clean_cache[layer_name][data_idx, :, feature_idx].cpu().numpy()
        corr_activations = corr_cache[layer_name][data_idx, :, feature_idx].cpu().numpy()

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
        layer = feature['layer']
        feature_idx = feature['feature_idx']
        y_ticks.append(f'{layer}.{feature_idx} (clean)')
        y_ticks.append(f'{layer}.{feature_idx} (corr)')

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
    filename = f"features/heatmap_position_{position_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()  # Close the plot to avoid display issues in loops

    print(f"Heatmap saved: {filename}")

# Example loop to generate heatmaps for each position
for position_name, top_25_features in top_25_features_for_all_positions.items():
    print(f"Top 25 Features for Position: {position_name}")

    # Take the top 5 features for the current position
    top_5_features = top_25_features[:5]

    # Replace 'str_tokens' with the actual list of input tokens (strings)
    # Example: str_tokens = ["This", "is", "an", "example", "input", "sentence", "."]
    str_tokens = model.to_str_tokens(clean_prompts[0])  # Example input tokens

    # Generate and save the heatmap for the current position
    plot_and_save_position_heatmap(data_idx=0, position_name=position_name, top_features=top_5_features, str_tokens=str_tokens, clean_cache=clean_cache, corr_cache=corrupted_cache)

# %%
for position_name, top_25_features in top_25_features_for_all_positions.items():
    print(f"Top 25 Features for Position: {position_name}")

    # Take the top 5 features for the current position
    top_5_features = top_25_features[:5]
    for feature in top_5_features:
        print(f"{feature['layer']}.{feature['feature_idx']}")
# %%
