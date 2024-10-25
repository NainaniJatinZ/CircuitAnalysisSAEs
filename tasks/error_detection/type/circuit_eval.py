# %%
import os 
import gc
import torch
os.chdir("/work/pi_jensen_umass_edu/jnainani_umass_edu/CircuitAnalysisSAEs")
import json
from sae_lens import SAE, HookedSAETransformer
from circ4latents import data_gen
from functools import partial
import einops

# Function to manage CUDA memory and clean up
def cleanup_cuda():
    torch.cuda.empty_cache()
    gc.collect()

# Load the config
with open("config.json", 'r') as file:
    config = json.load(file)
token = config.get('huggingface_token', None)
os.environ["HF_TOKEN"] = token

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

hf_cache = "/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub"
os.environ["HF_HOME"] = hf_cache

# Load the model
model = HookedSAETransformer.from_pretrained("google/gemma-2-9b", device=device, cache_dir=hf_cache)


# %%
# Generate samples
from tasks.error_detection.type.data import generate_samples

selected_templates = [2] # Adjust this as needed
N = 30
samples = generate_samples(selected_templates, N)

selected_pos = {"i_start": [], "i_end": [], "end": []}
for i in range(N):
    str_tokens_clean = model.to_str_tokens(samples[0][i])
    str_tokens_corr = model.to_str_tokens(samples[1][i])
    diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]
    pos_end = len(str_tokens_clean) - 1
    selected_pos["i_start"].append(diff_positions[0])
    selected_pos["i_end"].append(diff_positions[-1])
    selected_pos["end"].append(pos_end)

# Define logit diff function
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]

def logit_diff_fn(logits, selected_pos):
    err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
    no_err = logits[range(logits.size(0)), selected_pos, :][:, trip_arrow_token_id]
    return (err - no_err).mean()

# Disable gradients for all parameters
for param in model.parameters():
    param.requires_grad_(False)

clean_tokens = model.to_tokens(samples[0])
corr_tokens = model.to_tokens(samples[1])

# Compute logits for clean and corrupted samples
logits = model(clean_tokens)
clean_diff = logit_diff_fn(logits, selected_pos['end'])

logits = model(corr_tokens)
corr_diff = logit_diff_fn(logits, selected_pos['end'])

print(f"clean_diff: {clean_diff}")
print(f"corr_diff: {corr_diff}")

# Cleanup
del logits
cleanup_cuda()


# Define error type metric
def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = logit_diff_fn(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=selected_pos['end'])

# %%

# read the json file 
with open('tasks/error_detection/type/out/layers_top_10_features_for_rel_pos_positive.json') as f:
    results = json.load(f)

def flatten_and_sort_results(results, k=10):
    flattened_list = []
    
    # Iterate over layers and positions to extract values and feature indices
    for layer, positions in results.items():
        
        # Iterate over each position ('i_start', 'i_end', 'end')
        for pos_key, pos_val in positions.items():
            for i in range(k):  # We take the top k values and indices
                value = pos_val['top_values'][i]  # Attribution value
                feature_idx = pos_val['top_indices'][i]  # Feature index
                
                # Add a tuple (layer, feature_idx, value) to the flattened list
                flattened_list.append((int(layer), feature_idx, value))
    
    # Sort the flattened list by the attribution value (third element in the tuple)
    flattened_list.sort(key=lambda x: x[2], reverse=True)  # Sort by value (highest to lowest)
    
    return flattened_list

# Example usage
flattened_sorted_list = flatten_and_sort_results(results, k=10)

# Print the top 10 entries in the sorted list
for entry in flattened_sorted_list[:25]:
    print(entry)

# %%
import torch

def run_with_saes_filtered(tokens, filtered_ids, model, saes):
    # Ensure tokens are a torch.Tensor
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens).to(model.cfg.device)  # Move to the device of the model
    
    # Create a mask where True indicates positions to modify
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in filtered_ids:
        mask &= tokens != token_id

    # Expand the mask once, so it matches the shape [batch_size, seq_len, 1]
    mask_expanded = mask.unsqueeze(-1)  # Expand to allow broadcasting

    # Precompute mask-expansion before adding hooks to avoid repeated expansions inside the hook
    mask_expanded = mask_expanded.to(model.cfg.device)  # Move the mask to the same device as the model
    
    # For each SAE, add the appropriate hook
    for sae in saes:
        hook_point = sae.cfg.hook_name

        # Define the filtered hook function (optimized)
        def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
            # Apply the SAE only where mask_expanded is True
            modified_act = sae(act)  # Call the SAE once
            # In-place update where the mask is True
            act = torch.where(mask_expanded, modified_act, act)
            return act

        # Add the hook to the model
        model.add_hook(hook_point, filtered_hook, dir='fwd')

    with torch.no_grad():
        # Run the model with the tokens
        logits = model(tokens)

    # Reset the hooks after computation to free memory
    model.reset_hooks()

    return logits


# %%
layers= [7, 14, 21, 28, 35, 40]
saes = [SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical", device=device)[0] for layer in layers]

# %%
filtered_saes = [saes[0], saes[1], saes[2], saes[3], saes[5]]

# %%
# layer = 5
filtered_ids = [model.tokenizer.bos_token_id]
logits = run_with_saes_filtered(clean_tokens, filtered_ids, model, filtered_saes) #[saes[layer]])
clean_sae_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"clean_sae_diff: {clean_sae_diff}")
filtered_ids = [model.tokenizer.bos_token_id]
logits = run_with_saes_filtered(corr_tokens, filtered_ids, model,filtered_saes) # [saes[layer]])
corr_sae_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"corr_sae_diff: {corr_sae_diff}")
del logits
cleanup_cuda()

# %%
filtered_ids = [model.tokenizer.bos_token_id]
logits = run_with_saes_filtered(corr_tokens, [trip_arrow_token_id], model, saes)
corr_sae_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"corr_sae_diff: {corr_sae_diff}")
del logits
cleanup_cuda()

# %%

filtered_ids = [model.tokenizer.bos_token_id]
logits = run_with_saes_filtered(clean_tokens, [trip_arrow_token_id], model, saes)
clean_sae_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"clean_sae_diff: {clean_sae_diff}")
del logits
cleanup_cuda()
# %%
logits = model.run_with_saes(corr_tokens, saes=saes)
print(logit_diff_fn(logits, selected_pos['end']))
del logits
cleanup_cuda()

# %%

import torch

def run_with_saes_filtered_cache(tokens, filtered_ids, model, saes):
    # Ensure tokens are a torch.Tensor
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens).to(model.cfg.device)  # Move to the device of the model

    # Create a mask where True indicates positions to modify
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in filtered_ids:
        mask &= tokens != token_id

    # Expand the mask once, so it matches the shape [batch_size, seq_len, 1]
    mask_expanded = mask.unsqueeze(-1)  # Expand to allow broadcasting
    mask_expanded = mask_expanded.to(model.cfg.device)  # Move the mask to the same device as the model
    
    # Dictionary to store the modified activations
    sae_outs = {}

    # For each SAE, add the appropriate hook
    for sae in saes:
        hook_point = sae.cfg.hook_name

        # Define the filtered hook function (optimized)
        def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
            # Apply the SAE only where mask_expanded is True
            enc_sae = sae.encode(act)  # Call the SAE once
            # Store the updated activation in the dictionary
            sae_outs[hook.name] = enc_sae.detach().cpu()  
            modified_act = sae.decode(enc_sae)  # Call the SAE once
            # In-place update where the mask is True
            updated_act = torch.where(mask_expanded, modified_act, act)
        
            return updated_act

        # Add the hook to the model
        model.add_hook(hook_point, filtered_hook, dir='fwd')

    # Run the model with the tokens (no gradients needed)
    with torch.no_grad():
        logits = model(tokens)

    # Reset the hooks after computation to free memory
    model.reset_hooks()

    return logits, sae_outs  # Return logits and the updated activations

# print(f"clean_sae_diff: {clean_sae_diff}")
filtered_ids = [model.tokenizer.bos_token_id]
logits, clean_sae_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, filtered_saes) # [saes[layer]])
clean_sae_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"clean_sae_diff: {clean_sae_diff}")
del logits
cleanup_cuda()
# %%

clean_sae_cache.keys()
# %%

clean_sae_cache['blocks.7.hook_resid_post'].shape

# %%

# Initialize the dictionary to store the feature indices by layer
dict_feats = {}

# Iterate through the data list and populate the dictionary
for layer, feature_idx, _ in flattened_sorted_list[:25][:25]:
    if layer == 35:  # Skip layer 35
        continue
    
    # Create the key for this layer
    key = f"blocks.{layer}.hook_resid_post"
    
    # Add the feature index to the corresponding key in the dictionary
    if key not in dict_feats:
        dict_feats[key] = []
    
    dict_feats[key].append(feature_idx)

# Output the resulting dictionary
print(dict_feats)


# %%

import torch

def run_with_saes_filtered_patch(tokens, filtered_ids, model, saes, cache, dict_feats):
    # Ensure tokens are a torch.Tensor
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens).to(model.cfg.device)  # Move to the device of the model

    # Create a mask where True indicates positions to modify
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in filtered_ids:
        mask &= tokens != token_id

    # Expand the mask once, so it matches the shape [batch_size, seq_len, 1]
    mask_expanded = mask.unsqueeze(-1)  # Expand to allow broadcasting
    mask_expanded = mask_expanded.to(model.cfg.device)  # Move the mask to the same device as the model
    
    # For each SAE, add the appropriate hook
    for sae in saes:
        hook_point = sae.cfg.hook_name

        # Define the filtered hook function (optimized)
        def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
            # Apply the SAE only where mask_expanded is True
            enc_sae = sae.encode(act)  # Call the SAE once
            
            # If the current layer is in the cache and has specific feature indices to patch
            if hook.name in cache and hook.name in dict_feats:
                cached_enc_sae = cache[hook.name]  # Get cached activations from the cache
                feature_indices = dict_feats[hook.name]  # Get the feature indices to patch

                # Patch the specific feature indices in enc_sae with the ones from the cache
                for feature_idx in feature_indices:
                    enc_sae[:, :, feature_idx] = cached_enc_sae[:, :, feature_idx]

            # After patching, decode the modified enc_sae
            modified_act = sae.decode(enc_sae)

            # In-place update where the mask is True
            updated_act = torch.where(mask_expanded, modified_act, act)

            return updated_act

        # Add the hook to the model
        model.add_hook(hook_point, filtered_hook, dir='fwd')

    # Run the model with the tokens (no gradients needed)
    with torch.no_grad():
        logits = model(tokens)

    # Reset the hooks after computation to free memory
    model.reset_hooks()

    return logits  # Return only the logits

# %%

logits = run_with_saes_filtered_patch(corr_tokens, filtered_ids, model, saes, clean_sae_cache, dict_feats)
patched_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"patched_diff: {patched_diff}")
del logits
cleanup_cuda()

# %%
dict_feats


# %%

# save the dict_feats to a json
with open('tasks/error_detection/type/out/dict_feats.json', 'w') as f:
    json.dump(dict_feats, f)
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
    url = get_dashboard_html(sae_release="gemma-2-9b", sae_id=f"{layer}-gemmascope-res-16k", feature_idx=feature_idx)
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

circuit_with_description = {} 
for key, val in dict_feats.items():
    # get the layer out from the key 
    layer = int(key.split('.')[1])
    print("layer",layer)
    circuit_with_description[layer] = {}
    for feat_idx in val:
        print(feat_idx)
        desc = scrape_description(layer, feat_idx)
        print(desc)
        html_link = get_dashboard_html(sae_id=f"{layer}-gemmascope-res-16k", feature_idx=feat_idx)
        # get the value of the layer, feature idx from the flattened_sorted_list
        feat_val = 0
        for layer_, feature_idx_, value in flattened_sorted_list[:25]:
            if layer_ == layer and feature_idx_ == feat_idx:
                feat_val = value
                break
        print(feat_val)
        circuit_with_description[layer][feat_idx] = {"description": desc, "html_link": html_link, "value": feat_val}
        print("\n")
        
circuit_with_description

# %%
flattened_sorted_list[:25]
# %%

# save the circuit_with_description to a json
with open('tasks/error_detection/type/out/circuit_with_description.json', 'w') as f:
    json.dump(circuit_with_description, f)


# %%
_, corr_sae_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes) # [saes[layer]])
corr_sae_cache.keys()

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_and_save_layer_heatmap(layer_name: str, feature_indices: list, str_tokens_clean: list, str_tokens_corr: list, clean_cache: dict, corr_cache: dict):
    """
    Function to plot and save a heatmap for the averaged firing patterns across the dataset for a specific layer.

    Args:
    - layer_name: Name of the layer being analyzed (e.g., 'blocks.7.hook_resid_post').
    - feature_indices: List of feature indices to analyze.
    - str_tokens_clean: List of input tokens (strings) from the clean data.
    - str_tokens_corr: List of input tokens (strings) from the corrupted data.
    - clean_cache: Dictionary of activations from the clean cache.
    - corr_cache: Dictionary of activations from the corrupted cache.

    Returns:
    - Saves the heatmap as a PNG file in the 'features' directory.
    """
    activations_matrix = []

    # Iterate over the feature indices (one row for clean, one row for corrupted)
    for feature_idx in feature_indices:
        # Get activations from clean and corrupted caches and average across the dataset
        clean_activations_avg = clean_cache[:, :, feature_idx].mean(dim=0).cpu().detach().numpy()
        corr_activations_avg = corr_cache[:, :, feature_idx].mean(dim=0).cpu().detach().numpy()

        # Append averaged clean and corrupted activations to the matrix
        activations_matrix.append(clean_activations_avg)
        activations_matrix.append(corr_activations_avg)

    # Convert the activations matrix to a numpy array for plotting
    activations_matrix = np.array(activations_matrix)

    # Create a heatmap for the current layer
    plt.figure(figsize=(10, 6))  # Adjust the figure size based on the number of rows

    plt.imshow(activations_matrix, aspect='auto', cmap='coolwarm')

    # Create combined labels for x-axis by stacking clean and corrupted tokens from the first prompt
    combined_tokens = [f"{clean_token} | {corr_token}" for clean_token, corr_token in zip(str_tokens_clean, str_tokens_corr)]

    # Set x-axis to display the combined input tokens
    plt.xticks(ticks=np.arange(len(combined_tokens)), labels=combined_tokens, rotation=90)

    # Set y-axis labels to show clean and corrupted rows for each feature index
    y_ticks = []
    for feature_idx in feature_indices:
        y_ticks.append(f'{feature_idx} (clean)')
        y_ticks.append(f'{feature_idx} (corr)')

    plt.yticks(ticks=np.arange(len(y_ticks)), labels=y_ticks)

    # Add horizontal lines to separate clean and corrupted rows
    for i in range(1, len(y_ticks), 2):  # Add line after every two rows
        plt.axhline(i + 0.5, color='black', linewidth=1)

    # Add a color bar to the side
    plt.colorbar(label='Activation Value')

    # Set axis labels and title
    plt.xlabel("Tokens (Clean / Corrupted)")
    plt.ylabel("Features (Clean and Corrupted)")
    plt.title(f"Average Feature Activations for Layer {layer_name}")
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.3)  # Adjusted bottom for longer x-labels

    # Create the 'features' directory if it doesn't exist
    if not os.path.exists("features"):
        os.makedirs("features")

    # Save the heatmap as a PNG file
    filename = f"tasks/error_detection/type/out/circuit_features/heatmap_{layer_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()  # Close the plot to avoid display issues in loops

    print(f"Heatmap saved: {filename}")


# Example loop to generate heatmaps for each layer in dict_feats
def generate_layer_heatmaps(dict_feats, clean_cache, corr_cache, str_tokens_clean, str_tokens_corr):
    for layer_name, feature_indices in dict_feats.items():
        print(f"Generating heatmap for Layer: {layer_name}")

        # Generate and save the heatmap for the current layer
        plot_and_save_layer_heatmap(
            layer_name=layer_name,
            feature_indices=feature_indices,
            str_tokens_clean=str_tokens_clean,
            str_tokens_corr=str_tokens_corr,
            clean_cache=clean_cache[layer_name],  # Access the cache for this specific layer
            corr_cache=corr_cache[layer_name]     # Access the corrupted cache for this specific layer
        )


# Assuming dict_feats, clean_cache, corr_cache, and str_tokens are defined
# Example input tokens from the first prompt
str_tokens_clean = model.to_str_tokens(samples[0][0])  # Clean input tokens from the first prompt
str_tokens_corr = model.to_str_tokens(samples[1][0])  # Corrupted input tokens from the first prompt

# Generate heatmaps for each layer in dict_feats
generate_layer_heatmaps(dict_feats, clean_sae_cache, corr_sae_cache, str_tokens_clean, str_tokens_corr)

# %%

clean_sae_cache.keys()

# %%
