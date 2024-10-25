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
N = 50
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

from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint
# from torchtyping import TensorType as TT

# def get_cache_fwd_and_bwd(
#     model,
#     tokens,
#     metric,
#     sae,
#     error_term: bool = True,
#     retain_graph: bool = True
# ):
#     # torch.set_grad_enabled(True)
#     model.reset_hooks()
#     # model.reset_saes()
#     cache = {}
#     grad_cache = {}
#     filter_base_acts = lambda name: sae.cfg.hook_name in name #"blocks.21.hook_resid_post" in name
#     # filter_sae_acts = lambda name: "hook_sae_acts_post" in name

#     def forward_cache_hook(act, hook):
#         act.requires_grad_(True)
#         # act.retain_graph()
#         cache[hook.name] = act.detach()

#     def backward_cache_hook(grad, hook):
#         grad.requires_grad_(True)
#         # grad.retain_graph()
#         grad_cache[hook.name] = grad.detach()

#     # sae.use_error_term = error_term
#     # model.add_sae(sae)
#     model.add_hook(filter_base_acts, forward_cache_hook, "fwd")
#     model.add_hook(filter_base_acts, backward_cache_hook, "bwd")
#     # logits = run_with_saes_filtered(tokens, [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id], model, [sae])
#     value = metric(model(tokens)) #logits)
#     value.backward() #retain_graph=retain_graph)

#     model.reset_hooks()
#     # model.reset_saes()
#     # torch.set_grad_enabled(False)
#     return (
#         value,
#         ActivationCache(cache, model),
#         ActivationCache(grad_cache, model),
#     )

# # %%

# # Function to process each layer efficiently
# def process_layer(layer):
#     sae, _, _ = SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical", device=device)
    
#     _, clean_cache, _ = get_cache_fwd_and_bwd(model, clean_tokens, err_metric_denoising, sae)
#     _, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, corr_tokens, err_metric_denoising, sae)

#     sae_acts = sae.encode(clean_cache[f'blocks.{layer}.hook_resid_post'][:, 1:, :])
#     sae_acts_corr = sae.encode(corrupted_cache[f'blocks.{layer}.hook_resid_post'][:, 1:, :])
    
#     sae_grad_cache = torch.einsum('bij,kj->bik', corrupted_grad_cache[f'blocks.{layer}.hook_resid_post'][:, 1:, :], sae.W_dec)

#     top_feats_per_pos = {}
#     K = 10
#     for idx, val in selected_pos.items():
#         clean_residual_selected = sae_acts[:, val[0]-1,:]
#         corr_residual_selected = sae_acts_corr[:, val[0]-1,:]
#         corr_grad_residual_selected = sae_grad_cache[:, val[0]-1,:]

#         residual_attr_final = einops.reduce(
#             corr_grad_residual_selected * (clean_residual_selected - corr_residual_selected),
#             "batch n_features -> n_features",
#             "sum",
#         )

#         abs_residual_attr_final = torch.abs(residual_attr_final)
#         top_feats = torch.topk(abs_residual_attr_final, K)
#         top_indices = top_feats.indices
#         top_values = residual_attr_final[top_indices]

#         top_feats_per_pos[idx] = (top_indices, top_values)

#     # Cleanup
#     del sae_acts, sae_acts_corr, sae_grad_cache, clean_cache, corrupted_cache, corrupted_grad_cache
#     cleanup_cuda()

#     return top_feats_per_pos

# %%

# # Run analysis across layers
# layers_to_analyze = [7, 14, 21, 28, 35, 40]
# results = {}

# for layer in layers_to_analyze:
#     print(f"Processing layer {layer}")
#     top_feats_per_pos = process_layer(layer)
#     results[layer] = top_feats_per_pos

# %%
import json

# Convert tensor data to lists for JSON saving
def convert_results_to_serializable(results):
    serializable_results = {}
    for layer, positions in results.items():
        serializable_results[layer] = {}
        for pos_key, (top_indices, top_values) in positions.items():
            # Convert tensors to lists and floats for JSON serialization
            serializable_results[layer][pos_key] = {
                "top_indices": top_indices.tolist(),
                "top_values": top_values.tolist()
            }
    return serializable_results

# %%
from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint

# Function to handle both positive, negative and absolute top-k selection
def select_top_k(features, k, mode="absolute"):
    if mode == "positive":
        values, indices = torch.topk(features, k)
    elif mode == "negative":
        values, indices = torch.topk(-features, k)  # Reverse for negative top-k
        values = -values  # Reverse the sign back for the output
    elif mode == "absolute":
        abs_values = torch.abs(features)
        top_k_abs = torch.topk(abs_values, k)
        indices = top_k_abs.indices
        values = features[indices]
    else:
        raise ValueError("Invalid mode for top-k selection. Choose from 'positive', 'negative', or 'absolute'.")
    
    return indices, values

# %%

# Updated function to cache activations and gradients for multiple layers
def get_cache_fwd_and_bwd_multiple_layers(
    model,
    tokens,
    metric,
    sae_list,  # List of SAE models for the layers to process
    layers,
    error_term=True,
    retain_graph=True
):
    model.reset_hooks()
    cache = {}
    grad_cache = {}
    
    # Lambda to match hooks for each layer in layers
    filter_base_acts = lambda name: any(f"blocks.{layer}.hook_resid_post" in name for layer in layers)

    def forward_cache_hook(act, hook):
        act.requires_grad_(True)
        cache[hook.name] = act.detach()

    def backward_cache_hook(grad, hook):
        grad.requires_grad_(True)
        grad_cache[hook.name] = grad.detach()

    # Add hooks for all layers
    model.add_hook(filter_base_acts, forward_cache_hook, "fwd")
    model.add_hook(filter_base_acts, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward(retain_graph=retain_graph)

    model.reset_hooks()
    return (
        value,
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model)
    )

# %%

# Function to process multiple layers at once
def process_layers(layers, sae_list, k=10, top_k_mode="absolute"):
    _, clean_cache, _ = get_cache_fwd_and_bwd_multiple_layers(model, clean_tokens, err_metric_denoising, sae_list, layers)
    _, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd_multiple_layers(model, corr_tokens, err_metric_denoising, sae_list, layers)

    top_feats_per_layer = {}
    
    for layer, sae in zip(layers, sae_list):
        layer_cache_name = f'blocks.{layer}.hook_resid_post'
        
        sae_acts = sae.encode(clean_cache[layer_cache_name][:, 1:, :])
        sae_acts_corr = sae.encode(corrupted_cache[layer_cache_name][:, 1:, :])
        
        sae_grad_cache = torch.einsum('bij,kj->bik', corrupted_grad_cache[layer_cache_name][:, 1:, :], sae.W_dec)

        top_feats_per_pos = {}
        for idx, val in selected_pos.items():
            clean_residual_selected = sae_acts[:, val[0] - 1, :]
            corr_residual_selected = sae_acts_corr[:, val[0] - 1, :]
            corr_grad_residual_selected = sae_grad_cache[:, val[0] - 1, :]

            residual_attr_final = einops.reduce(
                corr_grad_residual_selected * (clean_residual_selected - corr_residual_selected),
                "batch n_features -> n_features",
                "sum",
            )

            # Get the top K features using the specified mode
            top_indices, top_values = select_top_k(residual_attr_final, k, mode=top_k_mode)

            top_feats_per_pos[idx] = (top_indices, top_values)

        top_feats_per_layer[layer] = top_feats_per_pos

    # Cleanup
    del sae_acts, sae_acts_corr, sae_grad_cache, clean_cache, corrupted_cache, corrupted_grad_cache
    cleanup_cuda()

    return top_feats_per_layer

# %%

# Main function to run the analysis
def run_analysis(layers, k=10, top_k_mode="absolute"):
    sae_list = [SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical", device=device)[0] for layer in layers]

    # Process all layers at once and get results
    results = process_layers(layers, sae_list, k=k, top_k_mode=top_k_mode)
    
    # Convert the results to a serializable format and save to a JSON file
    serializable_results = convert_results_to_serializable(results)
    with open(f'tasks/error_detection/type/out/layers_top_{k}_features_for_rel_pos_{top_k_mode}.json', 'w') as json_file:
        json.dump(serializable_results, json_file, indent=4)

    print("Results successfully saved to JSON.")

# %%

# Run the analysis across layers
layers_to_analyze = [7, 14, 21, 28, 35, 40]

# Specify the value of k and the top_k_mode (absolute, positive, or negative)
run_analysis(layers_to_analyze, k=10, top_k_mode="positive")

# You can also run:
# run_analysis(layers_to_analyze, k=10, top_k_mode="positive")
# run_analysis(layers_to_analyze, k=10, top_k_mode="negative")

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read the json file 
with open('tasks/error_detection/type/out/layers_top_10_features_for_rel_pos_positive.json') as f:
    results = json.load(f)

# %% 
k = 3
for layer, positions in results.items():
    layer_values = []
    layer_annotations = []
    
    # Flatten all top features for each position (i_start, i_end, end)
    for pos_key, pos_val in positions.items():
        # print(pos_key)
        for i in range(k):  # We take the top k values and indices
            layer_values.append(pos_val['top_values'][i])  # Heatmap value (attribution value)
            layer_annotations.append(pos_val['top_indices'][i]) 
    print(layer_values)
    print(layer_annotations)
    break
# %%

# Prepare the data for heatmap
def prepare_heatmap_data(results, k=10):
    # We will collect values and feature indices in a structured way
    heatmap_values = []
    heatmap_annotations = []
    layer_labels = []
    
    # Flatten results to extract values and annotations
    for layer, positions in results.items():
        layer_values = []
        layer_annotations = []
        
        # Flatten all top features for each position (i_start, i_end, end)
        for pos_key, pos_val in positions.items():
        # print(pos_key)
            for i in range(k):  # We take the top k values and indices
                layer_values.append(pos_val['top_values'][i])  # Heatmap value (attribution value)
                layer_annotations.append(pos_val['top_indices'][i]) 

        heatmap_values.append(layer_values)
        heatmap_annotations.append(layer_annotations)
        layer_labels.append(f"Layer {layer}")
    
    return np.array(heatmap_values), np.array(heatmap_annotations), layer_labels

# Function to plot the heatmap
def plot_heatmap(heatmap_values, heatmap_annotations, layer_labels, k=10):
    # Prepare x-axis labels
    num_positions = heatmap_values.shape[1]
    x_labels = []
    pos_categories = ['i_start', 'i_end', 'end']
    
    for pos_cat in pos_categories:
        for i in range(k):
            x_labels.append(f"{pos_cat}{i + 1}")
    
    # Create the heatmap
    plt.figure(figsize=(15, len(layer_labels) * 1))  # Dynamic height based on layers
    
    # Plot heatmap using seaborn
    ax = sns.heatmap(
        heatmap_values, 
        annot=heatmap_annotations, 
        fmt='d',  # Annotation format is integer (feature index)
        cmap="coolwarm",  # Choose color map
        cbar=True, 
        xticklabels=x_labels, 
        yticklabels=layer_labels,
        annot_kws={"size": 8}  # Adjust font size of annotations
    )
    
    # Set labels
    ax.set_xlabel("Positions", fontsize=12)
    ax.set_ylabel("Layers", fontsize=12)
    plt.title("Heatmap of Top Feature Attributions with Feature Index Annotations", fontsize=14)
    
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming results is the dictionary from the previous code
k = 10  # Number of top features per position
heatmap_values, heatmap_annotations, layer_labels = prepare_heatmap_data(results, k=k)

# Plot the heatmap
plot_heatmap(heatmap_values, heatmap_annotations, layer_labels, k=k)

# %%

samples[0][0]
# %%

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
for entry in flattened_sorted_list[:10]:
    print(entry)

# %%
len(flattened_sorted_list)
# %%
ttl_val = 0
for entry in flattened_sorted_list[:40]:
    # print(entry)
    ttl_val += entry[2]
ttl_val
# %%

import matplotlib.pyplot as plt

def plot_attribution_curve(flattened_sorted_list):
    # Extract the values (third element of the tuple)
    values = [entry[2] for entry in flattened_sorted_list]
    
    # Generate the x-axis based on the number of values
    x = range(1, len(values) + 1)
    
    # Plot the curve
    plt.figure(figsize=(10, 6))
    plt.plot(x, values, marker='o', linestyle='-', color='b', label='Attribution Values')
    
    # Add labels and title
    plt.xlabel('Ranked Features (by Attribution Value)', fontsize=12)
    plt.ylabel('Attribution Value', fontsize=12)
    plt.title('Curve of Attribution Values (Highest to Lowest)', fontsize=14)
    
    # Add a legend
    plt.legend()
    
    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
flattened_sorted_list = flatten_and_sort_results(results, k=10)
plot_attribution_curve(flattened_sorted_list)

# %%
flattened_sorted_list[:25]
# %%
