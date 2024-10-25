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
