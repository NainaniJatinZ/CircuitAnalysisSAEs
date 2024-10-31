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
# cleanup_cuda()
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

layers= [7, 14, 21, 28, 40]
saes = [SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical", device=device)[0] for layer in layers]

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
# %%

def run_with_saes_zero_ablation(tokens, filtered_ids, model, saes, dict_feats):
   # Ensure tokens are a torch.Tensor
   if not isinstance(tokens, torch.Tensor):
       tokens = torch.tensor(tokens).to(model.cfg.device)  # Move to the device of the model
   mask = torch.ones_like(tokens, dtype=torch.bool)
   for token_id in filtered_ids:
       mask &= tokens != token_id
   mask_expanded = mask.unsqueeze(-1) 
   mask_expanded = mask_expanded.to(model.cfg.device) 
   for sae in saes:
       hook_point = sae.cfg.hook_name
       def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
           enc_sae = sae.encode(act)  
           if hook.name in dict_feats: 
               feature_indices = dict_feats[hook.name]  
               for feature_idx in range(sae.cfg.d_sae):
                   if feature_idx not in feature_indices:
                       enc_sae[:, :, feature_idx] = torch.zeros_like(enc_sae[:, :, feature_idx])
           modified_act = sae.decode(enc_sae)
           updated_act = torch.where(mask_expanded, modified_act, act)
           return updated_act

       model.add_hook(hook_point, filtered_hook, dir='fwd')
   with torch.no_grad():
       logits = model(tokens)
   model.reset_hooks()
   return logits 


# %%

# %%

# read the json file
with open('tasks/error_detection/type/out/layers_top_100_features_for_rel_pos_absolute.json') as f:
   results = json.load(f)

def flatten_and_sort_results_abs(results, target_layer, k=100):
    flattened_list = []
    target_layer_list = []
    for layer, positions in results.items():
        for pos_key, pos_val in positions.items():
            # if pos_key == 'end':
            #     continue
            for i in range(k):  
                value = pos_val['top_values'][i] 
                feature_idx = pos_val['top_indices'][i] 
                flattened_list.append((int(layer), feature_idx, value))
                if int(layer) == target_layer:
                    target_layer_list.append((int(layer), feature_idx, value))
    flattened_list.sort(key=lambda x: abs(x[2]), reverse=True)  
    target_layer_list.sort(key=lambda x: abs(x[2]), reverse=True)  

    return flattened_list, target_layer_list

def flatten_and_sort_results(results, target_layer, k=10):
    flattened_list = []
    target_layer_list = []
    for layer, positions in results.items():
        for pos_key, pos_val in positions.items():
            # if pos_key == 'end':
            #     continue
            for i in range(k):  
                value = pos_val['top_values'][i] 
                feature_idx = pos_val['top_indices'][i] 
                flattened_list.append((int(layer), feature_idx, value))
                if int(layer) == target_layer:
                    target_layer_list.append((int(layer), feature_idx, value))
    flattened_list.sort(key=lambda x: x[2], reverse=True)  
    target_layer_list.sort(key=lambda x: x[2], reverse=True)  

    return flattened_list, target_layer_list

# Example usage
_, l40 = flatten_and_sort_results_abs(results, 40, k=30)
_, l28 = flatten_and_sort_results_abs(results, 28, k=30)
_, l21 = flatten_and_sort_results_abs(results, 21, k=50)

# Print the top 10 entries in the sorted list
# for entry in flattened_sorted_list[:25]:
#    print(entry)

# Print the top 10 entries in the sorted list
for entry in l40[:25]:
   print(entry)

# %%
model.reset_hooks()

# %%
dict_feats_40 = {}
for layer, feature_idx, _ in l40[:30][:30]:
    key = f"blocks.{layer}.hook_resid_post"
    if key not in dict_feats_40:
        dict_feats_40[key] = []
    dict_feats_40[key].append(feature_idx)

filtered_ids = [model.tokenizer.bos_token_id]
l40_logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_40)
l40_ablation = logit_diff_fn(l40_logits, selected_pos["end"])
l40_ablation



# %%

# Step 1: Define the initial feature dictionary with the top 30 latents
dict_feats_40 = {}
model.reset_hooks()
for layer, feature_idx, _ in l40[:30]:  # Only top 30 latents
    key = f"blocks.{layer}.hook_resid_post"
    if key not in dict_feats_40:
        dict_feats_40[key] = []
    dict_feats_40[key].append(feature_idx)

# Calculate the original ablation value with all 30 latents
filtered_ids = [model.tokenizer.bos_token_id]
l40_logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_40)
original_ablation_value = logit_diff_fn(l40_logits, selected_pos["end"])
print(f"Original Ablation Value: {original_ablation_value}")
# Step 2: Create a list to store the change in ablation value for each removed latent
importance_drops = []

# Step 3: Iterate over each latent, remove it, and calculate the new ablation value
for layer, feature_idx, _ in l40[:30]:
    key = f"blocks.{layer}.hook_resid_post"
    
    # Temporarily remove the current feature_idx
    modified_feats = {k: v[:] for k, v in dict_feats_40.items()}  # Make a copy
    modified_feats[key].remove(feature_idx)
    model.reset_hooks()
    # Re-run the ablation without the current feature
    l40_logits_modified = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, modified_feats)
    modified_ablation_value = logit_diff_fn(l40_logits_modified, selected_pos["end"])
    
    # Calculate the drop in ablation value when this latent is removed
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Ablation Value: {modified_ablation_value}")
    drop_in_value = original_ablation_value - modified_ablation_value
    importance_drops.append((layer, feature_idx, drop_in_value))

# Sort the importance drops by the magnitude of the drop
importance_drops.sort(key=lambda x: x[2], reverse=True)

# Output the sorted list of importance drops
for layer, feature_idx, drop in importance_drops:
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Drop in Ablation Value: {drop}")

# %%
model.reset_hooks()

# Step 1: Calculate the original ablation values for both `clean_tokens` and `corr_tokens`
filtered_ids = [model.tokenizer.bos_token_id]

# Original ablation for clean_tokens
l40_logits_clean = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_40)
original_ablation_clean = logit_diff_fn(l40_logits_clean, selected_pos["end"])
print(f"Original Ablation Value (Clean Tokens): {original_ablation_clean}")

# Original ablation for corr_tokens
l40_logits_corr = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, dict_feats_40)
original_ablation_corr = logit_diff_fn(l40_logits_corr, selected_pos["end"])
print(f"Original Ablation Value (Corr Tokens): {original_ablation_corr}")

# Step 2: List to store the summed drop in ablation value for each removed latent across both inputs
importance_drops = []

# Step 3: Iterate over each latent, remove it, and calculate the new ablation values for both inputs
for layer, feature_idx, _ in l40[:30]:
    key = f"blocks.{layer}.hook_resid_post"
    
    # Temporarily remove the current feature_idx
    modified_feats = {k: v[:] for k, v in dict_feats_40.items()}  # Make a copy
    modified_feats[key].remove(feature_idx)
    
    # Re-run the ablation without the current feature for clean_tokens
    model.reset_hooks()
    l40_logits_clean_modified = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, modified_feats)
    modified_ablation_clean = logit_diff_fn(l40_logits_clean_modified, selected_pos["end"])
    drop_in_clean = original_ablation_clean - modified_ablation_clean
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Clean Ablation Value: {modified_ablation_clean}")

    # Re-run the ablation without the current feature for corr_tokens
    model.reset_hooks()
    l40_logits_corr_modified = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, modified_feats)
    modified_ablation_corr = logit_diff_fn(l40_logits_corr_modified, selected_pos["end"])
    drop_in_corr = original_ablation_corr - modified_ablation_corr
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Corr Ablation Value: {modified_ablation_corr}")
    
    # Sum the drops for both inputs and store
    total_drop = abs(drop_in_clean) + abs(drop_in_corr)
    importance_drops.append((layer, feature_idx, total_drop))

# Step 4: Sort the importance drops by the sum of the drops and identify features to remove
importance_drops.sort(key=lambda x: x[2])  # Sort by the total drop in ascending order

# Output sorted list of importance drops
for layer, feature_idx, total_drop in importance_drops:
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Total Drop in Ablation Value: {total_drop}")

# %%
# plot the importance_drops 

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(importance_drops, marker='o')
plt.xlabel("Feature Index")
plt.ylabel("Total Drop in Ablation Value")
plt.title("Total Drop in Ablation Value vs Feature Index")
plt.show()

# %%

# get a list of top features from importance drops, so reverse the list
top_features = importance_drops[::-1]
top_features[:30]
filtered_40 = {}
for layer, feature_idx, feat_val in top_features[:30]:
    key = f"blocks.{layer}.hook_resid_post"
    if feat_val > 0.2:
        if key not in filtered_40:
            filtered_40[key] = []
        filtered_40[key].append(feature_idx)
filtered_40

# %% 

filtered_ids = [model.tokenizer.bos_token_id]
fil_40_logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, filtered_40)
fil_40_ablation = logit_diff_fn(fil_40_logits, selected_pos["end"])
fil_40_ablation


# %%
dict_feats_28 = {}
for layer, feature_idx, _ in l28[:30][:30]:
    key = f"blocks.{layer}.hook_resid_post"
    if key not in dict_feats_28:
        dict_feats_28[key] = []
    dict_feats_28[key].append(feature_idx)

filtered_ids = [model.tokenizer.bos_token_id]
l28_logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, dict_feats_28)
l28_ablation = logit_diff_fn(l28_logits, selected_pos["end"])
l28_ablation

# %%

dict_feats_28_40 = dict_feats_28 | filtered_40

l28_40_logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_28_40)
l28_40_ablation = logit_diff_fn(l28_40_logits, selected_pos["end"])
l28_40_ablation




# %%
# %%
model.reset_hooks()

# Step 1: Calculate the original ablation values for both `clean_tokens` and `corr_tokens`
filtered_ids = [model.tokenizer.bos_token_id]
_, l28 = flatten_and_sort_results_abs(results, 28, k=30)
dict_feats_28 = {}
for layer, feature_idx, _ in l28[:30][:30]:
    key = f"blocks.{layer}.hook_resid_post"
    if key not in dict_feats_28:
        dict_feats_28[key] = []
    dict_feats_28[key].append(feature_idx)
# Original ablation for clean_tokens
l28_logits_clean = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_28)
original_ablation_clean = logit_diff_fn(l28_logits_clean, selected_pos["end"])
print(f"Original Ablation Value (Clean Tokens): {original_ablation_clean}")

# Original ablation for corr_tokens
l28_logits_corr = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, dict_feats_28)
original_ablation_corr = logit_diff_fn(l28_logits_corr, selected_pos["end"])
print(f"Original Ablation Value (Corr Tokens): {original_ablation_corr}")

# Step 2: List to store the summed drop in ablation value for each removed latent across both inputs
importance_drops_28 = []

# Step 3: Iterate over each latent, remove it, and calculate the new ablation values for both inputs
for layer, feature_idx, _ in l28[:30]:
    key = f"blocks.{layer}.hook_resid_post"
    
    # Temporarily remove the current feature_idx
    modified_feats = {k: v[:] for k, v in dict_feats_28.items()}  # Make a copy
    modified_feats[key].remove(feature_idx)
    
    # Re-run the ablation without the current feature for clean_tokens
    model.reset_hooks()
    l28_logits_clean_modified = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, modified_feats)
    modified_ablation_clean = logit_diff_fn(l28_logits_clean_modified, selected_pos["end"])
    drop_in_clean = original_ablation_clean - modified_ablation_clean
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Clean Ablation Value: {modified_ablation_clean}")

    # Re-run the ablation without the current feature for corr_tokens
    model.reset_hooks()
    l28_logits_corr_modified = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, modified_feats)
    modified_ablation_corr = logit_diff_fn(l28_logits_corr_modified, selected_pos["end"])
    drop_in_corr = original_ablation_corr - modified_ablation_corr
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Corr Ablation Value: {modified_ablation_corr}")
    
    # Sum the drops for both inputs and store
    total_drop = abs(drop_in_clean) + abs(drop_in_corr)
    importance_drops_28.append((layer, feature_idx, total_drop))

# Step 4: Sort the importance drops by the sum of the drops and identify features to remove
importance_drops_28.sort(key=lambda x: x[2])  # Sort by the total drop in ascending order

# Output sorted list of importance drops
for layer, feature_idx, total_drop in importance_drops_28:
    print(f"Layer: {layer}, Feature Index: {feature_idx}, Total Drop in Ablation Value: {total_drop}")

# %%

# get a list of top features from importance drops, so reverse the list
top_features = importance_drops_28[::-1]
top_features[:30]
filtered_28 = {}
for layer, feature_idx, feat_val in top_features[:30]:
    key = f"blocks.{layer}.hook_resid_post"
    if feat_val > 0.1:
        if key not in filtered_28:
            filtered_28[key] = []
        filtered_28[key].append(feature_idx)

filtered_28[key].append(2102)

# %% 

filtered_ids = [model.tokenizer.bos_token_id]
fil_28_logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, filtered_28)
fil_28_ablation = logit_diff_fn(fil_28_logits, selected_pos["end"])
fil_28_ablation

# %%
filtered_40


# %%

dict_feats_28_40 = filtered_28 | filtered_40

l28_40_logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, dict_feats_28_40)
l28_40_ablation = logit_diff_fn(l28_40_logits, selected_pos["end"])
l28_40_ablation



# %%
dict_feats_21 = {}
for layer, feature_idx, _ in l21[:50][:50]:
    key = f"blocks.{layer}.hook_resid_post"
    if key not in dict_feats_21:
        dict_feats_21[key] = []
    dict_feats_21[key].append(feature_idx)

filtered_ids = [model.tokenizer.bos_token_id]
l21_logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, dict_feats_21)
l21_ablation = logit_diff_fn(l21_logits, selected_pos["end"])
l21_ablation



# %%

_, l21 = flatten_and_sort_results_abs(results, 21, k=15)

# Initialize dictionary for features of top N latents
dict_feats_21 = {}
for layer, feature_idx, _ in l21[:100][:100]:  # Use only top 75 latents
    key = f"blocks.{layer}.hook_resid_post"
    if key not in dict_feats_21:
        dict_feats_21[key] = []
    dict_feats_21[key].append(feature_idx)

# List to store ablation values for different top N latents
ablation_values = []

# Iterate over range of top N latents
for N in range(1, 100, 10):  # Adjusting top latents from 1 to 75
    # Create sub-dictionary of dict_feats_40 for the top N latents
    dict_feats_N = {k: v[:N] for k, v in dict_feats_21.items() if v[:N]}
    
    # Run ablation with selected latents
    filtered_ids = [model.tokenizer.bos_token_id]
    logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_N)
    ablation_value = logit_diff_fn(logits, selected_pos["end"])
    
    # Append ablation value to list
    ablation_values.append(ablation_value.detach().cpu().numpy())

# Plotting Top N Latents vs Ablation Value
top_latents = list(range(1, 100, 10))  # Top N latents from 1 to 75
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(top_latents, ablation_values, marker='o')
plt.xlabel("Top N 1st order Latents in L28")
plt.ylabel("Ablation Value")
plt.title("Top N Latents in Layer 28 sae with highest positive attribution values vs Ablation Value")
plt.grid()
plt.show()

# %%
ablation_values

# %%

final_40 = {}
for layer, feature_idx, _ in l40[:35][:35]:  # Use only top 75 latents
    key = f"blocks.{layer}.hook_resid_post"
    if key not in final_40:
        final_40[key] = []
    final_40[key].append(feature_idx)
final_40

# %%



# Example usage
flattened_sorted_list, l28 = flatten_and_sort_results_abs(results, 28, k=50)

# Print the top 10 entries in the sorted list
# for entry in flattened_sorted_list[:25]:
#    print(entry)

# Print the top 10 entries in the sorted list
for entry in l28[:25]:
   print(entry)

# %%
dict_feats_40 = {}
for layer, feature_idx, _ in l40[:30][:30]:
    key = f"blocks.{layer}.hook_resid_post"
    if key not in dict_feats_40:
        dict_feats_40[key] = []
    dict_feats_40[key].append(feature_idx)
dict_feats_40
# %%
filtered_ids = [model.tokenizer.bos_token_id]
l40_logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_40)
l40_ablation = logit_diff_fn(l40_logits, selected_pos["end"])
l40_ablation
