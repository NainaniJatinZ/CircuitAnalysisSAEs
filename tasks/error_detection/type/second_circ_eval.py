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


with open('tasks/error_detection/type/out/dict_feats.json') as f:
   dict_feats = json.load(f)
print(dict_feats.keys())
dict_feats['blocks.7.hook_resid_post']




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


# %%
parent_feat = ["L21_1408", "L28_1800", "L28_2102", "L40_11839", "L40_9447", "L14_14967", "L21_8255"]
smaller_feat = []
second_latents = {}
# Load the json with teh suffix from above

for feat in parent_feat:
   with open(f'tasks/error_detection/type/out/second_latents/top_20_results_{feat}.json') as f:
       results = json.load(f)
   print(f"Results for {feat} ###")
   for result in results:
       key = result['key']
       feature_idx = result['feature_idx']
       value = result['value']
       if key not in second_latents:
           second_latents[key] = []
       if abs(value) > 0.08:
           if feature_idx not in second_latents[key]:
               second_latents[key].append(feature_idx)
               print(f"Key: {key}, Feature Index: {feature_idx}, Value: {value}")

print(second_latents)

# %%
# Combine the dict_feats and second_latents
combined_dict = {}
for key in dict_feats.keys():
   if key in second_latents.keys():
       combined_dict[key] = dict_feats[key] + second_latents[key]
   else:
       combined_dict[key] = dict_feats[key]

print(combined_dict)

ttl_latents = 0
for key in combined_dict.keys():
   ttl_latents += len(combined_dict[key])
print(ttl_latents)


cleanup_cuda()
filtered_ids = [model.tokenizer.bos_token_id]
_, clean_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
_, corrupted_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)

# %%

corr_sae_cache_means = {layer: sae_cache.mean(dim=0) for layer, sae_cache in corrupted_cache.items()}


print(corrupted_cache['blocks.7.hook_resid_post'].shape)
print(corr_sae_cache_means['blocks.7.hook_resid_post'].shape)



# %%


def run_with_saes_mean_ablation(tokens, filtered_ids, model, saes, mean_acts, dict_feats):
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
           if '7' in hook.name or '14' in dict_feats:
               pass
           elif hook.name in mean_acts and hook.name in dict_feats:
               mean_enc_sae = mean_acts[hook.name]  # Get cached activations from the cache
               feature_indices = dict_feats[hook.name]  # Get the feature indices to patch


               # Patch the specific feature indices in enc_sae with the ones from the cache
               for feature_idx in range(sae.cfg.d_sae):
                   if feature_idx not in feature_indices:
                       enc_sae[:, :, feature_idx] = mean_enc_sae[:, feature_idx]


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

logits = run_with_saes_mean_ablation(clean_tokens, filtered_ids, model, saes, corr_sae_cache_means, combined_dict)


mean_logit = logit_diff_fn(logits, selected_pos['end'])
print(f"mean_logit: {mean_logit}")


# %%


def run_with_saes_zero_ablation(tokens, filtered_ids, model, saes, dict_feats):
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
           if hook.name in dict_feats: # or '28' in hook.name): # or '21' in hook.name): # or '14' in hook.name):
               # mean_enc_sae = mean_acts[hook.name]  # Get cached activations from the cache
               feature_indices = dict_feats[hook.name]  # Get the feature indices to patch


               # Patch the specific feature indices in enc_sae with the ones from the cache
               for feature_idx in range(sae.cfg.d_sae):
                   if feature_idx not in feature_indices:
                       # define the zero tensor
                       enc_sae[:, :, feature_idx] = torch.zeros_like(enc_sae[:, :, feature_idx])


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

logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats)


zero_logit = logit_diff_fn(logits, selected_pos['end'])
print(f"dict feats zero_logit: {zero_logit}")
# %%


logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, combined_dict)


zero_logit = logit_diff_fn(logits, selected_pos['end'])
print(f"comb feats zero_logit: {zero_logit}")
# %%

# Define the combinations of layers to ablate
layer_combinations = [
    [40], [28, 40], [21, 28, 40], [14, 21, 28, 40], [7, 14, 21, 28, 40],
    [7, 14, 21, 28], [7, 14, 21], [7, 14],  [7, 14, 40], [7, 21, 40], [7, 28, 40], [7, 40],
    [14, 40], [21, 40], [14, 21, 40], [14, 28, 40],
    [7, 14, 21, 40], [7, 14, 28, 40], [7, 21, 28, 40]
]

# Dictionary to hold results
results = {}

# Helper function to filter dictionary by layer indices
def filter_dict_by_layers(original_dict, layers):
    return {k: v for k, v in original_dict.items() if int(k.split('.')[1]) in layers}

# Iterate over each combination of layers
for layers in layer_combinations:
    # Filter dict_feats and combined_dict to only include the current combination of layers
    dict_feats_subset = filter_dict_by_layers(dict_feats, layers)
    combined_dict_subset = filter_dict_by_layers(combined_dict, layers)
    print(dict_feats_subset)
    # print(combined_dict_subset)
    # break
    # Run with dict_feats
    logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_subset)
    zero_logit_dict = logit_diff_fn(logits, selected_pos['end'])
    
    # Run with combined_dict
    logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, combined_dict_subset)
    zero_logit_comb = logit_diff_fn(logits, selected_pos['end'])
    
    # Store results
    layer_key = ', '.join(map(str, layers))
    results[layer_key] = (zero_logit_dict, zero_logit_comb)

    print(f"Layers {layer_key}: Dict Feats Zero Logit = {zero_logit_dict}, Comb Feats Zero Logit = {zero_logit_comb}")

# Results are stored in the `results` dictionary for further analysis or plotting

# Results are stored in the `results` dictionary
# %%

results
# %%

import matplotlib.pyplot as plt
import numpy as np
# Extract labels and values for plotting
labels = list(results.keys())
dict_feats_values = [values[0].detach().cpu().item() for values in results.values()]
comb_feats_values = [values[1].detach().cpu().item() for values in results.values()]

dict_feats_values

# %%
# Plotting the bar chart
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, dict_feats_values, width, label='1st latents')
bars2 = ax.bar(x + width/2, comb_feats_values, width, label='1st+2nd latents')

# Adding labels and title
ax.set_xlabel("Layer Combinations to Ablate")
ax.set_ylabel("Zero Ablation logit difference")
ax.set_title("Zero Ablation for Different Layer Combinations in First order latents and Second order latents")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()

plt.show()
# %%
