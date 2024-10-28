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


# Disable gradients for all parameters
for param in model.parameters():
   param.requires_grad_(False)


clean_tokens = model.to_tokens(samples[0])
corr_tokens = model.to_tokens(samples[1])

# %%

with open('tasks/error_detection/type/out/dict_feats.json') as f:
   dict_feats = json.load(f)
print(dict_feats.keys())

# %%


# read the json from out


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
       if value > 0.08:
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

# %%
combined_dict['blocks.40.hook_resid_post']
# %%
combined_dict['blocks.28.hook_resid_post']
# %%
combined_dict['blocks.21.hook_resid_post']
# %%


# Convert lists to sets
set1 = set(combined_dict['blocks.28.hook_resid_post'])
set2 = set(dict_feats['blocks.28.hook_resid_post'])

# Intersection (common elements)
intersection = list(set1 & set2)
print("Intersection:", intersection)

# Union (all unique elements from both lists)
union = list(set1 | set2)
print("Union:", union)

# Unique elements (elements only in one of the lists)
unique_elements = list(set1 - set2)
print("Unique Elements:", unique_elements)

# %%
parent_feat = ["L21_1408", "L28_1800", "L28_2102", "L40_11839", "L40_9447", "L14_14967", "L21_8255"]

feat = "L40_9447"
with open(f'tasks/error_detection/type/out/second_latents/top_20_results_{feat}.json') as f:
   results = json.load(f)
results
# %%
# Convert lists to sets
set1 = set(combined_dict['blocks.21.hook_resid_post'])
set2 = set(dict_feats['blocks.21.hook_resid_post'])
# Unique elements (elements only in one of the lists)
unique_elements = list(set1 - set2)
print("Unique Elements:", unique_elements)
# %%

# %% latents for L28.2102
  
def latent_patch_metric(cache, layer_ind=28, lat_ind=2102):
   # layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
   result = cache[f'blocks.{layer_ind}.hook_resid_post'][:, 1:, lat_ind].sum()
   # print(result.requires_grad)
   return result


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



# Define error type metric
def _latent_metric(cache, clean_logit_diff, corr_logit_diff, layer_ind, lat_ind):
   patched_logit_diff = latent_patch_metric(cache, layer_ind, lat_ind)
   return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)


# latent_metric_denoising = partial(_latent_metric, clean_logit_diff=latent_act_clean, corr_logit_diff=latent_act_corrupted) #, layer_ind=40, lat_ind=9447)


# %%

target_latents = [(28, 9487), (28, 13620), (28, 15574), (28, 12670), (28, 7692), (28, 14974), (28, 1001), (28, 6500)]

filtered_ids = [model.tokenizer.bos_token_id]
for layer, latent_idx in target_latents:
        # Define unique identifier for this latent
      save_string = f"L{layer}_{latent_idx}"

      # Calculate and store activations for both clean and corrupted caches
      _, clean_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
      latent_act_clean = latent_patch_metric(clean_cache, layer_ind=layer, lat_ind=latent_idx)
      
      _, corr_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)
      latent_act_corrupted = latent_patch_metric(corr_cache, layer_ind=layer, lat_ind=latent_idx)
      print(f"Results for {save_string}")
      print(f"Clean: {latent_act_clean}")
      print(f"Corrupted: {latent_act_corrupted}")
      print("\n")


# %%


# Disable gradients for all parameters
for param in model.parameters():
   param.requires_grad_(False)


def get_cache_fwd_and_bwd_fml(model, tokens, metric, saes, filtered_ids, target_layer):
   model.reset_hooks()
   # cache = {}
   # grad_cache = {}
   sae_outs = {}
   sae_grad_outs = {}
   value_dict = {}
   if not isinstance(tokens, torch.Tensor):
       tokens = torch.tensor(tokens).to(model.cfg.device)
   mask = torch.ones_like(tokens, dtype=torch.bool)
   for token_id in filtered_ids:
       mask &= tokens != token_id


   # Expand the mask once, so it matches the shape [batch_size, seq_len, 1]
   mask_expanded = mask.unsqueeze(-1)  # Expand to allow broadcasting
   mask_expanded = mask_expanded.to(model.cfg.device)  # Move the mask to the same device as the model
   for sae in saes:
       if sae.cfg.hook_name == target_layer:
           target_sae = sae
           break


   # Dictionary to store the modified activations


   # SAE forward
   for sae in saes:
       hook_point = sae.cfg.hook_name


       # Define the filtered hook function (optimized)
       def filtered_fwd_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
           act.requires_grad_(True)
           # Apply the SAE only where mask_expanded is True
           enc_sae = sae.encode(act)  # Call the SAE once
           # Store the updated activation in the dictionary
           sae_outs[hook.name] = enc_sae.detach().cpu() 
           modified_act = sae.decode(enc_sae)  # Call the SAE once
           # In-place update where the mask is True
           updated_act = torch.where(mask_expanded, modified_act, act)
      
           return updated_act


       # Add the hook to the model
       model.add_hook(hook_point, filtered_fwd_hook, dir='fwd')


   # SAE backward till target sae
   for sae in saes:
       if sae.cfg.hook_name == target_layer:
           break
       hook_point = sae.cfg.hook_name


       # Define the filtered hook function (optimized)
       def filtered_bwd_hook(grad, hook, sae=sae):
           grad.requires_grad_(True)
           # Apply the SAE only where mask_expanded is True
           # enc_sae = sae.encode(act)  # Call the SAE once
           # Store the updated activation in the dictionary
           sae_grad_outs[hook.name] = torch.einsum('bij,kj->bik', grad, sae.W_dec).detach().cpu()


       # Add the hook to the model
       model.add_hook(hook_point, filtered_bwd_hook, dir='bwd')
  
   def custom_metric_hook(act, hook):
       enc_sae = target_sae.encode(act)
       value = metric({target_sae.cfg.hook_name:enc_sae})
       value_dict['value'] = value
       value.backward(retain_graph=True)      
       # return act
  
   model.add_hook(target_layer, custom_metric_hook, 'fwd')


   logits = model(tokens)


   # Reset the hooks after computation to free memory
   model.reset_hooks()


   return logits, sae_outs, sae_grad_outs, value_dict




# %%


# lat_candidates = [(21, 1408), (28, 1800), (28, 2102), (40, 11839), (40, 9447), (14, 14967), (21, 8255)]


# %%

import torch
import json
import einops
import gc

# Define your list of target latents as (layer, latent_idx) tuples
# target_latents = [(28, 2102), (21, 8255), (40, 9447)]  # Example list of tuples

target_latents = [(28, 9487), (28, 13620), (28, 15574), (28, 12670), (28, 7692), (28, 14974), (28, 1001), (28, 6500)]
filtered_ids = [model.tokenizer.bos_token_id]
K = 20  # Number of top entries to retrieve
model.reset_hooks()
def save_top_latents_per_target(target_latents):
    results = {}  # Store results for each target latent
    for layer, latent_idx in target_latents:
        # Define unique identifier for this latent
        save_string = f"L{layer}_{latent_idx}"

        # Calculate and store activations for both clean and corrupted caches
        _, clean_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
        latent_act_clean = latent_patch_metric(clean_cache, layer_ind=layer, lat_ind=latent_idx)
        
        _, corr_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)
        latent_act_corrupted = latent_patch_metric(corr_cache, layer_ind=layer, lat_ind=latent_idx)

        # Partial metric function for this specific target latent
        latent_metric_denoising = partial(
            _latent_metric,
            clean_logit_diff=latent_act_clean,
            corr_logit_diff=latent_act_corrupted,
            layer_ind=layer,
            lat_ind=latent_idx
        )
        
        # Collect forward and backward caches for current target latent
        _, _, corr_grad_cache, _ = get_cache_fwd_and_bwd_fml(
            model, corr_tokens, latent_metric_denoising, saes, filtered_ids, f'blocks.{layer}.hook_resid_post'
        )

        # Initialize sub-dictionaries for each target latent and layer
        results[save_string] = {"positive": {}, "absolute": {}}

        # Process each key in the gradient cache for this target latent
        for key in corr_grad_cache.keys():
            # Compute resi_attr
            resi_attr = einops.reduce(
                corr_grad_cache[key] * (clean_cache[key] - corr_cache[key]),
                'batch pos n_features -> n_features',
                'sum'
            )

            # Save resi_attr tensor for current key
            torch.save(resi_attr, f'tasks/error_detection/type/out/third_latents_28/{save_string}_{key}.pt')

            # Compute both top K values with and without absolute value sorting
            top_values, top_indices = resi_attr.topk(K)
            top_abs_values, top_abs_indices = resi_attr.abs().topk(K)

            # Store the top K positive and absolute values for each key
            results[save_string]["positive"][key] = [
                {"feature_idx": int(idx), "value": float(value)} for idx, value in zip(top_indices, top_values)
            ]
            results[save_string]["absolute"][key] = [
                {"feature_idx": int(idx), "value": float(resi_attr[idx])} for idx in top_abs_indices
            ]

        # Reset caches and empty CUDA memory for the next iteration
        del clean_cache, corr_cache, corr_grad_cache
        torch.cuda.empty_cache()

    # Save results to JSON
    with open('tasks/error_detection/type/out/third_latents_28/top_latents_results.json', 'w') as f:
        json.dump(results, f)

# Run the function to save top latents for each target
save_top_latents_per_target(target_latents)




# %%

corr_grad_cache.keys()
# %%

# Convert lists to sets
set1 = set(combined_dict['blocks.21.hook_resid_post'])
set2 = set(dict_feats['blocks.21.hook_resid_post'])

# elements in set1 but not in set2
unique_elements = list(set1 - set2)
print("Unique Elements in 21:", unique_elements)
# %%


# [5478, 4490, 3377, 8626, 13842, 9427, 12987, 4541]
target_latents = [(21, 5478), (21, 4490), (21, 3377), (21, 8626), (21, 13842), (21, 9427), (21, 12987), (21, 4541)]

save_dir = "third_latents_21"

filtered_ids = [model.tokenizer.bos_token_id]
K = 20  # Number of top entries to retrieve
model.reset_hooks()
def save_top_latents_per_target(target_latents, save_dir):
    results = {}  # Store results for each target latent
    for layer, latent_idx in target_latents:
        # Define unique identifier for this latent
        save_string = f"L{layer}_{latent_idx}"

        # Calculate and store activations for both clean and corrupted caches
        _, clean_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
        latent_act_clean = latent_patch_metric(clean_cache, layer_ind=layer, lat_ind=latent_idx)
        
        _, corr_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)
        latent_act_corrupted = latent_patch_metric(corr_cache, layer_ind=layer, lat_ind=latent_idx)

        # Partial metric function for this specific target latent
        latent_metric_denoising = partial(
            _latent_metric,
            clean_logit_diff=latent_act_clean,
            corr_logit_diff=latent_act_corrupted,
            layer_ind=layer,
            lat_ind=latent_idx
        )
        
        # Collect forward and backward caches for current target latent
        _, _, corr_grad_cache, _ = get_cache_fwd_and_bwd_fml(
            model, corr_tokens, latent_metric_denoising, saes, filtered_ids, f'blocks.{layer}.hook_resid_post'
        )

        # Initialize sub-dictionaries for each target latent and layer
        results[save_string] = {"positive": {}, "absolute": {}}

        # Process each key in the gradient cache for this target latent
        for key in corr_grad_cache.keys():
            # Compute resi_attr
            resi_attr = einops.reduce(
                corr_grad_cache[key] * (clean_cache[key] - corr_cache[key]),
                'batch pos n_features -> n_features',
                'sum'
            )

            # Save resi_attr tensor for current key
            torch.save(resi_attr, f'tasks/error_detection/type/out/{save_dir}/{save_string}_{key}.pt')

            # Compute both top K values with and without absolute value sorting
            top_values, top_indices = resi_attr.topk(K)
            top_abs_values, top_abs_indices = resi_attr.abs().topk(K)

            # Store the top K positive and absolute values for each key
            results[save_string]["positive"][key] = [
                {"feature_idx": int(idx), "value": float(value)} for idx, value in zip(top_indices, top_values)
            ]
            results[save_string]["absolute"][key] = [
                {"feature_idx": int(idx), "value": float(resi_attr[idx])} for idx in top_abs_indices
            ]

        # Reset caches and empty CUDA memory for the next iteration
        del clean_cache, corr_cache, corr_grad_cache
        torch.cuda.empty_cache()

    # Save results to JSON
    with open(f'tasks/error_detection/type/out/{save_dir}/top_latents_results.json', 'w') as f:
        json.dump(results, f)

# Run the function to save top latents for each target
save_top_latents_per_target(target_latents, save_dir)



# %%
