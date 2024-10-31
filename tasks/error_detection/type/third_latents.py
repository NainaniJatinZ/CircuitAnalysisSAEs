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
for entry in flattened_sorted_list[:30]:
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
       if value > 0.07:
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

pos_latents_28 = [7692, 1001, 6500]

target_latents = [(21, 10795), (21, 14699), (21, 5478), (21, 4490), (21, 3377), (21, 8626), (21, 13842), (21, 9427), (21, 12987), (21, 4541)]

pos_latents_21 = [10795, 3377, 12987]

# 14727, 14967, 1289, 4718, 3959, 13884
target_latents = [(14, 14727), (14, 1289), (14, 4718), (14, 3959), (14, 13884)]

pos_latents_14 = [14727, 3959]

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
set1 = set(combined_dict['blocks.28.hook_resid_post'])
set2 = set(dict_feats['blocks.28.hook_resid_post'])

# elements in set1 but not in set2
unique_elements = list(set1 - set2)
print(set1)
print(set2)
print("Unique Elements in 14:", unique_elements)
# %%


# [5478, 4490, 3377, 8626, 13842, 9427, 12987, 4541]
# target_latents = [(21, 10795), (21, 14699), (21, 5478), (21, 4490), (21, 3377), (21, 8626), (21, 13842), (21, 9427), (21, 12987), (21, 4541)]

# 14727, 14967, 1289, 4718, 3959, 13884
target_latents = [(14, 14727), (14, 1289), (14, 4718), (14, 3959), (14, 13884)]

save_dir = "third_latents_14"

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

import json
import os

# Paths to the JSON files in their respective directories
json_paths = [
    "tasks/error_detection/type/out/third_latents_28/top_latents_results.json",
    "tasks/error_detection/type/out/third_latents_21/top_latents_results.json",
    "tasks/error_detection/type/out/third_latents_14/top_latents_results.json"
]

# Directory to save the filtered results
output_dir = "tasks/error_detection/type/out/third_latents_filtered/"
os.makedirs(output_dir, exist_ok=True)

# Function to filter latents based on threshold and fallback to top 3
def filter_latents(data, threshold=0.1, top_n=3):
   filtered_data = {}

   for target_latent, latent_data in data.items():
      filtered_data[target_latent] = []

      category = "positive"
      all_latents = []
      for key, latents in latent_data[category].items():
            all_latents.extend([(key, lat["feature_idx"], lat["value"]) for lat in latents])

      # Filter latents with values > threshold
      above_threshold = [lat for lat in all_latents if abs(lat[2]) > threshold]

      # If no values exceed the threshold, select the top N by absolute value
      if above_threshold:
         filtered_data[target_latent] = above_threshold
      else:
         # Sort by absolute value and take the top N
         top_latents = sorted(all_latents, key=lambda x: abs(x[2]), reverse=True)[:top_n]
         filtered_data[target_latent] = top_latents

   return filtered_data

# Process each JSON file and save filtered results
lats = [28, 21, 14]
i = 0
for json_path in json_paths:
   # Load data from JSON file
   with open(json_path, "r") as f:
      data = json.load(f)
   print(data)
   # Filter latents as per the requirements
   filtered_results = filter_latents(data)

   print(filtered_results)
   # Define output path and save the filtered results to a new JSON file
   output_path = os.path.join(output_dir, f"filtered_L{lats[i]}.json")
   i += 1
   with open(output_path, "w") as f:
      json.dump(filtered_results, f, indent=2)

   print(f"Filtered results saved to {output_path}")

# %%

json_paths = [
    "tasks/error_detection/type/out/third_latents_filtered/filtered_L14.json",
    "tasks/error_detection/type/out/third_latents_filtered/filtered_L21.json",
    "tasks/error_detection/type/out/third_latents_filtered/filtered_L28.json"
]

third_latents = {"blocks.7.hook_resid_post":[], "blocks.14.hook_resid_post":[], "blocks.21.hook_resid_post":[]}

pos_latents_28 = [7692, 1001, 6500]

pos_latents_21 = [10795, 3377, 12987]

pos_latents_14 = [14727, 3959]

for json_path in json_paths:
   # Load data from JSON file   
    with open(json_path, "r") as f:
        data = json.load(f)
    print(data)

    for target_latent, latents in data.items():
        target_layer = int(target_latent.split("_")[0][1:])
        target_latent = int(target_latent.split("_")[1])
        # if target_layer == 28:
        #     pos_latents = pos_latents_28
        # elif target_layer == 21:
        #     pos_latents = pos_latents_21
        # elif target_layer == 14:
        #     pos_latents = pos_latents_14
        
        # if target_latent not in pos_latents and target_layer != 28:
        #     print(f"Skippin Target Latent: {target_latent}")
        #     continue

        print(f"Target Latent: {target_latent}")
        for key, feature_idx, value in latents:
            # key is in the form of blocks.21.hook_resid_post, i want to get the layer out
            key_layer = int(key.split(".")[1])
            print(f"Key Layer: {key_layer}")

            if key_layer == 21: 
                if feature_idx not in third_latents[key]:
                    third_latents[key].append(feature_idx)
                    print(f"Key: {key}, Feature Index: {feature_idx}, Value: {value}")
            elif value > 0.1:
                if feature_idx not in third_latents[key]:
                    third_latents[key].append(feature_idx)
                    print(f"Key: {key}, Feature Index: {feature_idx}, Value: {value}")
        print("\n")

# %%

third_latents
# %%

# Save the third latents in third_latents_filtered

with open('tasks/error_detection/type/out/third_latents_filtered/third_latents_pos_onlu.json', 'w') as f:
   json.dump(third_latents, f)

# %% zero ablation
   

third_dict = {}
for key in combined_dict.keys():
   if key in third_latents.keys():
       third_dict[key] = combined_dict[key] + third_latents[key]
   else:
       third_dict[key] = combined_dict[key]
third_dict


# %%

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
traceback_token_id = model.tokenizer.encode("Syntax", add_special_tokens=False)[0]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]


def logit_diff_fn(logits, selected_pos):
   err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
   no_err = logits[range(logits.size(0)), selected_pos, :][:, trip_arrow_token_id]
   return (err - no_err).mean()

# %%

ttl_third = 0
for key in third_dict.keys():
    ttl_third += len(third_dict[key])
print(ttl_third)

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

third_dict['blocks.21.hook_resid_post']
combined_dict['blocks.21.hook_resid_post']

# %%

def filter_dict_by_layers(original_dict, layers):
    return {k: v for k, v in original_dict.items() if int(k.split('.')[1]) in layers}

layers = [21]
dict_feats_subset = filter_dict_by_layers(dict_feats, layers)
combined_dict_subset = filter_dict_by_layers(combined_dict, layers)
third_dict_subset = filter_dict_by_layers(third_dict, layers)

# Run with dict_feats
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_subset)
zero_logit_dict = logit_diff_fn(logits, selected_pos['end'])

# Run with combined_dict
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, combined_dict_subset)
zero_logit_comb = logit_diff_fn(logits, selected_pos['end'])

# Run with third_dict
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, third_dict_subset)
zero_logit_third = logit_diff_fn(logits, selected_pos['end'])

print(f"Zero Ablation logit difference with dict_feats: {zero_logit_dict}")
print(f"Zero Ablation logit difference with combined_dict: {zero_logit_comb}")
print(f"Zero Ablation logit difference with third_dict: {zero_logit_third}")



# %%

import matplotlib.pyplot as plt
import numpy as np

# Define the combinations of layers to ablate
layer_combinations = [
    [7], [14], [21], [28], [40], [7, 40], [28, 40], [21, 28], [14, 40], [21, 40], [7, 14, 21, 28, 40]
]

# Dictionary to hold results
results = {}
filtered_ids = [model.tokenizer.bos_token_id]
# Helper function to filter dictionary by layer indices
def filter_dict_by_layers(original_dict, layers):
    return {k: v for k, v in original_dict.items() if int(k.split('.')[1]) in layers}

# Iterate over each combination of layers
for layers in layer_combinations:
   # Filter dict_feats, combined_dict, and third_dict to only include the current combination of layers
   dict_feats_subset = filter_dict_by_layers(dict_feats, layers)
   combined_dict_subset = filter_dict_by_layers(combined_dict, layers)
   third_dict_subset = filter_dict_by_layers(third_dict, layers)

   # Run with dict_feats
   logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, dict_feats_subset)
   zero_logit_dict = logit_diff_fn(logits, selected_pos['end'])

   # Run with combined_dict
   logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, combined_dict_subset)
   zero_logit_comb = logit_diff_fn(logits, selected_pos['end'])

   # Run with third_dict
   logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, third_dict_subset)
   zero_logit_third = logit_diff_fn(logits, selected_pos['end'])

   # Store results
   layer_key = ', '.join(map(str, layers))
   results[layer_key] = (zero_logit_dict, zero_logit_comb, zero_logit_third)



# %%
   
# Extract labels and values for plotting
labels = list(results.keys())
dict_feats_values = [values[0].detach().cpu().item() for values in results.values()]
comb_feats_values = [values[1].detach().cpu().item() for values in results.values()]
third_dict_values = [values[2].detach().cpu().item() for values in results.values()]

# Plotting the bar chart
x = np.arange(len(labels))
width = 0.25  # Adjust width to accommodate three bars per group

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width, dict_feats_values, width, label='1st latents')
bars2 = ax.bar(x, comb_feats_values, width, label='1st+2nd latents')
bars3 = ax.bar(x + width, third_dict_values, width, label='1st+2nd+3rd latents')

# Adding labels and title
ax.set_xlabel("Layer Combinations to Ablate")
ax.set_ylabel("Zero Ablation logit difference")
ax.set_title("Zero Ablation for Different Layer Combinations in Latents")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()

plt.show()

# %%

third_dict['blocks.7.hook_resid_post']
# combined_dict['blocks.21.hook_resid_post']

# %%

sampled = {'blocks.21.hook_resid_post': [8255,
 14699,
 1408,
 10795,
 1408,
 8626,
 12987,
 14699,
 9427,
 3377,
 13842,
 9427,
 8626,
 4541,
 13842,
 3377,
 10532]}


samepled_v2 = {'blocks.21.hook_resid_post': [8255,
 14699,
 1408,
 10795,
 1408,
 8626,
 12987,
 14699,
 9427,
 3377,
 13842,
 9427,
 8626,
 4541,
 13842,
 3377,
 10532], 
  'blocks.28.hook_resid_post': [2102,
 1800,
 2102,
 4611,
 9419,
 7692,
 2102,
 9487,
 15574,
 13620,
 1001,
 14974,
 12670,
 6500]}

# Run with third_dict
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, sampled)
zero_logit_third = logit_diff_fn(logits, selected_pos['end'])

print(f"Zero Ablation logit difference 21: {zero_logit_third}")

# Run with third_dict
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, samepled_v2)
zero_logit_third_v2 = logit_diff_fn(logits, selected_pos['end'])

print(f"Zero Ablation logit difference 21, 28: {zero_logit_third_v2}")
# %%

list2 = set([9681,
 6963,
 13630,
 10093,
 9681,
 9244,
 6963,
 9244,
 10093,
 13630,
 1589,
 15125,
 9208,
 9681,
 6963,
 3786,
 9966,
 15278,
 3784,
 13873,
 14626,
 8643,
 488,
 632,
 9377,
 1327,
 10868,
 7030,
 10619,
 2128,
 8062,
 9827,
 2701,
 16049])

list2
# %%
import random



sampled= {'blocks.7.hook_resid_post': [488,
 632,
 1327,
 1589,
 2128,
 2701,
 3784,
 3786,
 6963,
 7030,
 8062,
 8643,
 9208,
 9244,
 9377,
 9681,
 9827,
 9966,
 10093,
 10619,
 10868,
 13630,
 13873,
 14626,
 15125,
 15278,
 16049]}

num_trials = 50
# Initial maximum results for sampled and sampled_v2
max_logit_third = -float('inf')
max_subset_third = None
max_logit_third_v2 = -float('inf')
max_subset_third_v2 = None

# Define a function to create random subsets from a given dictionary
def create_random_subset(input_dict, sample_size=None):
    subset = {}
    for key, values in input_dict.items():
        subset_size = sample_size if sample_size else random.randint(1, len(values))
        subset[key] = random.sample(values, subset_size)
    return subset

# Randomly iterate over subsets to maximize logit differences
for _ in range(num_trials):
    # Sample random subsets from each dictionary
    subset_sampled = create_random_subset(sampled)
    # subset_sampled_v2 = create_random_subset(samepled_v2)

    # Run model on subset_sampled
    logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, subset_sampled)
    logit_diff = logit_diff_fn(logits, selected_pos['end']).detach().cpu().item()

    # Check if we found a new max for sampled
    if logit_diff > max_logit_third:
        max_logit_third = logit_diff
        max_subset_third = subset_sampled
        print(f"New Max Logit Diff: {max_logit_third}")
        print(f"Subset: {max_subset_third}")

    # # Run model on subset_sampled_v2
    # logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, subset_sampled_v2)
    # logit_diff_v2 = logit_diff_fn(logits, selected_pos['end']).detach().cpu().item()

    # # Check if we found a new max for sampled_v2
    # if logit_diff_v2 > max_logit_third_v2:
    #     max_logit_third_v2 = logit_diff_v2
    #     max_subset_third_v2 = subset_sampled_v2

# Output the maximum results found
print(f"Maximum Zero Ablation logit difference for sampled: {max_logit_third} with subset {max_subset_third}")
# print(f"Maximum Zero Ablation logit difference for sampled_v2: {max_logit_third_v2} with subset {max_subset_third_v2}")
# %%

set_opt = set(max_subset_third['blocks.7.hook_resid_post'])
set_third = set(third_dict['blocks.7.hook_resid_post'])
set_comb = set(combined_dict['blocks.7.hook_resid_post'])

print(list(set_third - set_opt))
print(list(set_opt - set_comb))
# %%

new_set = combined_dict 
new_set['blocks.7.hook_resid_post'] = list(set_opt)
new_set['blocks.21.hook_resid_post'] = [13842, 12987, 4541, 9427, 3377, 14699]
new_set_selected = filter_dict_by_layers(new_set, [7, 14, 21, 28, 40])
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, new_set_selected)
logit_diff = logit_diff_fn(logits, selected_pos['end']).detach().cpu().item()
print(f"New Max Logit Diff: {logit_diff}")
# %%

# read the top_latents_results.json in third_latents_28
import json
with open('tasks/error_detection/type/out/third_latents_28/top_latents_results.json', 'r') as f:
    data = json.load(f)
    # print(data)

all_21 = []
all_14 = []
all_7 = []
json_paths = [
    "tasks/error_detection/type/out/third_latents_14/top_latents_results.json",
    "tasks/error_detection/type/out/third_latents_21/top_latents_results.json",
    "tasks/error_detection/type/out/third_latents_28/top_latents_results.json"
]

# third_latents = {"blocks.7.hook_resid_post":[], "blocks.14.hook_resid_post":[], "blocks.21.hook_resid_post":[]}

# pos_latents_28 = [7692, 1001, 6500]

# pos_latents_21 = [10795, 3377, 12987]

# pos_latents_14 = [14727, 3959]

for json_path in json_paths:
   # Load data from JSON file   
    with open(json_path, "r") as f:
        data = json.load(f)
    for target_latent, stuff in data.items():
        main_stuff = stuff['positive']
        for key, values in main_stuff.items():
            if '21' in key:
                for val in values:
                    if val['feature_idx'] not in all_21 and abs(val['value']) > 0:
                        all_21.append(val['feature_idx'])
            if '14' in key:
                for val in values:
                    if val['feature_idx'] not in all_14 and abs(val['value']) > 0:
                        all_14.append(val['feature_idx'])
            if '7' in key:
                for val in values:
                    if val['feature_idx'] not in all_7 and abs(val['value']) > 0:
                        all_7.append(val['feature_idx'])


# %%
len(list(set(all_21)))
# %%

list(set(third_dict['blocks.28.hook_resid_post']))

# %%

all_dict = {}

for key in combined_dict.keys():
#    if key in second_latents.keys():
    if '21' in key:
        all_dict[key] = list(set(combined_dict[key]) | set(all_21))
    elif '14' in key:
        all_dict[key] = list(set(combined_dict[key]) | set(all_14))
    elif '7' in key:
        all_dict[key] = list(set(combined_dict[key]) | set(all_7))
    else:
        all_dict[key] = combined_dict[key]

print(all_dict)

ttl_latents = 0
for key in all_dict.keys():
   ttl_latents += len(all_dict[key])
print(ttl_latents)


# %%


# samepled_v2 = {'blocks.14.hook_resid_post': list(set(all_14))} #, 'blocks.28.hook_resid_post': list(set(third_dict['blocks.28.hook_resid_post'])), 'blocks.14.hook_resid_post': list(set(third_dict['blocks.14.hook_resid_post']))} #, 'blocks.7.hook_resid_post': list(set(third_dict['blocks.7.hook_resid_post']))} #, 'blocks.40.hook_resid_post': list(set(third_dict['blocks.40.hook_resid_post']) )}
all_dict_selected = filter_dict_by_layers(all_dict, [21, 28])
# Run with third_dict
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, all_dict_selected)
zero_logit_third = logit_diff_fn(logits, selected_pos['end'])

print(f"Zero Ablation logit difference 21: {zero_logit_third}")

# %%
