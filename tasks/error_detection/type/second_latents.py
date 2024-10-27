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


# selected_pos = {"i_start": [], "i_end": [], "end": []}
# for i in range(N):
#     str_tokens_clean = model.to_str_tokens(samples[0][i])
#     str_tokens_corr = model.to_str_tokens(samples[1][i])
#     diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]
#     pos_end = len(str_tokens_clean) - 1
#     selected_pos["i_start"].append(diff_positions[0])
#     selected_pos["i_end"].append(diff_positions[-1])
#     selected_pos["end"].append(pos_end)


# # Define logit diff function
# traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
# trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]


# def logit_diff_fn(logits, selected_pos):
#     err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
#     no_err = logits[range(logits.size(0)), selected_pos, :][:, trip_arrow_token_id]
#     return (err - no_err).mean()


# Disable gradients for all parameters
for param in model.parameters():
   param.requires_grad_(False)


clean_tokens = model.to_tokens(samples[0])
corr_tokens = model.to_tokens(samples[1])


# # Compute logits for clean and corrupted samples
# logits = model(clean_tokens)
# clean_diff = logit_diff_fn(logits, selected_pos['end'])


# logits = model(corr_tokens)
# corr_diff = logit_diff_fn(logits, selected_pos['end'])


# print(f"clean_diff: {clean_diff}")
# print(f"corr_diff: {corr_diff}")


# # Cleanup
# del logits
# cleanup_cuda()




# # Define error type metric
# def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
#     patched_logit_diff = logit_diff_fn(logits, end_positions)
#     return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)


# err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=selected_pos['end'])


# %%


with open('tasks/error_detection/type/out/dict_feats.json') as f:
   dict_feats = json.load(f)
print(dict_feats.keys())
dict_feats['blocks.7.hook_resid_post']


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


# %% latents for L28.2102
  
def latent_patch_metric(cache, layer_ind=28, lat_ind=2102):
   # layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
   result = cache[f'blocks.{layer_ind}.hook_resid_post'][:, :, lat_ind].sum()
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


filtered_ids = [model.tokenizer.bos_token_id]
_, clean_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
latent_act_clean = latent_patch_metric(clean_cache)


_, corrupted_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)
latent_act_corrupted = latent_patch_metric(corrupted_cache)


print(f"Clean Patch: {latent_act_clean}, Corrupted Patch: {latent_act_corrupted}")


# %%


clean_cache['blocks.28.hook_resid_post'][0, :, 2102]




# %%
corrupted_cache['blocks.28.hook_resid_post'][0, :, 2102]




# %%


model.to_str_tokens(samples[0][0])




# %%


# Define error type metric
def _latent_metric(cache, clean_logit_diff, corr_logit_diff): #, layer_ind, lat_ind):
   patched_logit_diff = latent_patch_metric(cache) #, layer_ind, lat_ind)
   return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)


latent_metric_denoising = partial(_latent_metric, clean_logit_diff=latent_act_clean, corr_logit_diff=latent_act_corrupted) #, layer_ind=40, lat_ind=9447)


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


# import gc
# del cache, corrupted_cache, grad_cache
# gc.collect()


torch.cuda.empty_cache()


# %%


_, corr_cache, corr_grad_cache, val = get_cache_fwd_and_bwd_fml(model, corr_tokens, latent_metric_denoising, saes, filtered_ids, 'blocks.21.hook_resid_post')




# %%
# clean_cache.keys()


corr_grad_cache['blocks.7.hook_resid_post'].shape


# %%


latent_patch_metric(corr_cache)


# %%


latent_metric_denoising(corr_cache)


# %%


val


# %%


# # for key in corr_grad_cache.keys():
# key = 'blocks.21.hook_resid_post'
# resi_attr = einops.reduce(
#     corr_grad_cache[key] * (clean_cache[key] - corrupted_cache[key]),
#     'batch pos n_features -> n_features',
#     'sum'
# )
# resi_attr.shape


# %%
# K = 20
# topk_values, topk_indices = torch.topk(resi_attr, K)


# for i in range(K):
#     print(f"Feature Index: {topk_indices[i]}, Value: {topk_values[i]}")






# %%


import torch
import einops


keys = corr_grad_cache.keys()  # List of keys to process
K = 20  # Total number of top entries to retrieve


# List to store all (key, feature_idx, value) tuples
all_results = []


for key in keys:
   # Compute resi_attr for the current key
   resi_attr = einops.reduce(
       corr_grad_cache[key] * (clean_cache[key] - corr_cache[key]),
       'batch pos n_features -> n_features',
       'sum'
   )


   # save the resi_attr into out/second_latenst with key name as tensor
   # torch.save(resi_attr, f'tasks/error_detection/type/out/second_latents/L40_9447_{key}.pt')




   # Get the top values and indices for this resi_attr (considering all features)
   values, indices = resi_attr.topk(resi_attr.numel())  # Get all values sorted


   # Collect all (key, feature_idx, value) pairs
   all_results.extend([(key, int(idx), float(value)) for idx, value in zip(indices, values)])


# Sort all results by value and keep only the top 20
top_20_results = sorted(all_results, key=lambda x: x[2], reverse=True)[:K]


# Print the top 20 results across all keys
for key, feat_idx, value in top_20_results:
   print(f"Key: {key}, Feature Index: {feat_idx}, Value: {value}")




# %% absolute values


import torch
import einops


save_string = "L21_8255"


keys = corr_grad_cache.keys()  # List of keys to process
K = 20  # Total number of top entries to retrieve


# List to store all (key, feature_idx, value) tuples
all_results = []


for key in keys:
   # Compute resi_attr for the current key
   resi_attr = einops.reduce(
       corr_grad_cache[key] * (clean_cache[key] - corr_cache[key]),
       'batch pos n_features -> n_features',
       'sum'
   )
   # print the sum of resi_attr
   # print(f"Sum of resi_attr for {key}: {resi_attr.sum()}")


   # Save the resi_attr tensor to the specified directory with the key name
   torch.save(resi_attr, f'tasks/error_detection/type/out/second_latents/{save_string}_{key}.pt')


   # Get the sorted indices based on absolute values, but retain the signs
   abs_values, indices = resi_attr.abs().topk(resi_attr.numel())
  
   # Collect all (key, feature_idx, original signed value) pairs
   all_results.extend([(key, int(idx), float(resi_attr[idx])) for idx in indices])


# Sort all results by absolute value but retain original signs, and keep only the top 20
top_20_results = sorted(all_results, key=lambda x: abs(x[2]), reverse=True)[:K]


# Print the top 20 results across all keys, keeping their original signs
for key, feat_idx, value in top_20_results:
   print(f"Key: {key}, Feature Index: {feat_idx}, Value: {value}")




# %%


# save top 20 results as json
top_20_results_dict = [{"key": key, "feature_idx": feat_idx, "value": value} for key, feat_idx, value in top_20_results]
with open(f'tasks/error_detection/type/out/second_latents/top_20_results_{save_string}.json', 'w') as f:
   json.dump(top_20_results_dict, f)






# %%
K =20
flattened_tensor = resi_attr.flatten()


# Step 2: Use torch.topk to find the top K values and their indices
topk_values, topk_indices = torch.topk(flattened_tensor, K)


# Step 3: Convert the flattened indices back to (token_pos, feature_idx) pairs
token_pos_indices = topk_indices // resi_attr.size(1)
feature_indices = topk_indices % resi_attr.size(1)


# Combine them as pairs and include the values
topk_results = [(token_pos.item(), feature_idx.item(), value.item())
               for token_pos, feature_idx, value in zip(token_pos_indices, feature_indices, topk_values)]


# Print the results
for pos, feature, value in topk_results:
   print(f"(token_pos: {pos}, feature_idx: {feature}) -> value: {value}")








# %%


grad_cache['blocks.7.hook_resid_post']






# %%


hook = saes[1].cfg.hook_name


for sae in saes:
   if sae.cfg.hook_name == hook:
       print(sae.cfg.hook_name)
       break






# %%
  
{saes[1].hook_name:"hwjadw"}














# %%


lat_ind = 15191
layer_ind = 8


def latent_patch_metric(cache):
   # layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
   result = cache[:, :, 15191].sum()
   # print(result.requires_grad)
   return result
   # return cache[layer_name][:, :, lat_ind].sum()


_, clean_cache = model.run_with_cache_with_saes(clean_prompts, saes=saes[1])
_, corrupted_cache = model.run_with_cache_with_saes(corrupted_prompts, saes=saes[1])
clean_patch = latent_patch_metric(clean_cache[f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'])
corrupted_patch = latent_patch_metric(corrupted_cache[f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'])
print(f"Clean Patch: {clean_patch}, Corrupted Patch: {corrupted_patch}")






# %%
parent_feat = ["L21_1408", "L28_1800", "L28_2102", "L40_11839", "L40_9447", "L21_8255", "L14_14967"]
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
with open('tasks/error_detection/type/out/dict_feats.json') as f:
   dict_feats = json.load(f)
dict_feats
# %%
# Combine the dict_feats and second_latents
combined_dict = {}
for key in dict_feats.keys():
   if key in second_latents.keys():
       combined_dict[key] = dict_feats[key] + second_latents[key]
   else:
       combined_dict[key] = dict_feats[key]


print(combined_dict)




# %%


ttl_latents = 0
for key in combined_dict.keys():
   ttl_latents += len(combined_dict[key])
print(ttl_latents)


# %%


# save the combined_dict as json
with open('tasks/error_detection/type/out/second_dict_feats.json', 'w') as f:
   json.dump(combined_dict, f)


# %%
  
cleanup_cuda()
filtered_ids = [model.tokenizer.bos_token_id]
_, clean_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
_, corrupted_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)




# %%
  
corr_sae_cache_means = {layer: sae_cache.mean(dim=0) for layer, sae_cache in corrupted_cache.items()}


print(corrupted_cache['blocks.7.hook_resid_post'].shape)
print(corr_sae_cache_means['blocks.7.hook_resid_post'].shape)


# %%


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


selected_pos = {"i_start": [], "i_end": [], "end": []}
for i in range(N):
   str_tokens_clean = model.to_str_tokens(samples[0][i])
   str_tokens_corr = model.to_str_tokens(samples[1][i])
   diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]
   pos_end = len(str_tokens_clean) - 1
   selected_pos["i_start"].append(diff_positions[0])
   selected_pos["i_end"].append(diff_positions[-1])
   selected_pos["end"].append(pos_end)




# %%


logits = run_with_saes_mean_ablation(clean_tokens, filtered_ids, model, saes, corr_sae_cache_means, combined_dict)


traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]
def logit_diff_fn(logits, selected_pos):
   err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
   no_err = logits[range(logits.size(0)), selected_pos, :][:, trip_arrow_token_id]
   return (err - no_err).mean()


mean_logit = logit_diff_fn(logits, selected_pos['end'])
print(f"mean_logit: {mean_logit}")


# %%






with torch.no_grad():
   logits = model(corr_tokens)
corr_logit_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"corr_logit_diff: {corr_logit_diff}")


# %%
combined_dict


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
           if hook.name in dict_feats and ('40' in hook.name or '28' in hook.name):
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
combined_dict
# %%




