# %%

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

# # Compute logits for clean and corrupted samples
logits = model(clean_tokens)
clean_diff = logit_diff_fn(logits, selected_pos['end'])


logits = model(corr_tokens)
corr_diff = logit_diff_fn(logits, selected_pos['end'])


print(f"clean_diff: {clean_diff}")
print(f"corr_diff: {corr_diff}")


# # Cleanup
del logits
cleanup_cuda()

# # Define error type metric
def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = logit_diff_fn(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)


err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=selected_pos['end'])


# %%

with open('tasks/error_detection/type/out/dict_feats.json') as f:
   dict_feats = json.load(f)


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

# %% Helpers

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


def run_with_saes_latent_op_patch(new_tokens, filtered_ids, model, saes, cache, dict_feats):
   # Ensure tokens are a torch.Tensor
   if not isinstance(new_tokens, torch.Tensor):
       new_tokens = torch.tensor(new_tokens).to(model.cfg.device)  # Move to the device of the model

   # Create a mask where True indicates positions to modify
   mask = torch.ones_like(new_tokens, dtype=torch.bool)
   for token_id in filtered_ids:
       mask &= new_tokens != token_id

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
          
           if hook.name in cache and hook.name in dict_feats:
               prev_sae = cache[hook.name]  # Get cached activations from the cache
               feature_indices = dict_feats[hook.name]  # Get the feature indices to patch

               for feature_idx in range(sae.cfg.d_sae):
                   if feature_idx in feature_indices:
                       enc_sae[:, :, feature_idx] = prev_sae[:, :, feature_idx]

           # After patching, decode the modified enc_sae
           modified_act = sae.decode(enc_sae)

           # In-place update where the mask is True
           updated_act = torch.where(mask_expanded, modified_act, act)

           return updated_act

       # Add the hook to the model
       model.add_hook(hook_point, filtered_hook, dir='fwd')

   # Run the model with the tokens (no gradients needed)
   with torch.no_grad():
       logits = model(new_tokens)

   # Reset the hooks after computation to free memory
   model.reset_hooks()

   return logits  # Return only the logits

def run_with_saes_latent_op_patch_cache(new_tokens, filtered_ids, model, saes, cache, dict_feats):
   # Ensure tokens are a torch.Tensor
   if not isinstance(new_tokens, torch.Tensor):
      new_tokens = torch.tensor(new_tokens).to(model.cfg.device)  # Move to the device of the model

   # Create a mask where True indicates positions to modify
   mask = torch.ones_like(new_tokens, dtype=torch.bool)
   for token_id in filtered_ids:
      mask &= new_tokens != token_id

   # Expand the mask once, so it matches the shape [batch_size, seq_len, 1]
   mask_expanded = mask.unsqueeze(-1)  # Expand to allow broadcasting
   mask_expanded = mask_expanded.to(model.cfg.device)  # Move the mask to the same device as the model
   sae_outs = {}
   # For each SAE, add the appropriate hook
   for sae in saes:
      hook_point = sae.cfg.hook_name

      # Define the filtered hook function (optimized)
      def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
         # Apply the SAE only where mask_expanded is True
         enc_sae = sae.encode(act)  # Call the SAE once
         
         if hook.name in cache and hook.name in dict_feats:
            prev_sae = cache[hook.name]  # Get cached activations from the cache
            feature_indices = dict_feats[hook.name]  # Get the feature indices to patch

            for feature_idx in range(sae.cfg.d_sae):
               if feature_idx in feature_indices:
                  enc_sae[:, :, feature_idx] = prev_sae[:, :, feature_idx]
         sae_outs[hook.name] = enc_sae.detach().cpu()
         # After patching, decode the modified enc_sae
         modified_act = sae.decode(enc_sae)

         # In-place update where the mask is True
         updated_act = torch.where(mask_expanded, modified_act, act)

         return updated_act

      # Add the hook to the model
      model.add_hook(hook_point, filtered_hook, dir='fwd')

   # Run the model with the tokens (no gradients needed)
   with torch.no_grad():
      logits = model(new_tokens)

   # Reset the hooks after computation to free memory
   model.reset_hooks()

   return logits, sae_outs  # Return only the logits
# def run_with_saes_latent_edge_patch(new_tokens, filtered_ids, model, saes, cache, sender_feats, receiver_feats):
    

# %% clean cache 
filtered_ids = [model.tokenizer.bos_token_id]
logits, clean_sae_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids=filtered_ids, model=model, saes=saes)
clean_sae_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"clean_sae_diff: {clean_sae_diff}")


# %% corr cache 
logits, corr_sae_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids=filtered_ids, model=model, saes=saes)
corr_sae_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"corr_sae_diff: {corr_sae_diff}")

# %% denoising patching 
from tqdm import tqdm
total_steps = sum([len(latents) for latents in combined_dict.values()])
denoising_results = {}
with tqdm(total=total_steps, desc="Denoising Progress") as pbar:
   for layer, latents in combined_dict.items():
      # print(f"Layer: {layer}")
      denoising_results[layer] = {}
      for latent in latents:
         # print(f"Latent: {latent}")
         filtered_ids = [model.tokenizer.bos_token_id] 
         small_dict = {layer: [latent]}
         logits = run_with_saes_latent_op_patch(corr_tokens, filtered_ids, model, saes, clean_sae_cache, small_dict)
         patched_err_metric = logit_diff_fn(logits, selected_pos['end'])
         normalized_metric = (patched_err_metric - corr_sae_diff) / (clean_sae_diff - corr_sae_diff)
         # print(f"Error Metric: {normalized_metric}")
         denoising_results[layer][latent] = normalized_metric.detach().cpu().item()
         pbar.update(1)
denoising_results

# %% noising patching
from tqdm import tqdm
total_steps = sum([len(latents) for latents in combined_dict.values()])
noising_results = {}
with tqdm(total=total_steps, desc="Noising Progress") as pbar:
   for layer, latents in combined_dict.items():
      # print(f"Layer: {layer}")
      noising_results[layer] = {}
      for latent in latents:
         # print(f"Latent: {latent}")
         filtered_ids = [model.tokenizer.bos_token_id] 
         small_dict = {layer: [latent]}
         logits = run_with_saes_latent_op_patch(clean_tokens, filtered_ids, model, saes, corr_sae_cache, small_dict)
         patched_err_metric = logit_diff_fn(logits, selected_pos['end'])
         normalized_metric = (patched_err_metric - clean_sae_diff) / (corr_sae_diff - clean_sae_diff)
         # print(f"Error Metric: {normalized_metric}")
         noising_results[layer][latent] = normalized_metric.detach().cpu().item()
         pbar.update(1)
noising_results


# %%
import heapq

def get_top_k_latents(results_dict, k):
    # Flatten the results_dict into a list of tuples (layer, latent, metric)
    flattened = [(layer, latent, metric) for layer, latents in results_dict.items() for latent, metric in latents.items()]
    # Get the top K elements based on metric (in descending order)
    top_k = heapq.nlargest(k, flattened, key=lambda x: x[2])
    return top_k

# Calculate top K latents for each category
K = 5  # Replace with your desired top K value

# Top K latents for denoising
top_k_denoising = get_top_k_latents(denoising_results, K)
print("Top K Denoising Latents:", top_k_denoising)

# Top K latents for noising
top_k_noising = get_top_k_latents(noising_results, K)
print("Top K Noising Latents:", top_k_noising)

# Combine denoising and noising results to get denoising+noising scores
combined_results = {}
for layer in denoising_results.keys():
    combined_results[layer] = {}
    for latent in denoising_results[layer].keys():
        denoising_score = denoising_results[layer][latent]
        noising_score = noising_results[layer].get(latent, 0)
        combined_results[layer][latent] = denoising_score + noising_score

# Top K latents for denoising + noising
top_k_denoising_noising = get_top_k_latents(combined_results, K)
print("Top K Denoising + Noising Latents:", top_k_denoising_noising)


# %%

receiver_layer = 'blocks.28.hook_resid_post'
receiver_latent = 2102

def latent_patch_metric(cache, layer_ind=28, lat_ind=2102):
   # layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
   result = cache[f'blocks.{layer_ind}.hook_resid_post'][:, :, lat_ind].sum()
   # print(result.requires_grad)
   return result

clean_receiver_metric = latent_patch_metric(clean_sae_cache, layer_ind=28, lat_ind=2102)
corr_receiver_metric = latent_patch_metric(corr_sae_cache, layer_ind=28, lat_ind=2102)
print(f"clean_receiver_metric: {clean_receiver_metric}")
print(f"corr_receiver_metric: {corr_receiver_metric}")


sender_layer = 'blocks.21.hook_resid_post'
sender_latent = 1408
small_dict = {'blocks.21.hook_resid_post': [1408]}
filtered_ids = [model.tokenizer.bos_token_id]
_, patched_sender_cache = run_with_saes_latent_op_patch_cache(corr_tokens, filtered_ids, model, saes, clean_sae_cache, small_dict)
patched_latent_metric = latent_patch_metric(patched_sender_cache, layer_ind=28, lat_ind=2102)
print(f"patched_latent_metric denoising: {patched_latent_metric}")
normalized_metric = (patched_latent_metric - corr_receiver_metric) / (clean_receiver_metric - corr_receiver_metric)
print(f"Normalized Metric: {normalized_metric}")


_, patched_sender_cache = run_with_saes_latent_op_patch_cache(clean_tokens, filtered_ids, model, saes, corr_sae_cache, small_dict)
patched_latent_metric = latent_patch_metric(patched_sender_cache, layer_ind=28, lat_ind=2102)
print(f"patched_latent_metric noising: {patched_latent_metric}")
normalized_metric = (patched_latent_metric - clean_receiver_metric) / (corr_receiver_metric - clean_receiver_metric)
print(f"Normalized Metric: {normalized_metric}")

# %%

# Initialize a dictionary to store results
top_k_effects = {}

# Loop over each top K latent and treat it as the receiver
for receiver_layer, receiver_latent, _ in top_k_denoising_noising:
   # Parse the layer number of the receiver layer to set a range for prior layers
   receiver_layer_num = int(receiver_layer.split('.')[1])

   # Initialize results storage for the current receiver
   top_k_effects[(receiver_layer, receiver_latent)] = {}
   print(f"Receiver Layer: {receiver_layer}, Receiver Latent: {receiver_latent}")

   # Calculate clean and corrupted base metrics for this receiver latent
   clean_receiver_metric = latent_patch_metric(clean_sae_cache, layer_ind=receiver_layer_num, lat_ind=receiver_latent)
   corr_receiver_metric = latent_patch_metric(corr_sae_cache, layer_ind=receiver_layer_num, lat_ind=receiver_latent)
   print(f"  clean_receiver_metric: {clean_receiver_metric}")
   print(f"  corr_receiver_metric: {corr_receiver_metric}")

   # Iterate through each previous layer as potential sender layers
   for sender_layer_num in range(receiver_layer_num):
      # Check if this layer has latents in the combined_dict to avoid empty iterations
      if sender_layer in combined_dict:
         sender_layer = f'blocks.{sender_layer_num}.hook_resid_post'
         print(f"Sender Layer: {sender_layer}")
         # Iterate through each latent in the sender layer
         for sender_latent in combined_dict[sender_layer]:
               print(f"Sender Latent: {sender_latent}")
               small_dict = {sender_layer: [sender_latent]}
               filtered_ids = [model.tokenizer.bos_token_id]

               # Perform denoising and compute the normalized metric
               _, patched_sender_cache = run_with_saes_latent_op_patch_cache(
                  corr_tokens, filtered_ids, model, saes, clean_sae_cache, small_dict
               )
               patched_latent_metric_denoising = latent_patch_metric(
                  patched_sender_cache, layer_ind=receiver_layer_num, lat_ind=receiver_latent
               )
               normalized_metric_denoising = (patched_latent_metric_denoising - corr_receiver_metric) / \
                                             (clean_receiver_metric - corr_receiver_metric)

               # Perform noising and compute the normalized metric
               _, patched_sender_cache = run_with_saes_latent_op_patch_cache(
                  clean_tokens, filtered_ids, model, saes, corr_sae_cache, small_dict
               )
               patched_latent_metric_noising = latent_patch_metric(
                  patched_sender_cache, layer_ind=receiver_layer_num, lat_ind=receiver_latent
               )
               normalized_metric_noising = (patched_latent_metric_noising - clean_receiver_metric) / \
                                          (corr_receiver_metric - clean_receiver_metric)

               print(f"  Denoising Metric: {normalized_metric_denoising}")
               print(f"  Noising Metric: {normalized_metric_noising}")
               # Store the results for this sender and receiver pair
               top_k_effects[(receiver_layer, receiver_latent)][(sender_layer, sender_latent)] = {
                  'denoising_metric': normalized_metric_denoising.detach().cpu().item(),
                  'noising_metric': normalized_metric_noising.detach().cpu().item()
               }
         break
   break

# The results are now stored in `top_k_effects` with structure:
# top_k_effects = {
#     (receiver_layer, receiver_latent): {
#         (sender_layer, sender_latent): {
#             'denoising_metric': <value>,
#             'noising_metric': <value>
#         },
#         ...
#     },
#     ...
# }




# %%


def run_with_saes_latent_op_patch(new_tokens, filtered_ids, model, saes, cache, dict_feats):
   # Ensure tokens are a torch.Tensor
   if not isinstance(new_tokens, torch.Tensor):
       new_tokens = torch.tensor(new_tokens).to(model.cfg.device)  # Move to the device of the model

   # Create a mask where True indicates positions to modify
   mask = torch.ones_like(new_tokens, dtype=torch.bool)
   for token_id in filtered_ids:
       mask &= new_tokens != token_id

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
          
           if hook.name in cache and hook.name in dict_feats:
               prev_sae = cache[hook.name]  # Get cached activations from the cache
               feature_indices = dict_feats[hook.name]  # Get the feature indices to patch

               for feature_idx in range(sae.cfg.d_sae):
                   if feature_idx in feature_indices:
                       enc_sae[:, :, feature_idx] = prev_sae[:, :, feature_idx]

           # After patching, decode the modified enc_sae
           modified_act = sae.decode(enc_sae)

           # In-place update where the mask is True
           updated_act = torch.where(mask_expanded, modified_act, act)

           return updated_act

       # Add the hook to the model
       model.add_hook(hook_point, filtered_hook, dir='fwd')

   # Run the model with the tokens (no gradients needed)
   with torch.no_grad():
       logits = model(new_tokens)

   # Reset the hooks after computation to free memory
   model.reset_hooks()

   return logits  # Return only the logits
