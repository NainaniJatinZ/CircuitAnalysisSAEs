# %% model and imports 

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

layers= [7, 14, 21, 40]
saes = [SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical", device=device)[0] for layer in layers]

# %% data 

# Updated version to return JSON with more names and structure for correct and incorrect keying examples

import json
import random

# Expanding the name pool with a larger set of names
extended_name_pool = [
    "Bob", "Sam", "Lilly", "Rob", "Alice", "Charlie", "Sally", "Tom", "Jake", "Emily", 
    "Megan", "Chris", "Sophia", "James", "Oliver", "Isabella", "Mia", "Jackson", 
    "Emma", "Ava", "Lucas", "Benjamin", "Ethan", "Grace", "Olivia", "Liam", "Noah"
]

for name in extended_name_pool:
    assert len(model.tokenizer.encode(name)) == 2, f"Name {name} has more than 1 token"

# Function to generate the dataset with correct and incorrect keying into dictionaries
def generate_extended_dataset(name_pool, num_samples=5):
    dataset = []
    
    for _ in range(num_samples):
        # Randomly select 5 names from the pool
        selected_names = random.sample(name_pool, 5)
        # Assign random ages to the selected names
        age_dict = {name: random.randint(10, 19) for name in selected_names}
        
        # Create a correct example
        correct_name = random.choice(list(age_dict.keys()))
        correct_prompt = f'Type "help", "copyright", "credits" or "license" for more information.\n>>> age = {age_dict}\n>>> age["{correct_name}"]\n'
        correct_response = age_dict[correct_name]
        correct_token = str(correct_response)[0]
        
        # Create an incorrect example with a name not in the dictionary
        incorrect_name = random.choice([name for name in name_pool if name not in age_dict])
        incorrect_prompt = f'Type "help", "copyright", "credits" or "license" for more information.\n>>> age = {age_dict}\n>>> age["{incorrect_name}"]\n'
        incorrect_response = "Traceback"
        incorrect_token = "Traceback"
        
        # Append the pair of correct and incorrect examples
        dataset.append({
            "correct": {
                "prompt": correct_prompt,
                "response": correct_response,
                "token": correct_token
            },
            "error": {
                "prompt": incorrect_prompt,
                "response": incorrect_response,
                "token": incorrect_token
            }
        })
        
    return dataset

# Generate the extended dataset
json_dataset = generate_extended_dataset(extended_name_pool, num_samples=100_000)

clean_prompts = []
corr_prompts = []

answer_token = model.to_single_token("1")
traceback_token = model.to_single_token("Traceback")

for item in json_dataset[:50]:
    corr_prompts.append(item["correct"]["prompt"])
    clean_prompts.append(item["error"]["prompt"])

clean_tokens = model.to_tokens(clean_prompts)
corr_tokens = model.to_tokens(corr_prompts)

# %%
model.reset_hooks()
def logit_diff_fn(logits):
    err = logits[:, -1, traceback_token]
    no_err = logits[:, -1, answer_token]
    return (err - no_err).mean()

# Disable gradients for all parameters
for param in model.parameters():
   param.requires_grad_(False)

# # Compute logits for clean and corrupted samples
logits = model(clean_tokens)
clean_diff = logit_diff_fn(logits)

logits = model(corr_tokens)
corr_diff = logit_diff_fn(logits)

print(f"clean_diff: {clean_diff}")
print(f"corr_diff: {corr_diff}")

# # Cleanup
del logits
cleanup_cuda()

# # Define error type metric
def _err_type_metric(logits, clean_logit_diff, corr_logit_diff):
    patched_logit_diff = logit_diff_fn(logits)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff)

# %% helpers

def run_with_saes_filtered_cache(tokens, filtered_ids, model, saes):
    with torch.no_grad():  # Global no_grad for performance
        
        # Ensure tokens is on the correct device
        tokens = torch.as_tensor(tokens, device=model.cfg.device)

        # Vectorized mask creation
        mask = ~torch.isin(tokens, torch.tensor(filtered_ids, device=model.cfg.device))
        mask_expanded = mask[:, :, None]  # Equivalent to unsqueeze(-1)

        sae_outs = {}
        
        # Define and add hooks
        for sae in saes:
            hook_point = sae.cfg.hook_name
            
            def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
                enc_sae = sae.encode(act)
                sae_outs[hook.name] = enc_sae.detach().cpu()
                modified_act = sae.decode(enc_sae)
                
                # Apply torch.where only if mask_expanded has True values
                updated_act = torch.where(mask_expanded, modified_act, act) if mask_expanded.any() else act
                return updated_act
            
            model.add_hook(hook_point, filtered_hook, dir='fwd')
        
        logits = model(tokens)
        model.reset_hooks()
        
    return logits, sae_outs


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
               prev_sae = cache[hook.name].to(device)   # Get cached activations from the cache
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
    with torch.no_grad():  # Global no_grad for efficiency

        # Ensure new_tokens is a tensor on the correct device
        new_tokens = torch.as_tensor(new_tokens, device=model.cfg.device)
        
        # Create mask in a vectorized way
        mask = ~torch.isin(new_tokens, torch.tensor(filtered_ids, device=model.cfg.device))
        mask_expanded = mask[:, :, None]  # Expand dimensions directly
        
        # Cache hook names and feature indices outside the hook
        sae_hook_configs = [(sae.cfg.hook_name, sae, dict_feats.get(sae.cfg.hook_name, [])) for sae in saes]
        sae_outs = {}

        # Define and apply hooks
        for hook_name, sae, feature_indices in sae_hook_configs:
            def filtered_hook(act, hook, sae=sae, feature_indices=feature_indices, mask_expanded=mask_expanded):
                enc_sae = sae.encode(act)
                if hook.name in cache and feature_indices:
                    # Vectorized update of selected feature indices
                    enc_sae[:, :, feature_indices] = cache[hook.name][:, :, feature_indices].to(device) 

                sae_outs[hook.name] = enc_sae.detach().cpu()
                modified_act = sae.decode(enc_sae)
                
                # Only apply torch.where if needed
                updated_act = torch.where(mask_expanded, modified_act, act) if mask_expanded.any() else act
                return updated_act
            
            model.add_hook(hook_name, filtered_hook, dir='fwd')

        logits = model(new_tokens)
        model.reset_hooks()

    return logits, sae_outs
    
def run_with_saes_zero_ablation(tokens, filtered_ids, model, saes, dict_feats):
    with torch.no_grad():  # Global no_grad for performance

        # Ensure tokens is on the correct device
        tokens = torch.as_tensor(tokens, device=model.cfg.device)

        # Vectorized mask creation
        mask = ~torch.isin(tokens, torch.tensor(filtered_ids, device=model.cfg.device))
        mask_expanded = mask[:, :, None]

        # Define and add hooks
        for sae in saes:
            hook_point = sae.cfg.hook_name
            
            def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
                enc_sae = sae.encode(act)
                
                if hook.name in dict_feats:
                    feature_indices = dict_feats[hook.name]
                    # Use advanced indexing to zero-out non-selected features
                    all_indices = torch.arange(sae.cfg.d_sae, device=model.cfg.device)
                    zero_indices = all_indices[~torch.isin(all_indices, torch.tensor(feature_indices, device=model.cfg.device))]
                    enc_sae[:, :, zero_indices] = 0

                modified_act = sae.decode(enc_sae)
                updated_act = torch.where(mask_expanded, modified_act, act) if mask_expanded.any() else act
                return updated_act
            
            model.add_hook(hook_point, filtered_hook, dir='fwd')
        
        logits = model(tokens)
        model.reset_hooks()
        
    return logits


def run_with_saes_zero_ablation_cache(tokens, filtered_ids, model, saes, dict_feats):
    with torch.no_grad():  # Global no_grad for performance

        # Ensure tokens is on the correct device
        tokens = torch.as_tensor(tokens, device=model.cfg.device)

        # Vectorized mask creation
        mask = ~torch.isin(tokens, torch.tensor(filtered_ids, device=model.cfg.device))
        mask_expanded = mask[:, :, None]
        sae_out = {}
        # Define and add hooks
        for sae in saes:
            hook_point = sae.cfg.hook_name
            
            def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
                enc_sae = sae.encode(act)
                
                if hook.name in dict_feats:
                    feature_indices = dict_feats[hook.name]
                    # Use advanced indexing to zero-out non-selected features
                    all_indices = torch.arange(sae.cfg.d_sae, device=model.cfg.device)
                    zero_indices = all_indices[~torch.isin(all_indices, torch.tensor(feature_indices, device=model.cfg.device))]
                    enc_sae[:, :, zero_indices] = 0
                sae_out[hook.name] = enc_sae.detach().cpu()
                modified_act = sae.decode(enc_sae)
                updated_act = torch.where(mask_expanded, modified_act, act) if mask_expanded.any() else act
                return updated_act
            
            model.add_hook(hook_point, filtered_hook, dir='fwd')
        
        logits = model(tokens)
        model.reset_hooks()
        
    return logits, sae_out


def run_with_saes_zero_ablation_cache_tokens(tokens, filtered_ids, model, saes, dict_feats, token_pos):
    with torch.no_grad():  # Global no_grad for performance

        # Ensure tokens is on the correct device
        tokens = torch.as_tensor(tokens, device=model.cfg.device)

        # Vectorized mask creation
        mask = ~torch.isin(tokens, torch.tensor(filtered_ids, device=model.cfg.device))
        mask_expanded = mask[:, :, None]
        sae_out = {}
        # Define and add hooks
        for sae in saes:
            hook_point = sae.cfg.hook_name
            
            def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
                enc_sae = sae.encode(act)
                
                if hook.name in dict_feats:
                    feature_indices = dict_feats[hook.name]
                    # Use advanced indexing to zero-out non-selected features
                    all_indices = torch.arange(sae.cfg.d_sae, device=model.cfg.device)
                    zero_indices = all_indices[~torch.isin(all_indices, torch.tensor(feature_indices, device=model.cfg.device))]
                    # Zero out only at specified token positions
                    for pos in token_pos:
                        enc_sae[:, pos, zero_indices] = 0
                    # enc_sae[:, :, zero_indices] = 0
                sae_out[hook.name] = enc_sae.detach().cpu()
                modified_act = sae.decode(enc_sae)
                updated_act = torch.where(mask_expanded, modified_act, act) if mask_expanded.any() else act
                return updated_act
            
            model.add_hook(hook_point, filtered_hook, dir='fwd')
        
        logits = model(tokens)
        model.reset_hooks()
        
    return logits, sae_out


def run_with_saes_zero_ablation_cache_tokens_separate(tokens, filtered_ids, model, saes, dict_feats, dict_feats_v2, token_pos):
    with torch.no_grad():  # Global no_grad for performance

        # Ensure tokens is on the correct device
        tokens = torch.as_tensor(tokens, device=model.cfg.device)

        # Vectorized mask creation
        mask = ~torch.isin(tokens, torch.tensor(filtered_ids, device=model.cfg.device))
        mask_expanded = mask[:, :, None]
        sae_out = {}
        # Define and add hooks
        for sae in saes:
            hook_point = sae.cfg.hook_name
            
            def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
                enc_sae = sae.encode(act)
                
                if hook.name in dict_feats:
                    feature_indices = dict_feats[hook.name]
                    feature_indices_v2 = dict_feats_v2[hook.name]
                    # Use advanced indexing to zero-out non-selected features
                    all_indices = torch.arange(sae.cfg.d_sae, device=model.cfg.device)

                    zero_indices = all_indices[~torch.isin(all_indices, torch.tensor(feature_indices, device=model.cfg.device))]

                    enc_sae[:, :, zero_indices] = 0

                    zero_indices_v2 = all_indices[~torch.isin(all_indices, torch.tensor(feature_indices_v2, device=model.cfg.device))]
                    # Zero out only at specified token positions
                    if feature_indices_v2:
                        for pos in token_pos:
                            enc_sae[:, pos, zero_indices_v2] = 0
                    
                    # enc_sae[:, :, zero_indices] = 0
                sae_out[hook.name] = enc_sae.detach().cpu()
                modified_act = sae.decode(enc_sae)
                updated_act = torch.where(mask_expanded, modified_act, act) if mask_expanded.any() else act
                return updated_act
            
            model.add_hook(hook_point, filtered_hook, dir='fwd')
        
        logits = model(tokens)
        model.reset_hooks()
        
    return logits, sae_out

# %%
cleanup_cuda()


# %% running with saes 
model.reset_hooks()
filtered_ids = [model.tokenizer.bos_token_id]
clean_sae_logits, clean_sae_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
corr_sae_logits, corr_sae_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)

clean_sae_diff = logit_diff_fn(clean_sae_logits)
corr_sae_diff = logit_diff_fn(corr_sae_logits)

print(f"clean_sae_diff: {clean_sae_diff}")
print(f"corr_sae_diff: {corr_sae_diff}")

# %% json loading 

with open('mask_finding/mask.json') as f:
    mask = json.load(f)

# load clustered latnets as cluster_results 
with open('mask_finding/out/clustering/clustered_latents.json') as f:
    cluster_results = json.load(f)

# load the drop results json
with open('mask_finding/out/clustering/drop_results.json') as f:
    drop_results = json.load(f)

# %%
model.reset_hooks()
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, mask)
clean_sae_diff_ablation = logit_diff_fn(logits)
print(f"clean_sae_diff_ablation: {clean_sae_diff_ablation}")

logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
corr_sae_diff_ablation = logit_diff_fn(logits)
print(f"corr_sae_diff_ablation: {corr_sae_diff_ablation}")


# %% 
del logits 
cleanup_cuda()

# %% Hypothesis 1: Duplicate token latents are important for predicting the correct answer in error free case

# duplicate token latents are cluster 0 for layer 7 
duplicate_latents = cluster_results['blocks.7.hook_resid_post']['sum_clusters']['0']

filtered_mask = mask.copy()
filtered_mask['blocks.7.hook_resid_post'] = list(set(mask['blocks.7.hook_resid_post']) - set(duplicate_latents))
model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, filtered_mask)

log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, answer_token]
circuit_dup_ablated = log_probs.mean().detach().cpu().item()
print("Duplicate cluster ablated: ", circuit_dup_ablated)

model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, answer_token]
full_circuit_perf = log_probs.mean().detach().cpu().item()
print("Full circuit perf: ", full_circuit_perf)

del logits, log_probs
cleanup_cuda()


# %%
fully_filtered_mask = {'blocks.7.hook_resid_post': list(set(range(saes[0].cfg.d_sae)) - set(duplicate_latents))}

model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, fully_filtered_mask)
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, answer_token]
model_dup_ablated = log_probs.mean().detach().cpu().item()
print("Duplicate cluster ablated, all latents: ",model_dup_ablated)

model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, {})
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, answer_token]
full_model_perf = log_probs.mean().detach().cpu().item()
print("All latents: ",full_model_perf)

del logits, log_probs
cleanup_cuda()



# %%

import matplotlib.pyplot as plt
import numpy as np

# Define the labels and values
conditions = ['Model + SAE', 'Zero-Ablated Circuit']
ablated_values = [model_dup_ablated, circuit_dup_ablated]
non_ablated_values = [full_model_perf, full_circuit_perf]

# Bar positions
x = np.arange(len(conditions))

# Width for each bar in a group
bar_width = 0.35

# Plotting
plt.figure(figsize=(8, 6))
bars1 = plt.bar(x - bar_width/2, ablated_values, width=bar_width, label='Duplicate Latents Ablated (only 2)', color='#4C72B0')
bars2 = plt.bar(x + bar_width/2, non_ablated_values, width=bar_width, label='Duplicate Latents Not Ablated', color='#55A868')

# Title and labels
plt.title("Effect of ablating duplicate token latents on \n Probability of correct answer for error free code", fontsize=14, weight='bold')
plt.xlabel("Conditions", fontsize=14)
plt.ylabel("Log Probability of Correct Answer", fontsize=12)

# Set x-axis tick positions and labels
plt.xticks(x, conditions, fontsize=12)

# Adding legend
plt.legend(fontsize=12)

# Display the values on top of the bars
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + 0.01, 
            f"{yval:.4f}", 
            ha='center', 
            va='bottom', 
            fontsize=11
        )

plt.tight_layout()
plt.show()



# %% Hypothesis 6: “1 is next” by induction doesn’t fire at the end position if duplicate key is not detected 

# H6.1 Ablating the duplicate detectors fully causes this feature to turn off
cleanup_cuda()
# duplicate token latents are cluster 0 for layer 7 
duplicate_latents = cluster_results['blocks.7.hook_resid_post']['sum_clusters']['0']

filtered_mask = mask.copy()
filtered_mask['blocks.7.hook_resid_post'] = list(set(mask['blocks.7.hook_resid_post']) - set(duplicate_latents))

model.reset_hooks()
_, cache = run_with_saes_zero_ablation_cache(corr_tokens, filtered_ids, model, saes, mask)
L40_9682_activation_circuit = cache['blocks.40.hook_resid_post'][:, :, 9682].mean(0)
# print("L40_9682_activation: ", L40_9682_activation)
# print("L40_9682_activation: ", L40_9682_activation.mean(dim))

model.reset_hooks()
_, cache = run_with_saes_zero_ablation_cache(corr_tokens, filtered_ids, model, saes, filtered_mask)
L40_9682_activation_circuit_ablated = cache['blocks.40.hook_resid_post'][:, :, 9682].mean(0)
# print("L40_9682_activation: ", L40_9682_activation)

del cache
cleanup_cuda()

# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

# Stack the two activations into a 2x65 array for plotting
activations = np.vstack([L40_9682_activation_circuit, L40_9682_activation_circuit_ablated])

# Slice the last 40 tokens and their activations
activations = activations[:, -45:]  # Selecting the last 40 tokens
token_labels = model.to_str_tokens(corr_prompts[0])[-45:] 

# Create a custom colormap: White for zero, then viridis for the rest
cmap = sns.color_palette("viridis", as_cmap=True)
cmap_with_white = ListedColormap(["white"] + list(cmap(np.linspace(0.1, 1, 256))))

# Set up the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    activations,
    annot=False,
    xticklabels=token_labels,
    yticklabels=['Circuit', 'Ablated Duplicate Latents'],
    cmap=cmap_with_white,
    cbar_kws={'label': 'Activation Level'},
    mask=(activations == 0)  # Mask zero values to ensure they are white
)

# Add titles and labels for clarity
plt.title("Effect of ablating duplicate token latents on \n Activation for '1 is next by induction'", fontsize=16, weight='bold')
plt.xlabel("Tokens", fontsize=14)
plt.ylabel("Condition", fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# %%

# Step 1: Define duplicate latents and create filtered mask
duplicate_latents = cluster_results['blocks.7.hook_resid_post']['sum_clusters']['0']

filtered_mask = mask.copy()
filtered_mask['blocks.7.hook_resid_post'] = list(set(mask['blocks.7.hook_resid_post']) - set(duplicate_latents))

# Step 2: Compute mean activation for full circuit without ablation
model.reset_hooks()
_, cache = run_with_saes_zero_ablation_cache(corr_tokens, filtered_ids, model, saes, mask)
L40_9682_activation_circuit = cache['blocks.40.hook_resid_post'][:, -3:, 9682].mean().item()
print("L40_9682_activation_circuit:", L40_9682_activation_circuit)

# Step 3: Compute mean activation for circuit with duplicate token latents ablated
model.reset_hooks()
_, cache = run_with_saes_zero_ablation_cache(corr_tokens, filtered_ids, model, saes, filtered_mask)
L40_9682_activation_circuit_ablated = cache['blocks.40.hook_resid_post'][:, -3:, 9682].mean().item()
print("L40_9682_activation_circuit_ablated:", L40_9682_activation_circuit_ablated)

# Step 4: Compute activations for Model + SAE (Full circuit without ablation)
model.reset_hooks()
_, cache = run_with_saes_zero_ablation_cache(corr_tokens, filtered_ids, model, saes, {})
model_sae_activation = cache['blocks.40.hook_resid_post'][:, -3:, 9682].mean().item()
print("Model + SAE Activation:", model_sae_activation)

# Step 5: Compute activations for Model + SAE with duplicate token latents ablated
fully_filtered_mask = {'blocks.7.hook_resid_post': list(set(range(saes[0].cfg.d_sae)) - set(duplicate_latents))}
model.reset_hooks()
_, cache = run_with_saes_zero_ablation_cache(corr_tokens, filtered_ids, model, saes, fully_filtered_mask)
model_sae_activation_ablated = cache['blocks.40.hook_resid_post'][:, -3:, 9682].mean().item()
print("Model + SAE Activation (Ablated):", model_sae_activation_ablated)

# Cleanup after computation
del cache
cleanup_cuda()

# %%

import matplotlib.pyplot as plt
import numpy as np

# Define the labels and values
conditions = ['Model + SAE', 'Zero-Ablated Circuit']
ablated_values = [model_sae_activation_ablated, L40_9682_activation_circuit_ablated]
non_ablated_values = [model_sae_activation, L40_9682_activation_circuit]

# Bar positions
x = np.arange(len(conditions))

# Width for each bar in a group
bar_width = 0.35

# Plotting
plt.figure(figsize=(8, 6))
bars1 = plt.bar(x - bar_width / 2, ablated_values, width=bar_width, label='Duplicate Latents Ablated (only 2)', color='#4C72B0')
bars2 = plt.bar(x + bar_width / 2, non_ablated_values, width=bar_width, label='Duplicate Latents Not Ablated', color='#55A868')

# Title and labels
plt.title("Effect of ablating duplicate token latents on \n Mean Activation for L40_9682 for error free code \n (Last 3 Tokens)", fontsize=14, weight='bold')
plt.xlabel("Conditions", fontsize=14)
plt.ylabel("Mean Activation Value", fontsize=14)

# Set x-axis tick positions and labels
plt.xticks(x, conditions, fontsize=12)

# Adding legend
plt.legend(fontsize=12)

# Display the values on top of the bars
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + 0.01, 
            f"{yval:.4f}", 
            ha='center', 
            va='bottom', 
            fontsize=11
        )

plt.tight_layout()
plt.show()


# %% H6.2 Ablating the duplicate detectors on the key for error free prompt, causes increase in Traceback probs 

# Step 1: Define duplicate latents and create filtered mask
duplicate_latents = cluster_results['blocks.7.hook_resid_post']['sum_clusters']['0']

# filtered_mask = mask.copy()
# filtered_mask['blocks.7.hook_resid_post'] = list(set(mask['blocks.7.hook_resid_post']) - set( duplicate_latents))
token_pos = [-3, -2, -1]

empty_mask = {}
for key in mask.keys():
    empty_mask[key] = []

model.reset_hooks()
logits, _ = run_with_saes_zero_ablation_cache_tokens_separate(corr_tokens, filtered_ids, model, saes, mask, empty_mask, token_pos)
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, traceback_token]
circuit_full = log_probs.mean().detach().cpu().item()
print("Duplicate cluster ablated, all latents: ",circuit_full)

fully_filtered_mask = {}
for key in mask.keys():
    fully_filtered_mask[key] = []
fully_filtered_mask['blocks.7.hook_resid_post'] = list(set(range(saes[0].cfg.d_sae)) - set(duplicate_latents))
model.reset_hooks()
logits, _ = run_with_saes_zero_ablation_cache_tokens_separate(corr_tokens, filtered_ids, model, saes, mask, fully_filtered_mask, token_pos)
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, traceback_token]
circuit_ablated = log_probs.mean().detach().cpu().item()
print("Duplicate cluster ablated, all latents: ", circuit_ablated)

del logits, log_probs
cleanup_cuda()


# %%

fully_filtered_mask


# %%
# Step 1: Define duplicate latents and create filtered mask
duplicate_latents = cluster_results['blocks.7.hook_resid_post']['sum_clusters']['0']

model.reset_hooks()
logits, _ = run_with_saes_zero_ablation_cache_tokens(corr_tokens, filtered_ids, model, saes, {}, [])
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, traceback_token]
model_full = log_probs.mean().detach().cpu().item()
print("Duplicate cluster ablated, all latents: ",model_full)

fully_filtered_mask = {'blocks.7.hook_resid_post': list(set(range(saes[0].cfg.d_sae)) - set(duplicate_latents))}
model.reset_hooks()
logits, _ = run_with_saes_zero_ablation_cache_tokens(corr_tokens, filtered_ids, model, saes, fully_filtered_mask, [-3, -2, -1])
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs = log_probs[:, -1, traceback_token]
model_dup_ablated = log_probs.mean().detach().cpu().item()
print("Duplicate cluster ablated, all latents: ",model_dup_ablated)

del logits, log_probs
cleanup_cuda()



# %%

import matplotlib.pyplot as plt
import numpy as np

# Define labels for each set of bars
conditions_group1 = ['Circuit Full', 'Circuit Ablated']
values_group1 = [circuit_full, circuit_ablated]

conditions_group2 = ['Model Full', 'Model Duplicate Latents Ablated']
values_group2 = [model_full, model_dup_ablated]

# Set up subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Bar plot for Group 1
ax1.bar(conditions_group1, values_group1, color=['#4C72B0', '#55A868'])
ax1.set_title("Effect of ablating the Duplicate token latents at the Key token on the \nprobability of Traceback for error free code (Zero ablated Circuit)", fontsize=16, weight='bold')
ax1.set_ylabel("Probability of Traceback", fontsize=14)
for i, v in enumerate(values_group1):
    ax1.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=12)

# Bar plot for Group 2
ax2.bar(conditions_group2, values_group2, color=['#C44E52', '#8172B2'])
ax2.set_title("Effect of ablating the Duplicate token latents at the Key token on the \nprobability of Traceback for error free code (Model + SAEs)", fontsize=16, weight='bold')
ax2.set_ylabel("Probability of Traceback", fontsize=14)
for i, v in enumerate(values_group2):
    ax2.text(i, v , f"{v:.4f}", ha='center', va='bottom', fontsize=12)

# Improve layout and display the plots
plt.tight_layout()
plt.show()



# %% steer the duplicate tokens to predict 1 for clean distribution 


def steering_hook(
    activations,
    hook,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
):
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx]

# L7.9681, L14.14967,
_steering_hook = partial(
        steering_hook,
        sae=saes[0],
        latent_idx=10768,
        steering_coefficient=80,
    )
# model.add_sae(sae)
model.add_hook(saes[0].cfg.hook_name, _steering_hook, "fwd")
# model.add_hook(saes[0].cfg.hook_name, _steering_hook2, "fwd")
with torch.no_grad():
    logits = model(clean_tokens)
model.reset_hooks()
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs_answer = log_probs[:, -1, answer_token]
log_probs_traceback = log_probs[:, -1, traceback_token]
print("Log probability of 1: ", log_probs_answer.mean().item())
print("Log probability of Traceback: ", log_probs_traceback.mean().item())

# %%
import torch
import matplotlib.pyplot as plt
from functools import partial

# Range of steering coefficients to sweep over
coeff_values = range(-10, 111, 20)
top_k = 5  # Number of top tokens to track

# Data structure to store top token probabilities for each coefficient
log_probs_data = {}

# Iterate over each steering coefficient
for coeff in coeff_values:
    # Define the steering hook with the current coefficient
    _steering_hook = partial(
        steering_hook,
        sae=saes[0],
        latent_idx=10768,
        steering_coefficient=coeff,
    )
    
    # Add the steering hook to the model
    model.add_hook(saes[0].cfg.hook_name, _steering_hook, "fwd")
    
    # Run the model with no gradient
    with torch.no_grad():
        logits = model(clean_tokens)
    
    # Reset hooks after running
    model.reset_hooks()
    
    # Calculate log probabilities and mean over batch dimension
    log_probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)
    log_probs_last = log_probs[-1, :]  # Log probs of the last token position

    # Get the top 5 tokens and their log probabilities
    top_probs, top_tokens = torch.topk(log_probs_last, top_k)
    
    # Store the token IDs and log probabilities for this coefficient
    log_probs_data[coeff] = {
        "tokens": top_tokens.tolist(),
        "log_probs": top_probs.tolist()
    }
print(log_probs_data)
# %%
    
# Prepare the plot
plt.figure(figsize=(12, 8))

# Map each token to its log probabilities across different coefficients
token_lines = {}

for coeff in coeff_values:
    tokens = log_probs_data[coeff]["tokens"]
    probs = log_probs_data[coeff]["log_probs"]
    
    # Plot each point for the current coefficient
    for i, token_id in enumerate(tokens):
        # Decode the token ID to text
        token_text = model.tokenizer.decode([token_id])
        
        # Initialize a line for each unique token if not already tracked
        if token_id not in token_lines:
            token_lines[token_id] = {
                "coeffs": [],
                "probs": [],
                "label": token_text
            }
        
        # Append the coefficient and log probability for this token
        token_lines[token_id]["coeffs"].append(coeff)
        token_lines[token_id]["probs"].append(probs[i])

# Plot each token's line across coefficients
for token_id, data in token_lines.items():
    plt.plot(data["coeffs"], data["probs"], marker='o', label=data["label"])

# Plot settings
plt.title("Effect of steering the duplicate token latent on \nprobabilities of Top Tokens for Key error code", fontsize=16, weight='bold')
plt.xlabel("Steering Coefficient", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(title="Token", fontsize=12, loc="upper right")
plt.grid(True)
plt.show()

# %%
# Run the model without steering to get the base logits and log probabilities
with torch.no_grad():
    base_logits = model(clean_tokens)
    base_log_probs = base_logits.softmax(dim=-1)
    #torch.log_softmax(base_logits, dim=-1)
print(base_log_probs.shape)
# Get the top 5 tokens from the base log probabilities
top_5_token_indices = torch.topk(base_log_probs[1, -1], 5).indices  # Assuming single batch, last token position

print(f"Top 5 token indices: {top_5_token_indices}")
print(model.tokenizer.decode(top_5_token_indices))




# %% Hypothesis 4: Closing square bracket predictors, after dictionary definition, set the sub task of keying into a dictionary 



index_helper_latents = {'blocks.7.hook_resid_post': [7008, 14309], 'blocks.14.hook_resid_post': [14864]}

filtered_mask = mask.copy()
filtered_mask['blocks.7.hook_resid_post'] = list(set(mask['blocks.7.hook_resid_post']) - set(index_helper_latents["blocks.7.hook_resid_post"]))
filtered_mask['blocks.14.hook_resid_post'] = list(set(mask['blocks.14.hook_resid_post']) - set(index_helper_latents["blocks.14.hook_resid_post"]))
model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, filtered_mask)

log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs_answer = log_probs[:, -1, answer_token]
log_probs_traceback = log_probs[:, -1, traceback_token]
circuit_dup_ablated_answer = log_probs_answer.mean().detach().cpu().item()
circuit_dup_ablated_traceback = log_probs_traceback.mean().detach().cpu().item()
print("Duplicate cluster ablated: ", circuit_dup_ablated_answer)
print("Duplicate cluster ablated, all latents: ", circuit_dup_ablated_traceback)

model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs_answer = log_probs[:, -1, answer_token]
log_probs_traceback = log_probs[:, -1, traceback_token]
full_circuit_perf = log_probs_answer.mean().detach().cpu().item()
full_circuit_traceback = log_probs_traceback.mean().detach().cpu().item()
print("Full circuit perf: ", full_circuit_perf)
print("Full circuit perf: ", full_circuit_traceback)

del logits, log_probs
cleanup_cuda()

# %%
# Helper function to get top 5 tokens and their mean log probabilities across batch dimension
def get_top_5_tokens(logits, model, tokenizer):
    log_probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)  # Take mean over batch dimension
    log_probs_last = log_probs[-1, :]  # Log probs of the last token position
    top_probs, top_tokens = torch.topk(log_probs_last, 5)
    top_tokens = top_tokens.tolist()
    top_probs = top_probs.tolist()
    top_tokens_decoded = [tokenizer.decode([token]) for token in top_tokens]
    return list(zip(top_tokens_decoded, top_probs))

# Run the model for each condition and extract top 5 tokens
# Full circuit
model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
top_5_full_circuit = get_top_5_tokens(logits, model, model.tokenizer)

# Duplicate cluster ablated
model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, filtered_mask)
top_5_dup_ablated = get_top_5_tokens(logits, model, model.tokenizer)

# Print a side-by-side comparison of the top 5 tokens for each condition
from tabulate import tabulate

# Prepare the data for display
table_data = []
for i in range(5):
    table_data.append([
        top_5_full_circuit[i][0],  # Token from full circuit
        top_5_full_circuit[i][1],  # Probability from full circuit
        top_5_dup_ablated[i][0],   # Token from ablated circuit
        top_5_dup_ablated[i][1]    # Probability from ablated circuit
    ])

# %%
# Display the table
print(tabulate(
    table_data,
    headers=["Token (Full Circuit)", "Probability (Full Circuit)", "Token (Bracket Ablated)", "Probability (Bracket Ablated)"],
    tablefmt="pretty"
))



# %% H2: >>> latents put the tasks as code output prediction

index_helper_latents = {'blocks.7.hook_resid_post': [4287, 11707, 12134]}

filtered_mask = mask.copy()
filtered_mask['blocks.7.hook_resid_post'] = list(set(mask['blocks.7.hook_resid_post']) - set(index_helper_latents["blocks.7.hook_resid_post"]))

model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, filtered_mask)

log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs_answer = log_probs[:, -1, answer_token]
log_probs_traceback = log_probs[:, -1, traceback_token]
circuit_dup_ablated_answer = log_probs_answer.mean().detach().cpu().item()
circuit_dup_ablated_traceback = log_probs_traceback.mean().detach().cpu().item()
print("Duplicate cluster ablated: ", circuit_dup_ablated_answer)
print("Duplicate cluster ablated, all latents: ", circuit_dup_ablated_traceback)

model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
log_probs = torch.nn.functional.softmax(logits, dim=-1)
log_probs_answer = log_probs[:, -1, answer_token]
log_probs_traceback = log_probs[:, -1, traceback_token]
full_circuit_perf = log_probs_answer.mean().detach().cpu().item()
full_circuit_traceback = log_probs_traceback.mean().detach().cpu().item()
print("Full circuit perf: ", full_circuit_perf)
print("Full circuit perf: ", full_circuit_traceback)

del logits, log_probs
cleanup_cuda()

# %%


import matplotlib.pyplot as plt
import numpy as np

# Define the labels and values
conditions = ['Circuit', 'Circuit with >> latents ablated']
ablated_values = [full_circuit_perf, circuit_dup_ablated_answer]
non_ablated_values = [full_circuit_traceback,circuit_dup_ablated_traceback]

# Bar positions
x = np.arange(len(conditions))

# Width for each bar in a group
bar_width = 0.35

# Plotting
plt.figure(figsize=(8, 6))
bars1 = plt.bar(x - bar_width / 2, ablated_values, width=bar_width, label='Correct answer', color='#4C72B0')
bars2 = plt.bar(x + bar_width / 2, non_ablated_values, width=bar_width, label='Traceback', color='#55A868')

# Title and labels
plt.title("Effect of ablating >>> token latents on \n Probability of traceback and answer \n for error free code", fontsize=14, weight='bold')
plt.xlabel("Conditions", fontsize=14)
plt.ylabel("Probability", fontsize=14)

# Set x-axis tick positions and labels
plt.xticks(x, conditions, fontsize=12)

# Adding legend
plt.legend(fontsize=12)

# Display the values on top of the bars
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + 0.001, 
            f"{yval:.4f}", 
            ha='center', 
            va='bottom', 
            fontsize=11
        )

plt.tight_layout()
plt.show()




# %%
# Helper function to get top 5 tokens and their mean log probabilities across batch dimension
def get_top_5_tokens(logits, model, tokenizer):
    log_probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)  # Take mean over batch dimension
    log_probs_last = log_probs[-1, :]  # Log probs of the last token position
    top_probs, top_tokens = torch.topk(log_probs_last, 5)
    top_tokens = top_tokens.tolist()
    top_probs = top_probs.tolist()
    top_tokens_decoded = [tokenizer.decode([token]) for token in top_tokens]
    return list(zip(top_tokens_decoded, top_probs))

# Run the model for each condition and extract top 5 tokens
# Full circuit
model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
top_5_full_circuit = get_top_5_tokens(logits, model, model.tokenizer)

# Duplicate cluster ablated
model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, filtered_mask)
top_5_dup_ablated = get_top_5_tokens(logits, model, model.tokenizer)

# Print a side-by-side comparison of the top 5 tokens for each condition
from tabulate import tabulate

# Prepare the data for display
table_data = []
for i in range(5):
    table_data.append([
        top_5_full_circuit[i][0],  # Token from full circuit
        top_5_full_circuit[i][1],  # Probability from full circuit
        top_5_dup_ablated[i][0],   # Token from ablated circuit
        top_5_dup_ablated[i][1]    # Probability from ablated circuit
    ])

# %%
# Display the table
print(tabulate(
    table_data,
    headers=["Token (Full Circuit)", "Probability (Full Circuit)", "Token (Bracket Ablated)", "Probability (Bracket Ablated)"],
    tablefmt="pretty"
))

# %% H3: model has internal representation of dictionary 

import random
import pandas as pd

# Sample names and values
names = [
    "Bob", "Sam", "Lilly", "Rob", "Alice", "Charlie", "Sally", "Tom", "Jake", "Emily", 
    "Megan", "Chris", "Sophia", "James", "Oliver", "Isabella", "Mia", "Jackson", 
    "Emma", "Ava", "Lucas", "Benjamin", "Ethan", "Grace", "Olivia", "Liam", "Noah"
]
# ages = [13, 15, 17, 18, 20]
attributes = [
    "age", "name", "height", "weight", "interest", "gender", "occupation", "nationality",
    "marital_status", "education_level", "income", "languages_spoken", "place_of_birth",
    "eye_color", "hair_color", "blood_type", "allergies", "personality_traits",
    "favorite_color", "favorite_food", "relationship_status", "zodiac_sign",
    "social_media_handles", "hometown", "current_residence", "education_institution",
    "favorite_book", "favorite_movie", "favorite_music_genre", "favorite_sport", "pet_ownership"
]

# Expanded Hobbies List
hobbies = [
    "reading", "swimming", "hiking", "dancing", "coding", "painting", "traveling", "writing",
    "cooking", "gardening", "biking", 
    "yoga", "photography", "crafting", "fishing", "drawing", "knitting", "camping", "scuba_diving",
    "surfing", "volunteering"
]
# Templates for dictionary and non-dictionary structures
dict_template = "{var} = {{{pairs}}}"
list_template = "{var} = [{items}]"
string_template = "{var} = '{string}'"

# Generate a random variable name
def random_var_name():
    return random.choice(attributes)

# Generate a random dictionary structure
def generate_dict():
    var_name = random_var_name()
    pairs = ', '.join(f"'{random.choice(names)}': {random.randint(0, 99)}" for _ in range(3))
    return dict_template.format(var=var_name, pairs=pairs), 1

# Generate a random list or string structure
def generate_non_dict():
    var_name = random_var_name()
    structure_type = random.choice(["list", "string"])
    
    if structure_type == "list":
        items = ', '.join(f"'{random.choice(names)}'" for _ in range(3))
        return list_template.format(var=var_name, items=items), 0
    else:
        string = random.choice(hobbies)
        return string_template.format(var=var_name, string=string), 0

# Generate dataset
def generate_dataset(n_samples):
    data = []
    for _ in range(n_samples):
        if random.random() > 0.5:
            sample, label = generate_dict()
        else:
            sample, label = generate_non_dict()
        data.append({"text": sample, "label": label})
    return pd.DataFrame(data)

# Generate a dataset with N samples
N = 1000
dataset = generate_dataset(N)
print(dataset)

# %%
probe_data = model.to_tokens(list(dataset['text'].values))
print(probe_data.shape)
_, cache = run_with_saes_filtered_cache(probe_data, filtered_ids, model, saes)
relevant_latents = [1976, 6984, 7008, 7323, 10647]
# index the cache['blocks.7.hook_resid_post'] with relevant latents
relevant_cache = cache['blocks.7.hook_resid_post'][:, :, relevant_latents]
relevant_cache.shape

# %%

cleanup_cuda()

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Custom Dataset class to handle text and label data
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Generate batches of tokens and activations
def generate_activations_batch(batch_texts):
    # Convert text to tokens in batch
    probe_data = model.to_tokens(batch_texts).to(device)
    
    # Run model and retrieve cache
    _, cache = run_with_saes_filtered_cache(probe_data, filtered_ids, model, saes)
    
    # Extract relevant activations for specific latents
    relevant_latents = [1976, 6984, 7008, 7323, 10647]
    relevant_cache = cache['blocks.7.hook_resid_post'][:, :, relevant_latents].to(device)
    return relevant_cache  # Shape: (batch_size, seq_len, len(relevant_latents))

# Define the linear probe model
class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        x = x.mean(dim=1)  # Mean pooling over seq_len
        return torch.sigmoid(self.linear(x)).squeeze(-1)

# Initialize dataset, DataLoader, and model
texts = list(dataset['text'].values)
labels = list(dataset['label'].values)
text_dataset = TextDataset(texts, labels)
batch_size = 16  # Set batch size
data_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)

# %%
model.reset_hooks()
input_dim = len(relevant_latents)  # Number of relevant latents
probe = LinearProbe(input_dim).to(device)  # Move model to GPU

# Define loss and optimizer
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(probe.parameters(), lr=0.01)

# Lists to store loss and accuracy for each epoch
loss_history = []
accuracy_history = []

# Training loop with batching and CUDA
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_texts, batch_labels in data_loader:
        # Generate activations for the batch and move them to device
        batch_activations = generate_activations_batch(batch_texts)
        
        # Convert labels to tensor and move to device
        batch_labels = torch.tensor(batch_labels).float().to(device)
        
        # Forward pass
        outputs = probe(batch_activations)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate batch loss
        epoch_loss += loss.item()

        # Calculate accuracy for this batch
        predictions = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
        correct_predictions += (predictions == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = epoch_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions

    # Append to history
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# %%
len(loss_history)

# %%
    
# Plot loss and accuracy curves
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss - Linear probe for detecting dictionaries \nwith mean activations of 5 latents in circuit")
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, len(loss_history) + 1), accuracy_history, marker='o', color='orange', label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy - Linear probe for detecting dictionaries \nwith mean activations of 5 latents in circuit")
plt.legend()

plt.tight_layout()
plt.show()

# %%

# Specify the file path to save the model
model_save_path = "mask_finding/out/hypotheses/dictionary_linear_probe_model.pth"

# Save the model's state_dict (recommended for PyTorch models)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# %% Hypothesis 2: Triple arrow detectors decide that the task is code output prediction





 
