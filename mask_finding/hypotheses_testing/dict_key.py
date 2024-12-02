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

# %%
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

# Output the JSON structure

# %%
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

# %% Helpers

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


# def run_with_saes_latent_op_patch(new_tokens, filtered_ids, model, saes, cache, dict_feats):
#     with torch.no_grad():  # Global no_grad for efficiency

#         # Ensure new_tokens is on the correct device
#         device = model.cfg.device
#         new_tokens = torch.as_tensor(new_tokens, device=device)

#         # Vectorized mask creation
#         mask = ~torch.isin(new_tokens, torch.tensor(filtered_ids, device=device))
#         mask_expanded = mask[:, :, None]

#         # Define and add hooks
#         for sae in saes:
#             hook_point = sae.cfg.hook_name
            
#             def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
#                 enc_sae = sae.encode(act).to(device)  # Ensure enc_sae is on the correct device

#                 if hook.name in cache and hook.name in dict_feats:
#                     prev_sae = cache[hook.name].to(device)  # Move cache tensor to device
#                     feature_indices = torch.tensor(dict_feats[hook.name], device=device)  # Move feature indices to device if not already

#                     # Vectorized feature assignment
#                     enc_sae[:, :, feature_indices] = prev_sae[:, :, feature_indices]

#                 modified_act = sae.decode(enc_sae)
#                 updated_act = torch.where(mask_expanded, modified_act, act) if mask_expanded.any() else act
#                 return updated_act
            
#             model.add_hook(hook_point, filtered_hook, dir='fwd')
        
#         logits = model(new_tokens)
#         model.reset_hooks()
        
#     return logits


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

# %% loading the circuit

with open('mask_finding/mask.json') as f:
    mask = json.load(f)
mask

# %% running with saes 
model.reset_hooks()
filtered_ids = [model.tokenizer.bos_token_id]
clean_sae_logits, clean_sae_cache = run_with_saes_filtered_cache(clean_tokens, filtered_ids, model, saes)
corr_sae_logits, corr_sae_cache = run_with_saes_filtered_cache(corr_tokens, filtered_ids, model, saes)

clean_sae_diff = logit_diff_fn(clean_sae_logits)
corr_sae_diff = logit_diff_fn(corr_sae_logits)

print(f"clean_sae_diff: {clean_sae_diff}")
print(f"corr_sae_diff: {corr_sae_diff}")

# %%
model.reset_hooks()
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, mask)
clean_sae_diff_ablation = logit_diff_fn(logits)
print(f"clean_sae_diff_ablation: {clean_sae_diff_ablation}")

logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
corr_sae_diff_ablation = logit_diff_fn(logits)
print(f"corr_sae_diff_ablation: {corr_sae_diff_ablation}")
# %%

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict

# Step 1: Compute Mean Activation Vectors
clean_sae_cache_means = {layer: sae_cache.mean(dim=0) for layer, sae_cache in clean_sae_cache.items()}
corr_sae_cache_means = {layer: sae_cache.mean(dim=0) for layer, sae_cache in corr_sae_cache.items()}

# Step 2: Compute Sum and Difference Vectors
sum_vectors = {}
diff_vectors = {}

for layer, latents in mask.items():
    sum_vectors[layer] = []
    diff_vectors[layer] = []
    
    for latent in latents:
        clean_vector = clean_sae_cache_means[layer][:, latent]
        corr_vector = corr_sae_cache_means[layer][:, latent]
        sum_vectors[layer].append(clean_vector + corr_vector)
        diff_vectors[layer].append(clean_vector - corr_vector)
    
    sum_vectors[layer] = torch.stack(sum_vectors[layer])  # Shape: (n_latents, seq_len)
    diff_vectors[layer] = torch.stack(diff_vectors[layer])  # Shape: (n_latents, seq_len)

# Step 3: Optimal Clustering and Results Storage
results = defaultdict(lambda: {"sum_clusters": {}, "diff_clusters": {}})

def compute_optimal_clusters(vectors, vector_type):
    cos_sim_matrix = cosine_similarity(vectors.cpu().numpy())
    distance_matrix = 1 - cos_sim_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Define cluster search ranges for sum and diff vectors
    if vector_type == "sum":
        cluster_range = range(2, 50, 3)
    else:
        cluster_range = range(2, 8, 1)
    
    best_score, best_n_clusters = -1, 2
    for n_clusters in cluster_range:
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        labels = clustering_model.fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels, metric='precomputed')
        if score > best_score:
            best_score, best_n_clusters = score, n_clusters
    return best_n_clusters

# Define a function for clustering based on optimal clusters
def apply_clustering(vectors, n_clusters):
    cos_sim_matrix = cosine_similarity(vectors.cpu().numpy())
    distance_matrix = 1 - cos_sim_matrix
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return clustering_model.fit_predict(distance_matrix)

# Step 4: Process Each Layer
for layer in mask.keys():
    optimal_clusters = {}

    # Determine optimal clusters for sum and diff vectors
    sum_vectors_layer = sum_vectors[layer]
    diff_vectors_layer = diff_vectors[layer]
    optimal_clusters["sum"] = compute_optimal_clusters(sum_vectors_layer, "sum")
    optimal_clusters["diff"] = compute_optimal_clusters(diff_vectors_layer, "diff")

    # Apply clustering with optimal clusters
    sum_clusters = apply_clustering(sum_vectors_layer, optimal_clusters["sum"])
    diff_clusters = apply_clustering(diff_vectors_layer, optimal_clusters["diff"])

    # Group latents by cluster and store in results JSON structure
    for cluster_type, clusters in zip(["sum_clusters", "diff_clusters"], [sum_clusters, diff_clusters]):
        cluster_latent_indices = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_latent_indices[cluster_id].append(mask[layer][idx])
        results[layer][cluster_type] = {int(k): v for k, v in cluster_latent_indices.items()}

    # Save heatmaps for both sum and diff vectors
    def plot_heatmap(vectors, clusters, title_suffix):
        cluster_means = []
        tokens = model.to_str_tokens(clean_prompts[0])[-25:]  # Select last 25 tokens
        latents_per_cluster = []

        # Calculate mean activation for each cluster
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_mean = vectors[cluster_indices].mean(dim=0).cpu().numpy()[-25:]  # Last 25 tokens
            cluster_means.append(cluster_mean)
            latents_per_cluster.append(len(cluster_indices))

        # Mask zero values for white color in heatmap
        heatmap_data = np.array(cluster_means)
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')
        masked_data = np.ma.masked_where(heatmap_data == 0, heatmap_data)

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(masked_data, aspect='auto', cmap=cmap, interpolation='nearest')
        plt.colorbar(label='Mean Activation')
        plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90, fontsize=8)
        plt.yticks(ticks=range(len(cluster_means)), labels=[f'Cluster {i} - {latents_per_cluster[i]} latents' for i in range(len(cluster_means))])

        # Add lines to separate rows and columns
        for row in range(1, len(cluster_means)):
            plt.hlines(row - 0.5, xmin=-0.5, xmax=len(tokens) - 0.5, color='gray', linewidth=0.5)
        for col in range(1, len(tokens)):
            plt.vlines(col - 0.5, ymin=-0.5, ymax=len(cluster_means) - 0.5, color='gray', linewidth=0.5)

        plt.title(f'{title_suffix} Activation for Layer: {layer}')
        plt.xlabel('Tokens')
        plt.ylabel('Cluster ID')
        plt.tight_layout()
        plt.savefig(f"mask_finding/out/clustering/{layer}_{title_suffix}_heatmap.png")
        plt.show()

    plot_heatmap(sum_vectors_layer, sum_clusters, "Sum")
    plot_heatmap(diff_vectors_layer, diff_clusters, "Difference")

# Step 5: Save clustering results to JSON
with open('mask_finding/out/clustering/clustered_latents.json', 'w') as f:
    json.dump(results, f, indent=2)

print("End-to-end clustering and visualization completed. JSON and heatmaps saved.")


# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Placeholder to store all drops for each clustering type
drop_results = {
    "sum_clusters": [],
    "diff_clusters": []
}

# Helper function to calculate diff_ablation
def calculate_diff_ablation(tokens, mask):
    model.reset_hooks()
    logits = run_with_saes_zero_ablation(tokens, filtered_ids, model, saes, mask)
    return logit_diff_fn(logits)
# Step 1: Calculate the total number of iterations for the progress bar
total_iterations = sum(
    len(clusters_data[layer][clustering_type])
    for clustering_type in ["sum_clusters", "diff_clusters"]
    for layer in results.keys()
)

# Step 2: Initialize a single tqdm progress bar with the total count
with tqdm(total=total_iterations, desc="Processing All Clusters", leave=True) as pbar:
    for clustering_type in ["sum_clusters", "diff_clusters"]:
        clusters_data = results 
        for layer in clusters_data.keys():

            for cluster_id, latents in clusters_data[layer][clustering_type].items():
                # Create new mask by excluding the latents in the current cluster
                new_mask = {k: [v for v in latents if v not in clusters_data[layer][clustering_type][cluster_id]]
                            for k, latents in mask.items()}

                # Compute diff ablation with the modified mask
                new_clean_sae_diff_ablation = calculate_diff_ablation(clean_tokens, new_mask)
                new_corr_sae_diff_ablation = calculate_diff_ablation(corr_tokens, new_mask)

                drop_clean = abs(clean_sae_diff_ablation - new_clean_sae_diff_ablation)
                drop_corr = abs(corr_sae_diff_ablation - new_corr_sae_diff_ablation)
                drop = drop_clean + drop_corr

                drop_results[clustering_type].append({
                    "layer": layer,
                    "cluster_id": cluster_id,
                    "drop": drop.detach().cpu().item(),
                    "drop_clean": drop_clean.detach().cpu().item(),
                    "drop_corr": drop_corr.detach().cpu().item(),
                    "num_latents": len(latents)
                })
                pbar.update(1)

# %%
                
# save the drop results to a json file
with open('mask_finding/out/clustering/drop_results.json', 'w') as f:
    json.dump(drop_results, f, indent=2)

            
# %%
            
# Step 4: Plot all drops for each clustering type, sorted by drop value
for clustering_type, results in drop_results.items():
    # Sort results by drop value
    sorted_results = sorted(results, key=lambda x: x["drop"], reverse=True)
    # Prepare data for plotting
    labels = [f'{res["layer"]}, Cluster {res["cluster_id"]} ({res["num_latents"]} latents)' for res in sorted_results]
    drops = [res["drop"] for res in sorted_results]
    
    # Plot the sorted drops
    plt.figure(figsize=(12, 8))
    plt.bar(labels, drops)
    plt.xticks(rotation=90)
    plt.xlabel('Layer, Cluster (Number of Latents)')
    plt.ylabel('Absolute Drop in Diff Ablation')
    plt.title(f'Sorted Drop in Diff Ablation for {clustering_type} (All Clusters)')
    plt.tight_layout()
    plt.show()

# %%
# Step 5: Plot normalized drop by number of latents for each clustering type
for clustering_type, results in drop_results.items():
    # Sort results by normalized drop (drop per latent)
    normalized_results = sorted(results, key=lambda x: x["drop"] / x["num_latents"], reverse=True)
    
    # Prepare data for plotting
    labels = [f'{res["layer"]}, Cluster {res["cluster_id"]} ({res["num_latents"]} latents)' for res in normalized_results]
    normalized_drops = [res["drop"] / res["num_latents"] for res in normalized_results]
    
    # Plot the normalized sorted drops
    plt.figure(figsize=(12, 8))
    plt.bar(labels, normalized_drops)
    plt.xticks(rotation=90)
    plt.xlabel('Layer, Cluster (Number of Latents)')
    plt.ylabel('Normalized Drop (by number of latents)')
    plt.title(f'Sorted Normalized Drop in Diff Ablation per Latent for {clustering_type} (All Clusters)')
    plt.tight_layout()
    plt.show()


# %%
    
# Step 4: Plot all drops for each clustering type, sorted by drop value
for clustering_type, results in drop_results.items():
    # Sort results by drop value
    sorted_results = sorted(results, key=lambda x: x["drop"], reverse=True)
    
    # Prepare data for plotting
    labels = [f'L{int(res["layer"].split(".")[1])}, C{res["cluster_id"]} ({res["num_latents"]})'
              for res in sorted_results]
    drops = [res["drop"] for res in sorted_results]
    
    # Plot the sorted drops
    plt.figure(figsize=(12, 8))
    plt.bar(labels, drops)
    plt.xticks(rotation=90)
    plt.xlabel('Layer, Cluster (Number of Latents)')
    plt.ylabel('Absolute Drop in Diff Ablation')
    plt.title(f'Sorted Drop in Diff Ablation for {clustering_type} (All Clusters)')
    plt.tight_layout()
    plt.show()
# %%


# Step 5: Plot normalized drop by number of latents for each clustering type
for clustering_type, results in drop_results.items():
    # Sort results by normalized drop (drop per latent)
    normalized_results = sorted(results, key=lambda x: x["drop"] / x["num_latents"], reverse=True)
    
    # Prepare data for plotting
    labels = [f'L{int(res["layer"].split(".")[1])}, C{res["cluster_id"]} ({res["num_latents"]})'
              for res in normalized_results]
    normalized_drops = [res["drop"] / res["num_latents"] for res in normalized_results]
    
    # Plot the normalized sorted drops
    plt.figure(figsize=(12, 8))
    plt.bar(labels, normalized_drops)
    plt.xticks(rotation=90)
    plt.xlabel('Layer, Cluster (Number of Latents)')
    plt.ylabel('Normalized Drop (by number of latents)')
    plt.title(f'Sorted Normalized Drop in Diff Ablation per Latent for {clustering_type} (All Clusters)')
    plt.tight_layout()
    plt.show()



# %%
layer_str = 'blocks.40.hook_resid_post'
# extract layer number from str
layer = int(layer_str.split('.')[1])


# %% testing 

new_mask_test = mask.copy()
# new_mask_test[]

# %%
# load clustered latnets as cluster_results 
with open('mask_finding/out/clustering/clustered_latents.json') as f:
    cluster_results = json.load(f)

# %%
len(list(set(mask['blocks.40.hook_resid_post']) -  set(cluster_results['blocks.40.hook_resid_post']['diff_clusters']['2'])))

# %%
cluster_results['blocks.40.hook_resid_post']['diff_clusters']['2']
new_mask_test = mask.copy()
new_mask_test['blocks.40.hook_resid_post'] = list(set(mask['blocks.40.hook_resid_post']) -  set(cluster_results['blocks.40.hook_resid_post']['diff_clusters']['2']))
model.reset_hooks()
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, new_mask_test)
clean_sae_diff_ablation_testing = logit_diff_fn(logits)
print(f"clean_sae_diff_ablation_testing: {clean_sae_diff_ablation_testing}")
model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, new_mask_test)
corr_sae_diff_ablation_testing = logit_diff_fn(logits)
print(f"corr_sae_diff_ablation_testing: {corr_sae_diff_ablation_testing}")

# %%

# Step 1: Identify the lowest 20 clusters in sum_clusters based on normalized drop
normalized_results = sorted(drop_results['sum_clusters'], key=lambda x: x["drop"] / x["num_latents"])

# Get the lowest 20 clusters
lowest_20_clusters = normalized_results[:40]

# Step 2: Create a copy of the original mask and remove latents from the lowest 20 clusters
new_mask_test = mask.copy()
for cluster_info in lowest_20_clusters:
    layer = cluster_info["layer"]
    cluster_id = cluster_info["cluster_id"]
    
    # Get latents to remove for this specific cluster
    latents_to_remove = set(cluster_results[layer]['sum_clusters'][str(cluster_id)])
    
    # Update the mask by removing these latents
    new_mask_test[layer] = list(set(new_mask_test[layer]) - latents_to_remove)

# Step 3: Compute diff ablation for clean and corrupted tokens using the modified mask
model.reset_hooks()
logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, new_mask_test)
clean_sae_diff_ablation_testing = logit_diff_fn(logits)
print(f"clean_sae_diff_ablation_testing: {clean_sae_diff_ablation_testing}")

model.reset_hooks()
logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, new_mask_test)
corr_sae_diff_ablation_testing = logit_diff_fn(logits)
print(f"corr_sae_diff_ablation_testing: {corr_sae_diff_ablation_testing}")

# %%

cluster_results['blocks.7.hook_resid_post']['sum_clusters']['23']



# %%
# calc the total number of latents in lowest_20_clusters
total_latents = 0
for clus in lowest_20_clusters: 
    total_latents += clus['num_latents']
print(total_latents)


# %%

# Step 1: Sort the sum clusters by normalized drop (drop per latent) in descending order
sorted_sum_clusters = sorted(drop_results['sum_clusters'], key=lambda x: x["drop"] / x["num_latents"], reverse=True)

# Step 2: Print the layer, cluster index, and latent indices from highest to lowest
i = 0
ttl_lattss = 0
for cluster_info in sorted_sum_clusters:
    layer = cluster_info["layer"]
    cluster_id = cluster_info["cluster_id"]
    latents = cluster_results[layer]['sum_clusters'][str(cluster_id)]  # Retrieve latent indices for this cluster
    
    # Print information in the desired format
    print(f"Layer: {layer}, Cluster ID: {cluster_id}, Latent Indices: {latents}")
    i+=1
    ttl_lattss += len(latents)
    if cluster_id == 34 and layer == 'blocks.21.hook_resid_post': 
        break

print(f"Total number of latents: {ttl_lattss}")
print(f"Total number of clusters: {i}")

# %%



























# %%
model.reset_hooks()
layer = 'blocks.40.hook_resid_post'
# latent = 3765
filtered_ids = [model.tokenizer.bos_token_id] 
for lat in mask['blocks.40.hook_resid_post']:
    small_dict = {layer: [lat]}
    logits = run_with_saes_latent_op_patch(corr_tokens, filtered_ids, model, saes, clean_sae_cache, small_dict)
    patched_err_metric = logit_diff_fn(logits)
    normalized_metric = (patched_err_metric - corr_sae_diff) / (clean_sae_diff - corr_sae_diff)
    print(f"Error Metric: {normalized_metric}")

# %% denoising patching 
from tqdm import tqdm
total_steps = sum([len(latents) for latents in mask.values()])
denoising_results = {}
with tqdm(total=total_steps, desc="Denoising Progress") as pbar:
    for layer, latents in mask.items():
        # print(f"Layer: {layer}")
        denoising_results[layer] = {}
        for latent in latents:
            # print(f"Latent: {latent}")
            filtered_ids = [model.tokenizer.bos_token_id] 
            small_dict = {layer: [latent]}
            logits = run_with_saes_latent_op_patch(corr_tokens, filtered_ids, model, saes, clean_sae_cache, small_dict)
            patched_err_metric = logit_diff_fn(logits)
            normalized_metric = (patched_err_metric - corr_diff) / (clean_diff - corr_diff)
            # print(f"Error Metric: {normalized_metric}")
            denoising_results[layer][latent] = normalized_metric.detach().cpu().item()
            pbar.update(1)
denoising_results

# %%

# save the denoising results in json
with open('mask_finding/direct_eff_denoising_results.json', 'w') as f:
    json.dump(denoising_results, f)

# %%
    
# read the direct_eff denoising and noising json files 
with open('mask_finding/direct_eff_denoising_results.json') as f:
    denoising_results = json.load(f)
import matplotlib.pyplot as plt  
plt.figure(figsize=(12, 6))

# for key, latent_dict in denoising_results.items():
    # Sort items by value rather than index
key = list(denoising_results.keys())[3]
latent_dict = denoising_results[key]
sorted_latent_items = sorted(latent_dict.items(), key=lambda x: x[1])
sorted_indices, sorted_values = zip(*sorted_latent_items)

# Convert indices to strings for x-axis labels
sorted_indices_str = [str(index) for index in sorted_indices]

# Bar plot with sorted values on y-axis and latent indices as x-axis labels
plt.bar(sorted_indices_str, sorted_values, label=key, alpha=0.7)
    # break

plt.xlabel('Latent Index (Sorted by Value)')
plt.ylabel('Value')
plt.title('Values Sorted by Latent Index for Each Key (Bar Plot)')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.legend()
plt.tight_layout()  # Adjust layout for better fit
plt.show()

# %%

filtered_mask = mask
filtered_mask['blocks.40.hook_resid_post'] = sorted_indices[-50:]

logits = run_with_saes_zero_ablation(clean_tokens, filtered_ids, model, saes, mask)
clean_sae_diff_ablation_fil = logit_diff_fn(logits)
print(f"clean_sae_diff_ablation_fil: {clean_sae_diff_ablation_fil}")

logits = run_with_saes_zero_ablation(corr_tokens, filtered_ids, model, saes, mask)
corr_sae_diff_ablation_fil = logit_diff_fn(logits)
print(f"corr_sae_diff_ablation: {corr_sae_diff_ablation_fil}")

# %%

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Shared function to process a chunk of latents
def process_latent_chunk(layer, latents_chunk, model, saes, clean_sae_cache, filtered_ids, dict_feats, results_dict, device):
    torch.cuda.set_device(device)  # Ensure process uses the correct GPU
    results = {}

    # Batch process within the chunk
    for latent in latents_chunk:
        small_dict = {layer: [latent]}
        logits = run_with_saes_latent_op_patch(corr_tokens, filtered_ids, model, saes, clean_sae_cache, small_dict)
        patched_err_metric = logit_diff_fn(logits)
        normalized_metric = (patched_err_metric - corr_diff) / (clean_diff - corr_diff)
        results[latent] = normalized_metric.detach().cpu().item()

    results_dict[layer] = results

# Main function to split data and launch processes
def run_denoising_multiprocessing(mask, model, saes, clean_sae_cache, dict_feats):
    filtered_ids = torch.tensor([model.tokenizer.bos_token_id], device=model.cfg.device)
    device = model.cfg.device
    manager = mp.Manager()
    denoising_results = manager.dict()  # Shared dict to gather results

    total_steps = sum(len(latents) for latents in mask.values())
    with tqdm(total=total_steps, desc="Denoising Progress") as pbar:

        # Define processes for each layer
        processes = []
        for layer, latents in mask.items():
            # Split latents into chunks
            chunk_size = max(1, len(latents) // mp.cpu_count())  # Adjust chunk size based on CPU count
            latents_chunks = [latents[i:i + chunk_size] for i in range(0, len(latents), chunk_size)]

            # Create a process for each chunk
            for latents_chunk in latents_chunks:
                p = mp.Process(
                    target=process_latent_chunk,
                    args=(layer, latents_chunk, model, saes, clean_sae_cache, filtered_ids, dict_feats, denoising_results, device)
                )
                processes.append(p)
                p.start()

            # Wait for processes to finish
            for p in processes:
                p.join()
                pbar.update(chunk_size)

    return dict(denoising_results)



# %% noising patching
from tqdm import tqdm
total_steps = sum([len(latents) for latents in mask.values()])
noising_results = {}
with tqdm(total=total_steps, desc="Noising Progress") as pbar:
    for layer, latents in mask.items():
        # print(f"Layer: {layer}")
        noising_results[layer] = {}
        for latent in latents:
            model.reset_hooks()
            # print(f"Latent: {latent}")
            filtered_ids = [model.tokenizer.bos_token_id] 
            small_dict = {layer: [latent]}
            logits = run_with_saes_latent_op_patch(clean_tokens, filtered_ids, model, saes, corr_sae_cache, small_dict)
            patched_err_metric = logit_diff_fn(logits)
            normalized_metric = (patched_err_metric - clean_diff) / (corr_diff - clean_diff)
            # print(f"Error Metric: {normalized_metric}")
            noising_results[layer][latent] = normalized_metric.detach().cpu().item()
            pbar.update(1)
noising_results

# %%
# save the denoising results in json
with open('mask_finding/direct_eff_noising_results.json', 'w') as f:
    json.dump(noising_results, f)





# %%

# %%
import matplotlib.pyplot as plt  
plt.figure(figsize=(12, 6))

# for key, latent_dict in denoising_results.items():
    # Sort items by value rather than index
key = list(noising_results.keys())[3]
latent_dict = noising_results[key]
print(key)
print(latent_dict)
sorted_latent_items = sorted(latent_dict.items(), key=lambda x: x[1])
sorted_indices, sorted_values = zip(*sorted_latent_items)

# Calculating new values with the specified formula
new_values = [
    (((old_val * (corr_diff - clean_diff) + clean_diff) - clean_sae_diff) / (corr_sae_diff - clean_sae_diff)).detach().cpu().item()
    for old_val in sorted_values
]
sorted_indices_str = [str(index) for index in sorted_indices]
# Plotting with the new values
plt.figure(figsize=(12, 6))
plt.bar(sorted_indices_str, new_values, label=key, alpha=0.7)

plt.xlabel('Latent Index (Sorted by Value)')
plt.ylabel('Transformed Value')
plt.title('Transformed Values Sorted by Latent Index for Each Key (Bar Plot)')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.legend()
plt.tight_layout()  # Adjust layout for better fit
plt.show()
# %%

print(clean_prompts[0])

# %% second order latents 

# read the direct_eff denoising and noising json files 
with open('mask_finding/direct_eff_denoising_results.json') as f:
    denoising_results = json.load(f)

with open('mask_finding/direct_eff_noising_results.json') as f:
    noising_results = json.load(f)


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

# receiver_layer = 'blocks.28.hook_resid_post'
# receiver_latent = 2102
model.reset_hooks()
def latent_patch_metric(cache, layer_ind=28, lat_ind=2102):
   # layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
   result = cache[f'blocks.{layer_ind}.hook_resid_post'][1:, :, lat_ind].sum()
   # print(result.requires_grad)
   return result


# Initialize a dictionary to store results
top_k_effects = {}

# Loop over each top K latent and treat it as the receiver
for receiver_layer, receiver_latent, _ in top_k_denoising_noising:
    # Parse the layer number of the receiver layer to set a range for prior layers
    receiver_layer_num = int(receiver_layer.split('.')[1])

    # Initialize results storage for the current receiver
    receiver_latent = int(receiver_latent)
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
        sender_layer = f'blocks.{sender_layer_num}.hook_resid_post'
        if sender_layer in mask:
            
            print(f"Sender Layer: {sender_layer}")
            # Iterate through each latent in the sender layer
            for sender_latent in mask[sender_layer]:
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

# %%
receiver_layer = 'blocks.40.hook_resid_post'  # Replace with the actual receiver layer
receiver_latent = 9682
# Extract the sender metrics for the specified receiver
sender_metrics = top_k_effects.get((receiver_layer, receiver_latent), {})

print(sender_metrics)

# %%
import matplotlib.pyplot as plt
# Sort the sender metrics by the denoising metric values
sorted_sender_metrics = sorted(sender_metrics.items(), key=lambda x: x[1]['denoising_metric'])
sorted_senders, sorted_values = zip(*sorted_sender_metrics)

# Extract sender layers/latents and denoising/noising metrics
sorted_senders_str = [f"{int(layer.split('.')[1])}-{latent}" for layer, latent in sorted_senders]
denoising_values = [metric['denoising_metric'] for metric in sorted_values]
noising_values = [metric['noising_metric'] for metric in sorted_values]

# Plotting sorted denoising and noising metrics as bar plots
plt.figure(figsize=(14, 6))

# Plot denoising metrics
plt.bar(sorted_senders_str, denoising_values, label='Denoising Metric', alpha=0.6)
plt.bar(sorted_senders_str, noising_values, label='Noising Metric', alpha=0.6)

plt.xlabel('Sender Layer-Latent')
plt.ylabel('Normalized Metric Value')
plt.title(f'Sorted Denoising and Noising Metrics for Receiver: {receiver_layer}, Latent: {receiver_latent}')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.legend()
plt.tight_layout()  # Adjust layout for better fit
plt.show()
# %%
# Filter out entries where either denoising or noising metric is zero
# Filter out entries where either denoising or noising metric is zero
filtered_sender_metrics = {
    sender: metrics for sender, metrics in sender_metrics.items()
    if metrics['denoising_metric'] != 0 and metrics['noising_metric'] != 0
}

# Sort the filtered sender metrics by the denoising metric values
sorted_sender_metrics = sorted(filtered_sender_metrics.items(), key=lambda x: x[1]['denoising_metric'])
sorted_senders, sorted_values = zip(*sorted_sender_metrics)

# Extract sender layers/latents and denoising/noising metrics
sorted_senders_str = [f"{int(layer.split('.')[1])}-{latent}" for layer, latent in sorted_senders]
denoising_values = [metric['denoising_metric'] for metric in sorted_values]
noising_values = [metric['noising_metric'] for metric in sorted_values]

# Plotting sorted denoising and noising metrics as bar plots
plt.figure(figsize=(18, 6))  # Increase width of the plot for more x-axis space

# Plot denoising metrics
plt.bar(sorted_senders_str, denoising_values, label='Denoising Metric', alpha=0.6)
plt.bar(sorted_senders_str, noising_values, label='Noising Metric', alpha=0.6)

# Customize x-axis font size and other properties for better readability
plt.xlabel('Sender Layer-Latent')
plt.ylabel('Normalized Metric Value')
plt.title(f'Sorted Denoising and Noising Metrics for Receiver: {receiver_layer}, Latent: {receiver_latent}')
plt.xticks(rotation=90, fontsize=8)  # Reduce font size for x-axis labels
plt.legend()
plt.tight_layout()  # Adjust layout for better fit
plt.show()


# %%

sorted_senders[-10:]


# %%

import json


# Recursively convert all tuple keys to strings for JSON serialization
def make_keys_json_serializable(d):
    if isinstance(d, dict):
        return {f"{k[0]}-{k[1]}" if isinstance(k, tuple) else k: make_keys_json_serializable(v) for k, v in d.items()}
    return d

# Convert the dictionary
json_serializable_top_k_effects = make_keys_json_serializable(top_k_effects)

# Save the modified dictionary to JSON
with open('mask_finding/direct_eff_top_k_effects.json', 'w') as f:
    json.dump(json_serializable_top_k_effects, f)


# %%
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# 1. Average the activation cache across the batch dimension for both clean and corrupted
clean_sae_cache_means = {layer: sae_cache.mean(dim=0) for layer, sae_cache in clean_sae_cache.items()}
corr_sae_cache_means = {layer: sae_cache.mean(dim=0) for layer, sae_cache in corr_sae_cache.items()}

# 2. Initialize a dictionary to store the combined vectors for each latent and their clean-corrupted difference
combined_vectors = {}
sum_vectors = {}
diff_vectors = {}

# 3. Compute vectors for each layer and each latent specified in the mask
for layer, latents in mask.items():
    combined_vectors[layer] = []
    diff_vectors[layer] = []
    sum_vectors[layer] = []
    
    for latent in latents:
        # Get the clean and corrupted activations for the specified latent at the given layer
        clean_vector = clean_sae_cache_means[layer][:, latent]
        corr_vector = corr_sae_cache_means[layer][:, latent]
        
        # Stack clean and corrupted vectors to create a 2 * seq_len vector for each latent
        combined_vector = torch.cat([clean_vector, corr_vector], dim=0)
        combined_vectors[layer].append(combined_vector)
        
        # Compute the clean-corrupted difference vector
        diff_vector = clean_vector - corr_vector
        diff_vectors[layer].append(diff_vector)

        # Compute the sum vector
        sum_vector = clean_vector + corr_vector
        sum_vectors[layer].append(sum_vector)

    # Convert lists to tensors
    combined_vectors[layer] = torch.stack(combined_vectors[layer])  # shape: (n_latents, 2 * seq_len)
    diff_vectors[layer] = torch.stack(diff_vectors[layer])          # shape: (n_latents, seq_len)
    sum_vectors[layer] = torch.stack(sum_vectors[layer])          # shape: (n_latents, seq_len)

# 4. Compute cosine similarities between latents using the combined vectors
grouped_latents = {}
for layer in mask.keys():
    # Compute cosine similarity for combined vectors
    cos_sim_matrix = cosine_similarity(combined_vectors[layer].cpu().numpy())
    
    # 5. Apply clustering (Agglomerative Clustering) based on cosine similarity
    clustering_model = AgglomerativeClustering(metric='precomputed', linkage='average')
    latent_groups = clustering_model.fit_predict(1 - cos_sim_matrix)  # 1 - cosine similarity as a distance measure
    
    # Store the group labels for each latent in this layer
    grouped_latents[layer] = latent_groups

# Now `grouped_latents` contains clusters for each latent in each layer based on clean/corrupted vectors.
print(grouped_latents)

# %%
latent_n_clusters = {'blocks.40.hook_resid_post': 35, 'blocks.21.hook_resid_post': 26, 'blocks.14.hook_resid_post': 23, 'blocks.7.hook_resid_post': 38}
grouped_latents_sum = {}
for layer in mask.keys():
    # Compute cosine similarity for combined vectors
    cos_sim_matrix = cosine_similarity(sum_vectors[layer].cpu().numpy())
    
    # 5. Apply clustering (Agglomerative Clustering) based on cosine similarity
    clustering_model = AgglomerativeClustering(n_clusters=latent_n_clusters[layer], metric='precomputed', linkage='average')
    latent_groups = clustering_model.fit_predict(1 - cos_sim_matrix)  # 1 - cosine similarity as a distance measure
    
    # Store the group labels for each latent in this layer
    grouped_latents_sum[layer] = latent_groups

# Now `grouped_latents` contains clusters for each latent in each layer based on clean/corrupted vectors.
print(grouped_latents_sum)

# %%

# %%
latent_n_clusters = {'blocks.40.hook_resid_post': 3, 'blocks.21.hook_resid_post': 2, 'blocks.14.hook_resid_post': 3, 'blocks.7.hook_resid_post': 5}
grouped_latents_diff = {}
for layer in mask.keys():
    # Compute cosine similarity for combined vectors
    cos_sim_matrix = cosine_similarity(diff_vectors[layer].cpu().numpy())
    
    # 5. Apply clustering (Agglomerative Clustering) based on cosine similarity
    clustering_model = AgglomerativeClustering(n_clusters=latent_n_clusters[layer], metric='precomputed', linkage='average')
    latent_groups = clustering_model.fit_predict(1 - cos_sim_matrix)  # 1 - cosine similarity as a distance measure
    
    # Store the group labels for each latent in this layer
    grouped_latents_diff[layer] = latent_groups

# Now `grouped_latents` contains clusters for each latent in each layer based on clean/corrupted vectors.
print(grouped_latents_diff)


# %%

len(mask['blocks.40.hook_resid_post'])
# %%
from sklearn.metrics import silhouette_score
import numpy as np

# Range of cluster numbers to try
cluster_range = range(2, 50, 3)
layer = 'blocks.40.hook_resid_post'
best_score = -1
best_n_clusters = 2

# Compute cosine similarity
cos_sim_matrix = cosine_similarity(sum_vectors[layer].cpu().numpy())
distance_matrix = 1 - cos_sim_matrix  # Convert similarity to distance for clustering

# Ensure the diagonal is zero for silhouette_score
np.fill_diagonal(distance_matrix, 0)

# Test different numbers of clusters and calculate silhouette scores
for n_clusters in cluster_range:
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    latent_groups = clustering_model.fit_predict(distance_matrix)
    
    # Compute silhouette score
    score = silhouette_score(distance_matrix, latent_groups, metric='precomputed')
    print(f"Silhouette Score for {n_clusters} clusters: {score}")
    # Update best score and number of clusters if improved
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

print(f"Optimal number of clusters: {best_n_clusters} with silhouette score: {best_score}")

# %%

# find the index of 9682 for blocks.40.hook_resid_post 
index = mask['blocks.40.hook_resid_post'].index(11103)

# find the cluster of the index in the grouped_latents_diff
cluster = grouped_latents_diff['blocks.40.hook_resid_post'][index]
cluster

# %%

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Define layer and get the combined vectors and diff vectors
layer = 'blocks.40.hook_resid_post'
combined_vectors_layer = combined_vectors[layer]
diff_vectors_layer = diff_vectors[layer]
clusters = grouped_latents_sum[layer]

import matplotlib.pyplot as plt
import numpy as np

# Summarization of Mean Difference Distribution for Each Cluster as Bar Plots
cluster_diffs = {}
for cluster_id in np.unique(clusters):
    # Select diff vectors belonging to the current cluster
    cluster_indices = np.where(clusters == cluster_id)[0]
    cluster_diff_vectors = diff_vectors_layer[cluster_indices]
    
    # Calculate the mean difference across latents in this cluster
    cluster_mean_diff = cluster_diff_vectors.mean(dim=0).cpu().numpy()
    cluster_diffs[cluster_id] = cluster_mean_diff

    # Plot the mean diff distribution for this cluster as a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(cluster_mean_diff)), cluster_mean_diff, label=f'Cluster {cluster_id} Mean Difference')
    plt.title(f'Mean Difference Distribution for Cluster {cluster_id}')
    plt.xlabel('Sequence Length Position')
    plt.ylabel('Mean Difference in Activation')
    plt.legend()
    plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
# import matplotlib.colors as mcolors
# Assuming `model.to_str_tokens(clean_prompt[0])` provides the token labels for the x-axis
layer = 'blocks.40.hook_resid_post'
combined_vectors_layer = combined_vectors[layer]
diff_vectors_layer = diff_vectors[layer]
clusters = grouped_latents_diff[layer]
tokens = model.to_str_tokens(clean_prompts[0])  # Get token labels for sequence length positions

# Summarization of Mean Difference Distribution for Each Cluster as a Heatmap
cluster_diffs = {}
heatmap_data = []
latents_per_cluster = []
for cluster_id in np.unique(clusters):
    # Select diff vectors belonging to the current cluster
    cluster_indices = np.where(clusters == cluster_id)[0]
    cluster_diff_vectors = diff_vectors_layer[cluster_indices]
    latents_per_cluster.append(len(cluster_indices))
    # Calculate the mean difference across latents in this cluster
    cluster_mean_diff = cluster_diff_vectors.mean(dim=0).cpu().numpy()
    cluster_diffs[cluster_id] = cluster_mean_diff
    heatmap_data.append(cluster_mean_diff)  # Collect for heatmap plot

# Convert to numpy array for plotting, with each row as a cluster and columns as sequence length positions
heatmap_data = np.array(heatmap_data)

# Select only the last 5 tokens
heatmap_data = heatmap_data[:, -25:]
tokens = tokens[-25:]

# Create a color map with white for zero values
cmap = plt.cm.viridis
cmap.set_bad(color='white')

# Mask zeros in the heatmap data for white color
masked_heatmap_data = np.ma.masked_where(heatmap_data == 0, heatmap_data)

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(masked_heatmap_data, aspect='auto', cmap=cmap, interpolation='nearest')
plt.colorbar(label='Mean Difference in Activation')

# Set y-axis to show cluster labels and x-axis to show only the last 5 tokens
plt.yticks(ticks=range(len(cluster_diffs)), labels=[f'Cluster {i} - {latents_per_cluster[i]} latents' for i in range(len(cluster_diffs))])
plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
layer_ind = 7
plt.title(f'Mean Difference (clean - corrupted) Distribution for Each Cluster Latents in Layer: {layer_ind}')
plt.xlabel('Last 15 Tokens in Sequence')
plt.ylabel('Cluster ID')
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

# Assuming `model.to_str_tokens(clean_prompt[0])` provides the token labels for the x-axis
layer_ind = 40
layer = f'blocks.{layer_ind}.hook_resid_post'
combined_vectors_layer = combined_vectors[layer]
diff_vectors_layer = diff_vectors[layer]
sum_vectors_layer = sum_vectors[layer]
clusters = grouped_latents_diff[layer]
tokens = model.to_str_tokens(clean_prompts[0])  # Get token labels for sequence length positions

# Summarization of Mean Difference Distribution for Each Cluster as a Heatmap
cluster_sums = {}
heatmap_data = []
latents_per_cluster = []
for cluster_id in np.unique(clusters):
    if cluster_id != 0:
        continue
    # Select diff vectors belonging to the current cluster
    cluster_indices = np.where(clusters == cluster_id)[0]
    cluster_sum_vectors = sum_vectors_layer[cluster_indices]
    latents_per_cluster.append(len(cluster_indices))
    # Calculate the mean difference across latents in this cluster
    cluster_mean_sum = cluster_sum_vectors.mean(dim=0).cpu().numpy()
    cluster_sums[cluster_id] = cluster_mean_sum
    heatmap_data.append(cluster_mean_sum)  # Collect for heatmap plot

# Convert to numpy array for plotting, with each row as a cluster and columns as sequence length positions
heatmap_data = np.array(heatmap_data)

# Create a color map with white for zero values
cmap = plt.cm.viridis
cmap.set_bad(color='white')

# Mask zeros in the heatmap data for white color
masked_heatmap_data = np.ma.masked_where(heatmap_data == 0, heatmap_data)

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(masked_heatmap_data, aspect='auto', cmap=cmap, interpolation='nearest')
plt.colorbar(label='Mean Sum in Activation')

# Set y-axis to show cluster labels and x-axis to show the last 15 tokens
plt.yticks(ticks=range(len(cluster_sums)), labels=[f'Cluster {i} - {latents_per_cluster[i]} latents' for i in range(len(cluster_sums))])
plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90, fontsize=8)  # Reduce x-axis font size to avoid overlap

# Add lines to separate rows and columns
for row in range(1, len(cluster_sums)):
    plt.hlines(row - 0.5, xmin=-0.5, xmax=len(tokens) - 0.5, color='gray', linewidth=0.5)

for col in range(1, len(tokens)):
    plt.vlines(col - 0.5, ymin=-0.5, ymax=len(cluster_sums) - 0.5, color='gray', linewidth=0.5)

# Title and axis labels
plt.title(f'Mean Sum (clean + corrupted) Distribution for Each Cluster Latents in Layer: {layer_ind}')
plt.xlabel('Sequence Tokens')
plt.ylabel('Cluster ID')
plt.tight_layout()
plt.show()




# %%
from collections import defaultdict

# Initialize a dictionary to hold lists of latent indices for each cluster
cluster_latent_indices = defaultdict(list)

# Iterate over each latent index and its assigned cluster
for idx, cluster_id in enumerate(clusters):
    # Append the index of the latent to the appropriate cluster list
    cluster_latent_indices[cluster_id].append(mask[layer][idx])

# Print the latent indices for each cluster
for cluster_id, indices in cluster_latent_indices.items():
    print(f"Cluster {cluster_id}: Latent Indices: {indices}")

# %%

import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

# Load the denoising results
with open('mask_finding/direct_eff_denoising_results.json') as f:
    denoising_results = json.load(f)

# Assign a distinct color for each cluster
num_clusters = len(cluster_latent_indices)
colors = cm.get_cmap('tab10', num_clusters)  # Use 'tab10' colormap for up to 10 clusters

# Choose a key to plot from denoising_results
key = list(denoising_results.keys())[3]
latent_dict = denoising_results[key]

# Sort items by value for plotting
sorted_latent_items = sorted(latent_dict.items(), key=lambda x: x[1])
sorted_indices, sorted_values = zip(*sorted_latent_items)

# Convert indices to strings for x-axis labels
sorted_indices_str = [str(index) for index in sorted_indices]

# Map each latent index to its cluster for coloring
cluster_color_map = {}
for cluster_id, indices in cluster_latent_indices.items():
    for index in indices:
        cluster_color_map[index] = colors(cluster_id)  # Assign color based on cluster

# Plot with color-coding for clusters
plt.figure(figsize=(12, 6))

# Create bars with colors based on cluster membership
bar_colors = [cluster_color_map.get(int(index), 'gray') for index in sorted_indices]  # Default to gray if no cluster

plt.bar(sorted_indices_str, sorted_values, color=bar_colors, alpha=0.7)
plt.xlabel('Latent Index (Sorted by Direct Effect)')
plt.ylabel('Direct Effect to Logits (Denoising)')
plt.title('Direct effect for Latent Index in Layer 40')
plt.xticks(rotation=90)  # Rotate x-axis labels foCreate custom legend for clusters
handles = [plt.Line2D([0], [0], color=colors(cluster_id), lw=4, label=f'Cluster {cluster_id}')
           for cluster_id in range(num_clusters)]
plt.legend(handles=handles, title='Cluster ID')

plt.tight_layout()  # Adjust layout for better fit
plt.show()



# %%

