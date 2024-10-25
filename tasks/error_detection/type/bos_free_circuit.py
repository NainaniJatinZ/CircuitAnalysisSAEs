# %%
import os 
import gc
import torch
os.chdir("/work/pi_jensen_umass_edu/jnainani_umass_edu/CircuitAnalysisSAEs")
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload
import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal
from transformer_lens.utils import Slice, SliceInput
import sys 
import functools
import re
from collections import defaultdict
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import json
from sae_lens import SAE, HookedSAETransformer
from utils import plot
from circ4latents import data_gen
# sys.path.append("../../utils/")
with open("config.json", 'r') as file:
    config = json.load(file)
    token = config.get('huggingface_token', None)
os.environ["HF_TOKEN"] = token
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

hf_cache = "/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub"
os.environ["HF_HOME"] = "/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub"


# %%

model = HookedSAETransformer.from_pretrained("google/gemma-2-9b", device=device, cache_dir=hf_cache)

# %%
sae, cfg_dict, sparsity = SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id="layer_21/width_16k/canonical", device=device)

# %%

sae.cfg.hook_name


# %%

from tasks.error_detection.type.data import generate_samples

selected_templates = [2] #, 2, 3, 4, 5]
N = 50
samples = generate_samples(selected_templates, N)
for sample in samples[0]:
    prompt = sample
    print(prompt)
    break

selected_pos  = {
    # "s_start": [],
    # "s_end": [],
    "i_start": [],
    "i_end": [],
    "end": []
}

for i in range(N):
    str_tokens_clean = model.to_str_tokens(samples[0][i])
    str_tokens_corr = model.to_str_tokens(samples[1][i])
    # Find the positions with differences
    diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]
    pos_end = len(str_tokens_clean) - 1  # The last position
    # print(diff_positions, pos_end)
    selected_pos["i_start"].append(diff_positions[0])
    selected_pos["i_end"].append(diff_positions[-1])
    selected_pos["end"].append(pos_end)
selected_pos

# %%
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]

def logit_diff_fn(logits, selected_pos, traceback_token_id=traceback_token_id, trip_arrow_token_id=trip_arrow_token_id):
    err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
    no_err = logits[range(logits.size(0)), selected_pos, :][:, trip_arrow_token_id]
    return (err - no_err).mean()

# %%
for param in model.parameters():
    param.requires_grad_(False)

clean_tokens = model.to_tokens(samples[0])
corr_tokens = model.to_tokens(samples[1])

# clean 
logits = model(clean_tokens)
clean_diff = logit_diff_fn(logits, selected_pos['end'])

# corr
logits = model(corr_tokens)
corr_diff = logit_diff_fn(logits, selected_pos['end'])

print(f"clean_diff: {clean_diff}")
print(f"corr_diff: {corr_diff}")

# %%
import gc 

del logits 
gc.collect()

# %%

def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = logit_diff_fn(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=selected_pos['end'])


# %%

from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint
# from torchtyping import TensorType as TT

def get_cache_fwd_and_bwd(
    model,
    tokens,
    metric,
    sae,
    error_term: bool = True,
    retain_graph: bool = True
):
    # torch.set_grad_enabled(True)
    model.reset_hooks()
    # model.reset_saes()
    cache = {}
    grad_cache = {}
    filter_base_acts = lambda name: "blocks.21.hook_resid_post" in name
    # filter_sae_acts = lambda name: "hook_sae_acts_post" in name

    def forward_cache_hook(act, hook):
        act.requires_grad_(True)
        # act.retain_graph()
        cache[hook.name] = act.detach()

    def backward_cache_hook(grad, hook):
        grad.requires_grad_(True)
        # grad.retain_graph()
        grad_cache[hook.name] = grad.detach()

    # sae.use_error_term = error_term
    # model.add_sae(sae)
    model.add_hook(filter_base_acts, forward_cache_hook, "fwd")
    model.add_hook(filter_base_acts, backward_cache_hook, "bwd")
    # logits = run_with_saes_filtered(tokens, [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id], model, [sae])
    value = metric(model(tokens)) #logits)
    value.backward() #retain_graph=retain_graph)

    model.reset_hooks()
    # model.reset_saes()
    # torch.set_grad_enabled(False)
    return (
        value,
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )

# %%

clean_value, clean_cache, _ = get_cache_fwd_and_bwd(model, clean_tokens, err_metric_denoising, sae)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
# print("Clean Gradients Cached:", len(clean_grad_cache))

corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, corr_tokens, err_metric_denoising, sae)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# %%

clean_cache['blocks.21.hook_resid_post'][:, 1:, :].shape
# %%

full_sae = sae.encode(clean_cache['blocks.21.hook_resid_post'])
bos_free = sae.encode(clean_cache['blocks.21.hook_resid_post'][:, 1:, :])

print(full_sae.shape, bos_free.shape)


# %%

torch.equal(full_sae[:, 1:, :],bos_free)

# %%

sae_acts = sae.encode(clean_cache['blocks.21.hook_resid_post'][:, 1:, :])
sae_acts_corr = sae.encode(corrupted_cache['blocks.21.hook_resid_post'][:, 1:, :])
print(sae_acts.shape, sae_acts_corr.shape)

sae_grad_cache = torch.einsum('bij,kj->bik', corrupted_grad_cache['blocks.21.hook_resid_post'][:, 1:, :], sae.W_dec)
print(sae_grad_cache.shape)

# %% 

selected_pos["i_end"][0]
for idx, val in selected_pos.items():
    print(val)
    break

# %%

top_feats_per_pos = {}
K = 10
for idx, val in selected_pos.items():
    # Get the selected activations and gradients
    clean_residual_selected = sae_acts[:, val[0]-1,:]
    #sae_acts[torch.arange(sae_acts.shape[0]), val, :]
    corr_residual_selected = sae_acts_corr[:, val[0]-1,:]
    #sae_acts_corr[torch.arange(sae_acts_corr.shape[0]), val, :]
    corr_grad_residual_selected = sae_grad_cache[:, val[0]-1,:]
    #sae_grad_cache[torch.arange(sae_grad_cache.shape[0]), val, :]

    # Residual attribution calculation only for the selected positions
    residual_attr_final = einops.reduce(
        corr_grad_residual_selected * (clean_residual_selected - corr_residual_selected),
        "batch n_features -> n_features",
        "sum",
    )

    # Get the top K features based on the absolute values
    abs_residual_attr_final = torch.abs(residual_attr_final)
    top_feats = torch.topk(abs_residual_attr_final, K)
    
    # Retrieve the top indices and the original signed values for these indices
    top_indices = top_feats.indices
    top_values = residual_attr_final[top_indices]  # Use original residual attribution values (with signs)

    # Save the results
    top_feats_per_pos[idx] = (top_indices, top_values)

top_feats_per_pos
# %%
ttl_latent_attr = 0
for key, val in top_feats_per_pos.items():
    print(val[1])
    ttl_latent_attr += val[1].sum()

ttl_latent_attr


# %%
residual_attr_final = einops.reduce(
    corrupted_grad_cache['blocks.21.hook_resid_post'] * (clean_cache['blocks.21.hook_resid_post'] - corrupted_cache['blocks.21.hook_resid_post']),
    "batch pos d_model -> pos",
    "sum",
)
residual_attr_final.sum()
# %%

import torch
import einops
import requests
from bs4 import BeautifulSoup
import re
import json

# Function to get HTML for a specific feature
def get_dashboard_html(sae_release="gemma-2-9b", sae_id="21-gemmascope-res-16k", feature_idx=0):
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)

# Function to scrape the description for a feature
def scrape_description(layer, feature_idx):
    url = get_dashboard_html(sae_release="gemma-2-9b", sae_id=f"{layer}-gemmascope-res-16k", feature_idx=feature_idx)
    response = requests.get(url)
    
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        soup_str = str(soup)

        # Use regex to find the "description" field in the JSON structure
        all_descriptions = re.findall(r'description\\":\\"(.*?)",', soup_str)
        
        if all_descriptions:
            return all_descriptions[-1]  # Return the last description
        else:
            return "No description found."
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"

# %%
layer = 21
top_10_features_for_rel_pos = {}
interesting_keys = list(top_feats_per_pos.keys())
print(interesting_keys)
for key in interesting_keys:
    print(f"Position: {key}")
    indices, values = top_feats_per_pos[key]
    top_10_features_for_rel_pos[key] = []
    for idx, val in zip(indices, values):
        print(f"Feature Index: {idx}, Value: {val}")
        description = scrape_description(layer, idx)
        html_link = get_dashboard_html(sae_release="gemma-2-9b", sae_id="21-gemmascope-res-16k", feature_idx=idx)
        print(description)
        top_10_features_for_rel_pos[key].append((idx.item(), val.item(), description, html_link))

# Save the results to a JSON file
with open('tasks/error_detection/type/out/layer21_top_10_features_for_rel_pos_abs.json', 'w') as json_file:
    json.dump(top_10_features_for_rel_pos, json_file, indent=4)


# %%


42 

7, 14, 21, 28, 35, 40
