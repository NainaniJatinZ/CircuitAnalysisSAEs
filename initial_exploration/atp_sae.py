# %%

import itertools
import math
import os
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import torch as t
from datasets import IterableDataset, load_dataset
from IPython.display import HTML, IFrame, clear_output, display
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from torchtyping import TensorType as TT

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# %% 
import sys 
sys.path.append("../")
from utils import plot

# %% Loading the models and saes

model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

saes = [
    SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=str(device),
    )[0]
    for layer in tqdm(range(model.cfg.n_layers))
]

# %% Data 

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = t.tensor(
    [
        [model.to_single_token(answers[i][j]) for j in range(2)]
        for i in range(len(answers))
    ],
    device=model.cfg.device,
)
print("Answer token indices", answer_token_indices)

# %%
# %% IOI 
from transformer_lens import patching

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

# %%
CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff


def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )


print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")
# %% base caches 

Metric = Callable[[TT["batch_and_pos_dims", "d_model"]], float]
model.set_use_attn_in(True)
model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)
filter_not_qkv_input = lambda name: "_input" not in name


def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, clean_tokens, ioi_metric
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, corrupted_tokens, ioi_metric
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# %% actual attribution patching attempt

def attr_patch_residual(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
) -> TT["component", "pos"]:
    clean_residual, residual_labels = clean_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=True
    )
    corrupted_residual = corrupted_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=False
    )
    corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=False
    )
    residual_attr = einops.reduce(
        corrupted_grad_residual * (clean_residual - corrupted_residual),
        "component batch pos d_model -> component pos",
        "sum",
    )
    return residual_attr, residual_labels


residual_attr, residual_labels = attr_patch_residual(
    clean_cache, corrupted_cache, corrupted_grad_cache
)
plot.imshow(
    residual_attr,
    y=residual_labels,
    yaxis="Component",
    xaxis="Position",
    title="Residual Attribution Patching",
)


# %% replicate the above patching for layer 5

clean_resi, resi_labels = clean_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=True
    )
clean_resi.shape

# %%

clean_residual = clean_cache[("resid_pre", 5)]
corr_residual = corrupted_cache[("resid_pre",5)]
corr_grad_residual = corrupted_grad_cache[("resid_pre", 5)]
residual_attr_final = einops.reduce(
    corr_grad_residual * (clean_residual - corr_residual),
    "batch pos d_model -> pos",
    "sum",
)
residual_attr_final.shape
# %%
print(residual_attr[10, -1])
print(residual_attr_final[-1])

# %%

saes[0]

# %% cache with saes 


def get_cache_fwd_and_bwd_sae(model, sae, tokens, metric, use_error_term=True):
    model.reset_hooks(including_permanent=True)
    model.reset_saes()
    sae.reset_hooks()
    cache = {}
    model.add_sae(sae)
    sae.use_error_term = use_error_term
    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


clean_value, sae_clean_cache, _ = get_cache_fwd_and_bwd_sae(
    model, saes[5], clean_tokens, ioi_metric
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(sae_clean_cache))
corrupted_value, sae_corrupted_cache, sae_corrupted_grad_cache = get_cache_fwd_and_bwd_sae(
    model, saes[5], corrupted_tokens, ioi_metric
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(sae_corrupted_cache))
print("Corrupted Gradients Cached:", len(sae_corrupted_grad_cache))
# %%

hook_point = saes[5].cfg.hook_name + '.hook_sae_acts_post'
sae_clean_cache[hook_point].shape
# %%
clean_residual = sae_clean_cache[hook_point]
corr_residual = sae_corrupted_cache[hook_point]
corr_grad_residual = sae_corrupted_grad_cache[hook_point]
residual_attr_final = einops.reduce(
    corr_grad_residual * (clean_residual - corr_residual),
    "batch pos n_features -> pos n_features",
    "sum",
)
residual_attr_final.shape

# %%
top_feats = t.topk(residual_attr_final[10,:], 50)
top_feats.indices
# %%

residual_attr_final[10, top_feats.indices]

# %%

model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[5].reset_hooks()
def patch_with_sae_features_with_hook(model, sae, clean_cache, corr_tokens, patching_metric, use_error_term=True, 
                                feature_list=None, next_token_column=True, progress_bar=True):
    # Initialize batch and sequence size based on the activation store settings

    def patching_hook(corrupted_activation, hook, index, clean_activation, feature_idx):
        corrupted_activation[:, index, feature_idx, ...] = clean_activation[:, index, feature_idx, ...]
        return corrupted_activation
    feature_effects = []
    # for i in range(corr_tokens.shape[1]):
    i = 10
    for feature_ind in feature_list: 

        # current_activation_name = utils.get_act_name("hook_sae_acts_post", layer=0)
        hook_point = sae.cfg.hook_name + '.hook_sae_acts_post'
        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=i,
            clean_activation=clean_cache[hook_point],
            feature_idx=feature_ind
        )

        model.add_sae(sae)
        sae.use_error_term = use_error_term
        # Define the hook point in the model where the ablation hook will be attached
        
        model.add_hook(hook_point, current_hook, "fwd")
        # Run the model with the hooks
        patched_logits, sae_cache = model.run_with_cache(corr_tokens, names_filter=[hook_point])
        patched_metric = patching_metric(patched_logits)
        feature_effects.append(patched_metric.item())
        print(f"patching metric output at token {i}, feature ind {feature_ind}: {patched_metric}")

        model.reset_hooks()
        model.reset_saes()
        sae.reset_hooks()
    return patched_logits, sae_cache, feature_effects

model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[5].reset_hooks()

_, _, feature_effects = patch_with_sae_features_with_hook(model, saes[5], sae_clean_cache, corrupted_tokens, ioi_metric, feature_list=top_feats.indices)
feature_effects

# %%
list1 = top_feats.values.cpu().numpy().tolist()  # Convert tensor to NumPy array, then to list
list2 = [feat_effect for feat_effect in feature_effects]  # Convert tensor to NumPy array, then to list
ids = top_feats.indices.cpu().numpy().tolist()
# list1 = top_feats.values.tolist()
# list2 = feature_effects
# ids = top_feats.indices.tolist()
# %%
import matplotlib.pyplot as plt

# Scatter Plot with Line of Equality
plt.figure(figsize=(10, 6))
plt.scatter(list1, list2, color='blue', alpha=0.5)
plt.plot([min(list1), max(list1)], [min(list1), max(list1)], color='red', linestyle='--')  # Line of equality
plt.xlabel('Attribution patching')
plt.ylabel('Activation patching')
plt.title('Attribution vs Activation patching for gpt2-small-res-jb layer 5 SAEs')
plt.grid(True)
plt.show()

# # Difference Plot (Bar Chart of Differences)
# differences = np.array(list2) - np.array(list1)
# plt.figure(figsize=(10, 6))
# plt.bar(ids, differences, color='purple')
# plt.xlabel('IDs')
# plt.ylabel('Difference (List 2 - List 1)')
# plt.title('Difference between Values of List 2 and List 1')
# plt.xticks(rotation=90)
# plt.grid(axis='y')
# plt.show()

# Side-by-Side Bar Chart
x = np.arange(len(ids))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(14, 7))
plt.bar(x - width/2, list1, width, label='Attribution', color='green')
plt.bar(x + width/2, list2, width, label='Activation', color='orange')

plt.xlabel('IDs')
plt.ylabel('Values')
plt.title('Attribution vs Activation patching for gpt2-small-res-jb layer 5 SAEs')
plt.xticks(x, ids, rotation=90)
plt.legend()
plt.grid(axis='y')
plt.show()





# %%
