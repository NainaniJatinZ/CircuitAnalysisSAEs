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

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)


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

prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
name_pairs = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]

# Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
prompts = [
    prompt.format(name)
    for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1]
]
# Define the answers for each prompt, in the form (correct, incorrect)
answers = [names[::i] for names in name_pairs for i in (1, -1)]
# Define the answer tokens (same shape as the answers)
answer_tokens = t.concat([
    model.to_tokens(names, prepend_bos=False).T for names in answers
])

rprint(prompts)
rprint(answers)
rprint(answer_tokens)

tokens = model.to_tokens(prompts, prepend_bos=True)
# Move the tokens to the GPU
tokens = tokens.to(device)
# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)




# %% IOI 
from transformer_lens import patching

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
) -> Float[Tensor, "*batch"]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # SOLUTION
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

clean_tokens = tokens
# Swap each adjacent pair to get corrupted tokens
indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(tokens))]
corrupted_tokens = clean_tokens[indices]

print(
    "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
    "Corrupted string 0:", model.to_string(corrupted_tokens[0])
)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")


# %%


def ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    corrupted_logit_diff: float = corrupted_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    # SOLUTION
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)


# %%


act_patch_resid_pre = patching.get_act_patch_resid_pre(
    model = model,
    corrupted_tokens = corrupted_tokens,
    clean_cache = clean_cache,
    patching_metric = ioi_metric
)


labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
# %%
import torch
import plotly.express as px

# Assuming utils.to_numpy is defined elsewhere, which converts tensor to numpy array.

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis",
    "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid",
    "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "width", "height"
}

def imshow(tensor, renderer=None, x=None, y=None, labels=None, title=None, **kwargs):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    
    # Split the kwargs into update_layout and general kwargs
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    
    # Handle the default color scale
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"

    # Extract label values for x and y axes
    if labels:
        if isinstance(labels, dict):
            if "x" in labels:
                kwargs_post["xaxis_title"] = labels["x"]
            if "y" in labels:
                kwargs_post["yaxis_title"] = labels["y"]

    # Allow direct x and y axis labels overrides
    if x:
        kwargs_post["xaxis_title"] = x
    if y:
        kwargs_post["yaxis_title"] = y

    # Create figure using px.imshow and update layout based on additional settings
    fig = px.imshow(utils.to_numpy(tensor), **kwargs_pre)
    
    # Add title if provided
    if title:
        fig.update_layout(title=title)
    
    # Update layout with other post-creation configurations
    fig.update_layout(**kwargs_post)

    # Handle facet labels if provided
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    
    # Show the figure
    fig.show(renderer)

# Example usage
labels = {"x": "Position", "y": "Layer"}
imshow(
    act_patch_resid_pre,
    labels=labels,  # Using labels to pass axis labels
    title="resid_pre Activation Patching",
    width=600
)


# %%
def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation

# %%
# Defining activation patching functions for a range of common activation patches.
get_act_patch_resid_pre = partial(
    patching.generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_pre",
    index_axis_names=("layer", "pos"),
)

act_patch_resid_pre = get_act_patch_resid_pre(
    model = model,
    corrupted_tokens = corrupted_tokens,
    clean_cache = clean_cache,
    patching_metric = ioi_metric
)

imshow(
    act_patch_resid_pre,
    labels=labels,  # Using labels to pass axis labels
    title="resid_pre Activation Patching",
    width=600
)

# %% SAEs 
model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[0].reset_hooks()
def run_model_and_sae_with_hook(model, sae, tokens, use_error_term=True,
                                feature_list=None, next_token_column=True, progress_bar=True):
    # Initialize batch and sequence size based on the activation store settings
    n_batch, n_seq = tokens.shape
    # Define a hook function applied to SAE latents
    def patching_hook(feature_activations, hook, feature_ids=None, position=None):
        return feature_activations
    # Add the SAE (Supervised AutoEncoder) to the model and configure the error term usage
    model.add_sae(sae)
    sae.use_error_term = use_error_term
    # Define the hook point in the model where the ablation hook will be attached
    hook_point = sae.cfg.hook_name + '.hook_sae_acts_post'
    model.add_hook(hook_point, patching_hook, "fwd")
    # Run the model with the hooks
    patched_logits, sae_cache = model.run_with_cache(tokens, names_filter=[hook_point])
    # Reset hooks to run the model normally
    # model.reset_hooks()
    # sae.reset_hooks()  # Ensure all hooks are properly reset
    # loss, _ = model.run_with_cache(tokens, return_type='loss', loss_per_token=True, names_filter=[])
    # Final cleanup of hooks to reset the model's state
    model.reset_hooks()
    model.reset_saes()
    sae.reset_hooks()
    return patched_logits, sae_cache

sae_logits, sae_cache = run_model_and_sae_with_hook(model, saes[0], clean_tokens)
model.reset_hooks()
base_logits, base_cache = model.run_with_cache(clean_tokens)

sae_logits == base_logits
# %%

sae_cache.keys()
print(sae_cache['blocks.0.hook_resid_pre.hook_sae_acts_post'].shape)

# %%
clean_tokens.shape

# %%
def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation

model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[0].reset_hooks()
def patch_with_sae_with_hook(model, sae, clean_cache, corr_tokens, patching_metric, use_error_term=True, 
                                feature_list=None, next_token_column=True, progress_bar=True):
    # Initialize batch and sequence size based on the activation store settings
    n_batch, n_seq = tokens.shape

    def patching_hook(corrupted_activation, hook, index, clean_activation, feature_idx):
        corrupted_activation[:, index, ...] = clean_activation[:, index, ...]
        return corrupted_activation
    
    for i in range(corr_tokens.shape[1]):

        # current_activation_name = utils.get_act_name("hook_sae_acts_post", layer=0)
        hook_point = sae.cfg.hook_name + '.hook_sae_acts_post'
        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=i,
            clean_activation=clean_cache[hook_point],
        )

        model.add_sae(sae)
        sae.use_error_term = use_error_term
        # Define the hook point in the model where the ablation hook will be attached
        
        model.add_hook(hook_point, current_hook, "fwd")
        # Run the model with the hooks
        patched_logits, sae_cache = model.run_with_cache(corr_tokens, names_filter=[hook_point])

        print(f"patching metric output at token {i}:  {patching_metric(patched_logits)}")
        # Reset hooks to run the model normally
        # model.reset_hooks()
        # sae.reset_hooks()  # Ensure all hooks are properly reset
        # loss, _ = model.run_with_cache(tokens, return_type='loss', loss_per_token=True, names_filter=[])
        # Final cleanup of hooks to reset the model's state
        model.reset_hooks()
        model.reset_saes()
        sae.reset_hooks()
    return patched_logits, sae_cache

model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[0].reset_hooks()

patch_with_sae_with_hook(model, saes[0], sae_cache, corrupted_tokens, ioi_metric)

# %%

model.tokenizer.decode(corrupted_tokens[:,10])
# %%

model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[0].reset_hooks()
def patch_with_sae_features_with_hook(model, sae, clean_cache, corr_tokens, patching_metric, use_error_term=True, 
                                feature_list=None, next_token_column=True, progress_bar=True):
    # Initialize batch and sequence size based on the activation store settings
    n_batch, n_seq = tokens.shape

    def patching_hook(corrupted_activation, hook, index, clean_activation, feature_idx):
        corrupted_activation[:, index, feature_idx, ...] = clean_activation[:, index, feature_idx, ...]
        return corrupted_activation
    
    for i in range(corr_tokens.shape[1]):

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

            print(f"patching metric output at token {i}, feature ind {feature_ind}:  {patching_metric(patched_logits)}")

            model.reset_hooks()
            model.reset_saes()
            sae.reset_hooks()
    return patched_logits, sae_cache

model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[0].reset_hooks()


feature_list = [idx for idx in range(0, 1000, 100)]
patch_with_sae_features_with_hook(model, saes[0], sae_cache, corrupted_tokens, ioi_metric, feature_list=feature_list)

# %% Take a list of feature IDX  for activation patching 

def patch_with_sae_features_list_with_hook(model, sae, clean_cache, corr_tokens, patching_metric, use_error_term=True, 
                                feature_list=None, step_size=None, next_token_column=True, progress_bar=True):
    # Initialize batch and sequence size based on the activation store settings
    n_batch, n_seq = tokens.shape

    def patching_hook(corrupted_activation, hook, index, clean_activation, feature_idx, step_size):
        feat_list = [i for i in range(feature_idx, feature_idx + step_size)]
        corrupted_activation[:, index, feat_list, ...] = clean_activation[:, index, feat_list, ...]
        return corrupted_activation
    
    for i in range(corr_tokens.shape[1]):

        for feature_ind in feature_list: 

            # current_activation_name = utils.get_act_name("hook_sae_acts_post", layer=0)
            hook_point = sae.cfg.hook_name + '.hook_sae_acts_post'
            # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
            current_hook = partial(
                patching_hook,
                index=i,
                clean_activation=clean_cache[hook_point],
                feature_idx=feature_ind, 
                step_size=step_size
            )

            model.add_sae(sae)
            sae.use_error_term = use_error_term
            # Define the hook point in the model where the ablation hook will be attached
            
            model.add_hook(hook_point, current_hook, "fwd")
            # Run the model with the hooks
            patched_logits, sae_cache = model.run_with_cache(corr_tokens, names_filter=[hook_point])

            print(f"patching metric output at token {i}, feature ind {feature_ind}:  {patching_metric(patched_logits)}")

            model.reset_hooks()
            model.reset_saes()
            sae.reset_hooks()
    return patched_logits, sae_cache

# sae_cache["blocks.0.hook_resid_pre.hook_sae_acts_post"][:, 10, [1, 4, 5, 6], ...].shape

model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[0].reset_hooks()


feature_list = [idx for idx in range(0, 24000, 1000)]
patch_logits, patch_cache = patch_with_sae_features_list_with_hook(model, saes[0], sae_cache, corrupted_tokens, ioi_metric, feature_list=feature_list, step_size=1000)

# %%

def patch_with_sae_features_list_with_hook_specific_index(model, sae, clean_cache, corr_tokens, patching_metric, use_error_term=True, 
                                feature_list=None, step_size=None, next_token_column=True, progress_bar=True):
    # Initialize batch and sequence size based on the activation store settings
    n_batch, n_seq = tokens.shape

    def patching_hook(corrupted_activation, hook, index, clean_activation, feature_idx, step_size):
        feat_list = [i for i in range(feature_idx, feature_idx + step_size)]
        corrupted_activation[:, index, feat_list, ...] = clean_activation[:, index, feat_list, ...]
        return corrupted_activation
    
    # for i in range(corr_tokens.shape[1]):

    for feature_ind in feature_list: 

        # current_activation_name = utils.get_act_name("hook_sae_acts_post", layer=0)
        hook_point = sae.cfg.hook_name + '.hook_sae_acts_post'
        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=10,
            clean_activation=clean_cache[hook_point],
            feature_idx=feature_ind, 
            step_size=step_size
        )

        model.add_sae(sae)
        sae.use_error_term = use_error_term
        # Define the hook point in the model where the ablation hook will be attached
        
        model.add_hook(hook_point, current_hook, "fwd")
        # Run the model with the hooks
        patched_logits, sae_cache = model.run_with_cache(corr_tokens, names_filter=[hook_point])

        print(f"patching metric output at token 10 feature ind {feature_ind}:  {patching_metric(patched_logits)}")

        model.reset_hooks()
        model.reset_saes()
        sae.reset_hooks()
    return patched_logits, sae_cache


model.reset_hooks(including_permanent=True)
model.reset_saes()
saes[0].reset_hooks()


feature_list = [idx for idx in range(0, 24000, 1000)]
patch_logits, patch_cache = patch_with_sae_features_list_with_hook_specific_index(model, saes[0], sae_cache, corrupted_tokens, ioi_metric, feature_list=feature_list, step_size=1000)
# %%

feature_list = [idx for idx in range(1000, 2000, 100)]
patch_logits, patch_cache = patch_with_sae_features_list_with_hook_specific_index(model, saes[0], sae_cache, corrupted_tokens, ioi_metric, feature_list=feature_list, step_size=100)


# %%
feature_list = [idx for idx in range(1200, 1300, 10)]
patch_logits, patch_cache = patch_with_sae_features_list_with_hook_specific_index(model, saes[0], sae_cache, corrupted_tokens, ioi_metric, feature_list=feature_list, step_size=10)




# %%
step_size = 1
feature_list = [idx for idx in range(1260, 1270, step_size)]
patch_logits, patch_cache = patch_with_sae_features_list_with_hook_specific_index(model, saes[0], sae_cache, corrupted_tokens, ioi_metric, feature_list=feature_list, step_size=step_size)
# found feature 1269 - Tom feature

# %%
step_size = 1
feature_list = [idx for idx in range(1200, 1210, step_size)]
patch_logits, patch_cache = patch_with_sae_features_list_with_hook_specific_index(model, saes[0], sae_cache, corrupted_tokens, ioi_metric, feature_list=feature_list, step_size=step_size)

# found feature 1208 - John feature


# %%
