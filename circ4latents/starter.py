# %% Loading libraries and gemma 2 2 imports 
import os 
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
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", device = device, cache_dir = hf_cache)

# %% 

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# TODO: Make this nicer.
df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
df[df['model']=='gemma-2-2b'] # Each row is a "release" which has multiple SAEs which may have different configs / match different hook points in a model. 
import re
from collections import defaultdict

sae_keys = list(df.loc['gemma-scope-2b-pt-res']['saes_map'].keys())
# Dictionary to store the closest string for each layer
closest_strings = {}

# Regular expression to extract the layer number and l0 value
pattern = re.compile(r'layer_(\d+)/width_16k/average_l0_(\d+)')

# Organize strings by layer
layer_dict = defaultdict(list)

for s in sae_keys:
    match = pattern.search(s)
    if match:
        layer = int(match.group(1))
        l0_value = int(match.group(2))
        layer_dict[layer].append((s, l0_value))

# Find the string with l0 value closest to 100 for each layer
for layer, items in layer_dict.items():
    closest_string = min(items, key=lambda x: abs(x[1] - 100))
    closest_strings[layer] = closest_string[0]

# %% Loading saes residual 0-8
layers = [3, 5]
saes = [
    SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id=closest_strings[layer],
        device=str(device)
    )[0]
    for layer in tqdm(layers)
]


# %% Creating clean and corrupted samples 

# import circ4latents.data_gen as dg
n = 10
# pairs = dg.generate_two_templates_pairs(n)
# for clean, corrupted in pairs:
#     print(f"Clean: {clean}\nCorrupted: {corrupted}\n")

# clean_prompts = [pair[0] for pair in pairs]
# corrupted_prompts = [pair[1] for pair in pairs]

# %% Testing latent firing for samples 
# lat_ind = 8566
# layer_ind = 5
# _, cache = model.run_with_cache_with_saes(corrupted, saes=saes[1])
# layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
# cache = cache[layer_name]
# print(cache[0, :, lat_ind])

# %%
# cache[0, :, lat_ind]

# %% Patching metric for lat_ind latent firing 
import random
code_like_names = ["print", "sum", "return", "fetch", "process"]
english_phrases = ["hello", "hi", "ask", "tell", "goodbye"]

# Define list of variable names (with inverted commas)
variable_names = ["'age'", "'score'", "'temperature'", "'value'", "'location'", "'distance'"] #, "''", "''"]

# Function to generate N clean and corrupted pairs based on the two templates
def generate_two_templates_pairs(n):
    pairs = []
    
    for _ in range(n):
        # Randomly pick a code-like function and an English phrase
        code_func = random.choice(code_like_names)
        english_phrase = random.choice(english_phrases)
        
        # Randomly pick a variable name with inverted commas
        var_name = random.choice(variable_names)
        
        # Template 1: code-like vs. non-code-like
        clean_1 = f"{code_func} {var_name}"
        corrupted_1 = f"{english_phrase} {var_name}"
        
        # Template 2: function call-like vs. plain
        clean_2 = f"{english_phrase}({var_name})"
        corrupted_2 = f"{english_phrase} {var_name}"
        
        # Append both pairs to the list
        pairs.append((clean_1, corrupted_1))
        pairs.append((clean_2, corrupted_2))
    
    return pairs

pairs = generate_two_templates_pairs(n)
for clean, corrupted in pairs:
    print(f"Clean: {clean}\nCorrupted: {corrupted}\n")

clean_prompts = [pair[0] for pair in pairs]
corrupted_prompts = [pair[1] for pair in pairs]
lat_ind = 8566
layer_ind = 5

def latent_patch_metric(cache, lat_ind):
    # layer_name = f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'
    result = cache[:, :, lat_ind].sum()
    # print(result.requires_grad)
    return result 
    # return cache[layer_name][:, :, lat_ind].sum()

_, clean_cache = model.run_with_cache_with_saes(clean_prompts, saes=saes[1])
_, corrupted_cache = model.run_with_cache_with_saes(corrupted_prompts, saes=saes[1])
clean_patch = latent_patch_metric(clean_cache[f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'], lat_ind)
corrupted_patch = latent_patch_metric(corrupted_cache[f'blocks.{layer_ind}.hook_resid_post.hook_sae_acts_post'], lat_ind)
print(f"Clean Patch: {clean_patch}, Corrupted Patch: {corrupted_patch}")

def latent_patch_metric_denoising(cache, lat_ind, clean_diff, corr_diff):
    # Make sure the result has requires_grad enabled
    result = (latent_patch_metric(cache, lat_ind) - corr_diff) / (clean_diff - corr_diff)
    assert result.requires_grad, "Result tensor must require gradients"
    return result


# %%
clean_logits, clean_cache = model.run_with_cache(clean_prompts) #, saes=saes[1])
# clean_cache['blocks.5.hook_resid_post'].requires_grad
clean_logits.requires_grad
# %%
# import gc
# del clean_cache, corrupted_cache
# gc.collect()
# # Empty the CUDA cache
# torch.cuda.empty_cache()


# %% Foward and backward caching 
lat_ind = 8566
latent_metric_denoising = partial(latent_patch_metric_denoising, lat_ind=lat_ind, clean_diff=clean_patch, corr_diff=corrupted_patch)

# %%
from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint
# from torchtyping import TensorType as TT

def get_cache_fwd_and_bwd(model: HookedSAETransformer, saes: list[SAE], input, metric, error_term: bool = True, retain_graph: bool = True):
    """
    Get forward and backward caches for a model, including hooks for 'hook_sae_acts_post' and 'hook_sae_error'.
    """
    # Filters to identify relevant hooks
    filter_sae_acts = lambda name: "hook_sae_acts_post" in name
    filter_sae_error = lambda name: "hook_sae_error" in name

    # This hook function will store activations in the appropriate cache
    cache_dict = {"fwd": {}, "bwd": {}}

    def cache_hook(act, hook, dir: Literal["fwd", "bwd"]):
        cache_dict[dir][hook.name] = act.detach()

    with model.saes(saes=saes, use_error_term=error_term):
        # Adding hooks for both 'hook_sae_acts_post' and 'hook_sae_error'
        with model.hooks(
            fwd_hooks=[
                (filter_sae_acts, partial(cache_hook, dir="fwd")),
                (filter_sae_error, partial(cache_hook, dir="fwd"))
            ],
            bwd_hooks=[
                (filter_sae_acts, partial(cache_hook, dir="bwd")),
                (filter_sae_error, partial(cache_hook, dir="bwd"))
            ],
        ):
            _, cache = model.run_with_cache(input)
            relveant_cache = cache['blocks.5.hook_resid_post.hook_sae_acts_post']
            if not relveant_cache.requires_grad:
                relveant_cache.requires_grad_()  # Enable gradients if not already enabled

            # relveant_cache.requires_grad_()
            value = metric(relveant_cache)
            _ = value.backward(retain_graph=retain_graph)
            # value.backward(retain_graph=retain_graph)

    return (
        value,
        ActivationCache(cache_dict["fwd"], model),
        ActivationCache(cache_dict["bwd"], model),
    )


# %%

clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, saes, clean_prompts, latent_metric_denoising
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))

# %%
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, saes, corrupted_prompts, latent_metric_denoising
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))



# %% Attribution calculation

saes[0].cfg

# %% Top K features per sae 

clean_cache

# %% Visualizing features from neuronpedia



# %% Capture Intermediate Output and Gradients
from torch import autograd

def capture_residual_and_compute_grad(model, saes, input_tokens, layer_ind, err_metric_denoising, retain_graph=True):
    """
    Capture the residual stream output of a specific layer (layer_ind) and compute the gradient of every
    component with respect to this intermediate output.
    """
    # Create a cache to store the residual stream output of the desired layer
    residual_cache = {"output": None}

    # Define the hook to capture the residual stream output at the desired layer
    def capture_residual_hook(act, hook):
        residual_cache["output"] = act.clone().detach()  # Store the residual stream output
    
    # Filter to target the specific layer's residual stream
    layer_name = f'blocks.{layer_ind}.hook_resid_post'
    
    # Register the forward hook to capture the residual stream at layer_ind
    with model.hooks([(layer_name, capture_residual_hook)]):
        # Perform forward pass to fill the residual stream cache
        _ = model(input_tokens) # Forward pass with metric

    # Now residual_cache["output"] contains the residual stream output of layer_ind
    residual_output = residual_cache["output"]
    
    # Check that the residual output is captured
    assert residual_output is not None, "Residual stream output not captured!"

    # To compute gradients, we need to ensure residual_output has requires_grad=True
    residual_output.requires_grad_()

    # Now, perform backward pass on the residual_output to compute the gradients
    residual_output_sum = residual_output.sum()  # We can take the sum for gradient accumulation
    residual_output_sum.backward(retain_graph=retain_graph)  # Backpropagate to compute gradients
    
    # The gradients for each preceding component are now stored in the model's parameters
    # You can access gradients of any parameter or intermediate result by accessing `.grad`
    
    # Return the captured residual output and model gradients
    return residual_output, model

# %% Now call this function in your pipeline

# Get the intermediate output and gradients
residual_output, model_with_grads = capture_residual_and_compute_grad(
    model=model,
    saes=saes,
    input_tokens=clean_prompts,
    layer_ind=layer_ind,  # The layer you're interested in
    err_metric_denoising=latent_metric_denoising,
    retain_graph=True  # If you need to keep the graph for further backpropagation
)

print(f"Captured Residual Output at Layer {layer_ind}:", residual_output)


# %%

model_with_grads
# %%
