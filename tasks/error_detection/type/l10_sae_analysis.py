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
sae, cfg_dict, sparsity = SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id="layer_21/width_131k/canonical", device=device)

# %%
from transformer_lens.utils import test_prompt

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> numbers = [67, 51, 79]
>>> result = numbers + [67]
"""
# model.add_sae(sae)
test_prompt(prompt, "Traceback", model, prepend_space_to_answer=False)
# model.reset_saes()


# %%
from transformer_lens.utils import test_prompt

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> print("weight" + "20")
"""
# model.add_sae(sae)
# test_prompt(prompt, "Traceback", model, prepend_space_to_answer=False)
logits = model.run_with_saes(prompt, saes = [sae])
# model.reset_saes()

logits #.shape


# %%

torch.topk(logits[0, -1, :], 5)

# %%




# %%

# def run_with_saes_filtered(mode)

# %%

from tasks.error_detection.type.data import generate_samples

selected_templates = [4] #, 2, 3, 4, 5]
N = 50
samples = generate_samples(selected_templates, N)
for sample in samples[0]:
    prompt = sample
    print(prompt)

# %%
# Regular expression pattern to match the string inside print("...")
pattern = r'print\("([^"]+)"'

# Function to get the first match from each string in the list
def extract_first_match(input_list):
    first_matches = []
    for item in input_list:
        match = re.search(pattern, item)  # Use re.search to get only the first match
        if match:
            first_matches.append(match.group(1))  # Append the matched string to the list
    return first_matches

# Get the first match from each string in the list
first_matches = extract_first_match(samples[0])

# Print the results
for match in first_matches:
    print(match)

# %%
# answer_tokens = torch.concat([
#     model.to_tokens(names, prepend_bos=False).T for names in first_matches
# ])
# answer_tokens
    
no_err_toks = model.to_tokens(first_matches, prepend_bos=False)[:, 0]

# %%

print(len(model.to_str_tokens(samples[0][0])), len(model.to_str_tokens(samples[1][0])))

# %%
# from tasks.error_detection.type.data import generate_samples

# selected_templates = [1] #, 2, 3, 4, 5]
# N = 50
# samples = generate_samples(selected_templates, N)
# for sample in samples[0]:
#     prompt = sample
#     print(prompt)

# Token ID for "Traceback"
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
# %%

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

    # Find positions of the first '("', the first '"' after '("', and the end position
    # pos_open_paren_quote = str_tokens_clean.index(' "')
    # pos_first_quote_after_open = pos_open_paren_quote + str_tokens_clean[pos_open_paren_quote:].index('"') 
    pos_end = len(str_tokens_clean) - 1  # The last position
    print(diff_positions, pos_end)
    # if i>5:
    #     break

    # Return the positions with differences, and the positions found
    # print(diff_positions, pos_open_paren_quote, pos_first_quote_after_open, pos_end)
    # print(str_tokens_clean[pos_first_quote_after_open])
#     selected_pos["s_start"].append(pos_open_paren_quote)
#     selected_pos["s_end"].append(pos_first_quote_after_open)
    selected_pos["i_start"].append(diff_positions[0])
    selected_pos["i_end"].append(diff_positions[-1])
    selected_pos["end"].append(pos_end)

selected_pos

# %%

for param in model.parameters():
    param.requires_grad_(False)


# %%
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]
print(traceback_token_id, trip_arrow_token_id)  
print(model.tokenizer.decode([traceback_token_id]), model.tokenizer.decode([trip_arrow_token_id]))  

# %%
model.add_sae(sae)
with torch.no_grad():
    logits = model(samples[0])
err = logits[range(logits.size(0)), selected_pos["end"], :][:, traceback_token_id]
# no_err = logits[range(logits.size(0)), selected_pos["end"], no_err_toks]
no_err = logits[range(logits.size(0)), selected_pos["end"], :][:, trip_arrow_token_id]
print((err - no_err).mean())
model.reset_saes()

# %%

model.cfg.n_layers



# %%

def type_error_patch_metric_prob(logits, end_positions, err1_tok=traceback_token_id, no_err_tok = trip_arrow_token_id):
    # probs = logits.softmax(dim=-1)
    err1_logits = logits[range(logits.size(0)), end_positions, :][:, err1_tok]
    no_err_logits = logits[range(logits.size(0)), end_positions, :][:, no_err_tok]
    return (err1_logits - no_err_logits).mean()

model.add_sae(sae)
with torch.no_grad():
    logits = model(samples[0])
clean_diff = type_error_patch_metric_prob(logits, selected_pos['end'])
model.reset_saes()
print(clean_diff)
model.add_sae(sae)
with torch.no_grad():
    logits = model(samples[1])
corr_diff = type_error_patch_metric_prob(logits, selected_pos['end'])
print(corr_diff)
model.reset_saes()

# %%

def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = type_error_patch_metric_prob(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=selected_pos['end'])


# %%

# Load the JSON data
with open('tasks/error_detection/type/out/layer10_top_10_features_for_rel_pos_abs.json', 'r') as json_file:
    top_10_features_for_rel_pos = json.load(json_file)
top_10_features_for_rel_pos
# %%
imp_feats = []
for key, items in top_10_features_for_rel_pos.items(): 
    if key in ['i_start', 'i_end']:
        for item in items:
            imp_feats.append(item[0])
imp_feats

# %% Steering 

from transformer_lens.utils import test_prompt

model.add_sae(sae)
test_prompt(samples[1][0], "Traceback", model, prepend_space_to_answer=False)
model.reset_saes()

# %%
model.reset_saes()
model.reset_hooks()
# %%
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

_steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=imp_feats[0],
        steering_coefficient=40,
    )
# model.add_sae(sae)
model.add_hook(sae.cfg.hook_name, _steering_hook, "fwd")
test_prompt(samples[1][0], "Traceback", model, prepend_space_to_answer=False)
model.reset_hooks()
# model.reset_saes()
# sae.reset_hooks()

# %%
test_prompt(samples[1][0], "Traceback", model, prepend_space_to_answer=False)
# %%
