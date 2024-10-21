# %%
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

model = HookedSAETransformer.from_pretrained("google/gemma-2-9b", device = device, cache_dir = hf_cache)

from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id="layer_10/width_16k/canonical", device=device)


# %%
from transformer_lens.utils import test_prompt
from tasks.error_detection.type.data import generate_samples

selected_templates = [1] #, 2, 3, 4, 5]
N = 20
samples = generate_samples(selected_templates, N)
for sample in samples[0]:
    prompt = sample
    print(prompt)
    # test_prompt(prompt, "Traceback", model, prepend_space_to_answer=False)



# %%

traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
print(traceback_token_id)


# %%
# def logit_diff_error_type(logits, end_positions, err1_tok =type_token_id, err2_tok = syntax_token_id):
#     err1_logits = logits[range(logits.size(0)), end_positions, :][:, err1_tok]
#     err2_logits = logits[range(logits.size(0)), end_positions, :][:, err2_tok]
#     logit_diff = (err1_logits - err2_logits).mean()
#     return logit_diff

def type_error_patch_metric(logits, end_positions, err1_tok=traceback_token_id):
    err1_logits = logits[range(logits.size(0)), end_positions, :][:, err1_tok]
    return err1_logits.mean()
# %% clean

logits = model(samples[0])
attention_mask = model.tokenizer(samples[0]).attention_mask
end_positions = [len(mask) - mask[::-1].index(1) - 1 for mask in attention_mask]
print(end_positions)
print(type_error_patch_metric(logits, end_positions))
del logits
import gc 
gc.collect()
torch.cuda.empty_cache()

# %% corr
logits_corr = model(samples[1])
attention_mask = model.tokenizer(samples[1]).attention_mask
end_positions = [len(mask) - mask[::-1].index(1) - 1 for mask in attention_mask]
print(end_positions)
print(type_error_patch_metric(logits_corr, end_positions))
del logits_corr
gc.collect()
torch.cuda.empty_cache()


# %%




# %%

for stk in model.to_str_tokens(samples[0]):
    print(len(stk))

# %%
    
for stk in model.to_str_tokens(samples[1]):
    print(len(stk))
# %%
end_positions
# %%

test_prompt(samples[1][0], "Traceback", model, prepend_space_to_answer=False)

# %%
