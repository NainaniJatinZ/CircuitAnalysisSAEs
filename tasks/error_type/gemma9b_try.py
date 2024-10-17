# %%

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
import os 
from sae_lens import SAE, HookedSAETransformer
import error_data
sys.path.append("../../utils/")
import plot

with open("../../config.json", 'r') as file:
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

# %%

from transformer_lens.utils import test_prompt

prompt = """print("age: " + "20"
# Python raises a"""
test_prompt(prompt, " TypeError", model)

# %%

prompt = """result = "abc" + 5
# When this code is executed, Python will raise a"""

test_prompt(prompt, " TypeError", model)

# %%
# print("age: " + "200"
# %%
result = int("abc")
# %%


prompt= """person = {'name': 'Alice'}
print(person['age'])
# When this code is executed, Python will raise a"""

prompt ="""my_list = [1, 2, 3]
print(my_list[5])
# When this code is executed, Python will raise a"""

test_prompt(prompt, " KeyError", model)
# %%
person = {'name': 'Alice'}
print(person[0])


# %%

prompt = """result = int(None)
# When this code is executed, Python will raise a"""
test_prompt(prompt, " ValueError", model)

# %%
result = sum("25a")
# %% Candidates 

prompt = """result = int("25a")
# When this code is executed, Python will raise a"""
# ValueError

prompt = """result = sum("25a")
# When this code is executed, Python will raise a"""
# TypeError

prompt = """result = print("25a")
# When this code is executed, Python will raise a"""
# No Error

prompt = """result = int(None)
# When this code is executed, Python will raise a"""
# TypeError

