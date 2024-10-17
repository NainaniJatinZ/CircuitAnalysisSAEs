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
# import error_data
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

prompt = """def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output
    
assert f([1, 1, 3, 1, 3, 1]) == [("""
test_prompt(prompt, " TypeError", model)

# %%
model.generate(prompt, max_new_tokens=20)
# %%


prompt = """
def f(n, m):
    arr = list(range(1, n+1))
    for i in range(m):
        arr.clear()
    return arr

# Predict the input of the function that gives this output
assert f(??) == []
Answer: Possible value of ?? is"""
print(model.generate(prompt, max_new_tokens=15))
# test_prompt(prompt, " TypeError", model)
# %%

prompt = """def f(cart):
    while len(cart) > 5:
        cart.popitem()
    return cart

assert f({}) =="""

print(model.generate(prompt, max_new_tokens=15))

# %%
prompt = """def f(name):
    return '*'.join(name.split(' '))

assert f('Fred Smith') =="""
print(model.generate(prompt, max_new_tokens=15))

# %% NICE 

prompt = """def f(value, width):
    if value >= 0:
        return str(value).zfill(width)

    if value < 0:
        return '-' + str(-value).zfill(width)
    return ''

# Predict the output of the function
assert f(5, 3) =="""

print(model.generate(prompt, max_new_tokens=20))
# %%
def f(value, width):
    if value >= 0:
        return str(value).zfill(width)

    if value < 0:
        return '-' + str(-value).zfill(width)
    return ''

f(5, 3)
# %%

prompt = """def f(text):
    s = 0
    for i in range(1, len(text)):
        s += len(text.rpartition(text[i])[0])
    return s
    
# Predict the output of the function
assert f('wdj') =="""

print(model.generate(prompt, max_new_tokens=20))


# %%

def f(text):
    s = 0
    for i in range(1, len(text)):
        s += len(text.rpartition(text[i])[0])
    return s
f('wdj')
# %% NICE 

prompt = """def f(single_digit):
    result = []
    for c in range(1, 11):
        if c != single_digit:
            result.append(c)
    return result
# Predict the output of the function   
output of f(5) is"""
print(model.generate(prompt, max_new_tokens=30))

# %%

prompt = """def f(char):
    if char not in 'aeiouAEIOU':
        return None
    if char in 'AEIOU':
        return char.lower()
    return char.upper()
# Predict the output of the function
output of f('o') is"""
print(model.generate(prompt, max_new_tokens=25))

# %%

prompt = """def f(single_digit):
    result = []
    for c in range(1, 11):
        if c != single_digit:
            result.append(c)
    return result
# Predict the output of the function   
assert f(5) =="""
print(model.generate(prompt, max_new_tokens=25))

# %%

prompt = """def f(single_digit):
    result = []
    for c in range(1, 11):
        if c != single_digit:
            result.append(c)
    return result
# Predict the output of the function   
assert f(5) =="""
print(model.generate(prompt, max_new_tokens=25))

# %%

prompt = """from typing import List, Tuple
def rolling_min(numbers: List[int]) -> List[int]:
    \"\"\" From a given list of integers, generate a list of rolling minimum element found until given moment in the sequence.
    >>> rolling_max([5, 2, 3, 2, 3, 4, 1])
    [5, 2, 2, 2, 2, 2, 1]
    \"\"\"\n"""
print(model.generate(prompt, max_new_tokens=100))
answer = """  
    running_max = None
    result = []\n\n   
    for n in numbers:\n        
        if running_max is None:\n            
            running_max = n\n        
        else:\n            
            running_max = max(running_max, n)\n\n        
        result.append(running_max)\n\n    
    return result\n"""
# %%
