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

# %%
from transformer_lens.utils import test_prompt
    # %%

from transformer_lens.utils import test_prompt


# Type Error vs Syntax Error
prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> print("abc" + 20)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
"""
test_prompt(prompt, " TypeError", model)

# %%

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> def print_value():
...     value += 1
...     print(value)
... 
>>> value = 10
>>> print_value()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in print_value
"""
print(model.generate(prompt, max_new_tokens=20, temperature=0.5))
# test_prompt(prompt, " Un", model)


# %%

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> def concatenate_strings(a, b):
...     result = a + b
...     print(result)
... 
>>> concatenate_strings("Hello", 5) 
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in concatenate_strings
"""
test_prompt(prompt, " Syntax", model)
# print(model.generate(prompt, max_new_tokens=40))
# test_prompt(prompt, "UnboundLocalError", model)


# %%

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> print("abc" + 20)
"""
test_prompt(prompt, " TypeError", model)

# %% Bracket Completion

from transformer_lens.utils import test_prompt
prompt = """dict = {1: [2], 
    3: [4],
    5: [6
    """
test_prompt(prompt, " TypeError", model)

# %%

prompt = """Type "help", "copyright", "credits" or "license" for more information.
>>> colors = ["red"]
>>> colors.append("blue")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
"""

test_prompt(prompt, " TypeError", model)

# %%
# print("age: " + "200"


# %% code output pred 


prompt = """def f(s1, s2):
    position = 1
    count = 0
    while position > 0:
        position = s1.find(s2, position)
        count += 1
        position += 1
    return count
assert f('xinyyexyxx', 'xx') =="""
print(model.generate(prompt, max_new_tokens=30))


# %%
text = "Hello"
print(text.append("!"))

# %%

prompt= """person = {1: 2, 3: 4}
print(person[1])
# Output of this code is"""

prompt = """text = "Hello"
print(text.append("!"))
# Output of this code is"""
print(model.to_str_tokens(prompt))
test_prompt(prompt, " KeyError", model)

# %%

prompt ="""person = [1, 2, 3, 4]
print(person[5])
# When this code is executed, Python will raise a"""

test_prompt(prompt, " IndexError", model)

# %%
person = {'name': 'Alice'}
print(person[0])


# %%

prompt = """result = int("25")
print(type(result))
# When this code is executed, output will be:"""

model.generate(prompt, max_new_tokens=20)

# test_prompt(prompt, " ValueError", model)

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


prompt = ">>> [1, 2 ,3 ][0]"
model.generate(prompt, max_new_tokens=30)

# %%
prompt = """predict the output of the following code:
>>> [1, 3, 5][1]
"""
test_prompt(prompt, " Traceback", model)
# model.generate(prompt, max_new_tokens=30)
# %% Error Type Prediction 

prompt = """def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output
# Predict the output of the following code:
assert f([1, 1, 3, 1, 3, 1]) =="""

print(model.generate(prompt, max_new_tokens=30))





# %% Error Detection 






# %% Code ouptut prediction 
