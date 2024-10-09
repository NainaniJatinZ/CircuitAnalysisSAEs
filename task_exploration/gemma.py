# %%
import os
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from neel_plotly import line, imshow, scatter
import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload

import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import types
from transformer_lens.utils import Slice, SliceInput

import functools
import re
from collections import defaultdict
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# Imports for displaying vis in Colab / notebook

torch.set_grad_enabled(False)

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

import torch
from collections import defaultdict

# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

import random

# %%

import json
with open("../../config.json", 'r') as file:
    config = json.load(file)
    token = config.get('huggingface_token', None)

os.environ["HF_TOKEN"] = token
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device = device)


# %%

prompt = """
Here is a Python function. Can you identify any syntax errors in it?

def add_numbers(a, b)
    return a + b

The error in the function is"""

# prompt = """
# Check this Python code for errors:

# if (x > 5 and y == 10:
#     print("Condition met")
# The error in the code is"""

prompt = """
my_list = [1, 2, [3, 4]
The type of error in this code is: """

prompt = """
print('Age: ' + 25)
The type of error in the code above is: """

prompt = """What is the type of error in this code?
print(age)"""
print(model.generate(prompt, max_new_tokens=40))

# %%


from transformer_lens.utils import test_prompt

prompt = """
print(age)
# When this code is executed, Python will raise a"""
answer = 'NameError'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)

# %%
prompt = """
print(person['age'])
# When this code is executed, Python will raise a"""

answer = 'KeyError'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)

# %%
prompt = """
abc = [1, 2, 3] + 5 
# When this code is executed, Python will raise a"""
answer = 'TypeError'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)

# %%
prompt = """abc = [1, 2, 3] + [5] 
# When this code is executed, Python will raise a"""

answer = 'TypeError'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)
# %%
prompt = """abc=1
print(abc)
# When this code is executed, Python will raise a"""
answer = 'TypeError'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)

# %%
prompt = """print(abc)
# When this code is executed, Python will raise a"""

answer = 'TypeError'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)
# %%


prompt = """print("age: " + 25)
# When this code is executed, Python will raise a"""
answer = 'Zero'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)

# %%
prompt = """print("age: " + "25"
# When this code is executed, Python will raise a"""

answer = 'TypeError'
# Show that the model can confidently predict the next token.
test_prompt(prompt, answer, model)


# my_list = [1, 2, [3, 4]
# %%

print('Age: ' + 25)
# %%
total = [1, 2, 3] + [5]
# %%
result = 10 / 0
# %%
print(age)
# %%
my_list = [1, 2, 3]
my_list.append(4)
my_list.push(5)
# %%
my_list = [1, 2, 3]
print(my_list[5])
# %%
