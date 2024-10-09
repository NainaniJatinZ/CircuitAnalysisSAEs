# %%

# import plotly.express as px

# from nnsight import LanguageModel, util
# from nnsight.tracing.Proxy import Proxy

import sys
import time
from collections import defaultdict
from pathlib import Path
import openai

import circuitsvis as cv
import einops
import numpy as np
import torch as t
from IPython.display import display
from jaxtyping import Float
from nnsight import LanguageModel, CONFIG
from rich import print as rprint
from rich.table import Table
from torch import Tensor

# Hide bunch of info logging messages from nnsight
import logging
logging.disable(sys.maxsize)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)
import sys
sys.path.append('..')
from utils import plot

# %%
# Load gpt2
model = LanguageModel('EleutherAI/gpt-j-6b', device_map='auto', torch_dtype=t.bfloat16)
tokenizer = model.tokenizer

N_HEADS = model.config.n_head
N_LAYERS = model.config.n_layer
D_MODEL = model.config.n_embd
D_HEAD = D_MODEL // N_HEADS

print(f"Number of heads: {N_HEADS}")
print(f"Number of layers: {N_LAYERS}")
print(f"Model dimension: {D_MODEL}")
print(f"Head dimension: {D_HEAD}\n")

print("Entire config: ", model.config)
# %%

# Calling tokenizer returns a dictionary, containing input ids & other data.
# If returned as a tensor, then by default it will have a batch dimension.
print(tokenizer("This must be Thursday", return_tensors="pt"))
tokens = tokenizer("This must be Thursday", return_tensors="pt")["input_ids"]
# Decoding a list of integers, into a concatenated string.
print(tokenizer.decode([40, 1239, 714, 651, 262, 8181, 286, 48971, 12545, 13]))

# Using batch decode, on both 1D and 2D input.
print(tokenizer.batch_decode([4711, 2456, 481, 307, 6626, 510]))
print(tokenizer.batch_decode([[1212, 6827, 481, 307, 1978], [2396, 481, 428, 530]]))

# Split sentence into tokens (note we see the special Ġ character in place of prepended spaces).
print(tokenizer.tokenize("This sentence will be tokenized"))



# %%
REMOTE = True
CONFIG.set_default_api_key("GQCjQgiA3H5VlapR42xN")
prompt = 'The Eiffel Tower is in the city of'

with model.trace(prompt, remote=REMOTE):
    # Save the model's hidden states
    hidden_states = model.transformer.h[-1].output[0].save()

    # Save the model's logit output
    logits = model.lm_head.output[0, -1].save()

# Get the model's logit output, and it's next token prediction
print(f"logits.shape = {logits.value.shape} = (vocab_size,)")
print("Predicted token ID =", predicted_token_id := logits.value.argmax().item())
print(f"Predicted token = {tokenizer.decode(predicted_token_id)!r}")

# Print the shape of the model's residual stream
print(f"\nresid.shape = {hidden_states.value.shape} = (batch_size, seq_len, d_model)")

# %%

with model.trace(prompt, remote=REMOTE):
    # Save the model's hidden states
    hidden_states = model.transformer.h[0].attn.attn_dropout.input.save()
val1 = hidden_states.value #shape
val1.shape
# attention = val1[]
# %%
tokens = model.tokenizer.encode(prompt, return_tensors="pt")#.to(model.device)
cv.attention.attention_patterns(
    tokens=tokens[0],
    attention=val1,
)

# %%

plot.imshow(val1[0, -1, :, :])

# %%
len(tokens[0])


# %%

# Load the word pairs from the text file
with open("antonym_pairs.txt", "r") as f:
    ANTONYM_PAIRS = [line.split() for line in f.readlines()]

# %%
ANTONYM_PAIRS[:10]
# %%
class ICLSequence:
    '''
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    '''
    def __init__(self, word_pairs: list[list[str]]):
        self.word_pairs = word_pairs
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        '''Returns the prompt, which contains all but the second element in the last word pair.'''
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[:-len(self.completion())]

    def completion(self):
        '''Returns the second element in the last word pair (with padded space).'''
        return " " + self.y[-1]

    def __str__(self):
        '''Prints a readable string representation of the prompt & completion (indep of template).'''
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")


word_list = [["hot", "cold"], ["yes", "no"], ["in", "out"], ["up", "down"]]
seq = ICLSequence(word_list)

print("Tuple-representation of the sequence:")
print(seq)
print("\nActual prompt, which will be fed into the model:")
print(seq.prompt())
# %%
class ICLDataset:
    '''
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs:
            list of ICL task, e.g. [["old", "young"], ["top", "bottom"], ...] for the antonym task
        size:
            number of prompts to generate
        n_prepended:
            number of antonym pairs before the single-word ICL task
        bidirectional:
            if True, then we also consider the reversed antonym pairs
        corrupted:
            if True, then the second word in each pair is replaced with a random word
        seed:
            random seed, for consistency & reproducibility
    '''

    def __init__(
        self,
        word_pairs: list[list[str]],
        size: int,
        n_prepended: int,
        bidirectional: bool = True,
        seed: int = 0,
        corrupted: bool = False,
    ):
        assert n_prepended+1 <= len(word_pairs), "Not enough antonym pairs in dataset to create prompt."

        self.word_pairs = word_pairs
        self.word_list = [word for word_pair in word_pairs for word in word_pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random word pairs, and constructing `ICLSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended+1, replace=False)
            # Randomize the order of each word pair (x, y). If not bidirectional, we always have x -> y not y -> x
            random_orders = np.random.choice([1, -1], n_prepended+1)
            if not(bidirectional): random_orders[:] = 1
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            # If corrupted, then replace y with a random word in all (x, y) pairs except the last one
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            seq = ICLSequence(word_pairs)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        '''Creates a corrupted version of the dataset (with same random seed).'''
        return ICLDataset(self.word_pairs, self.size, self.n_prepended, self.bidirectional, corrupted=True, seed=self.seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.seqs[idx]
# %%

dataset = ICLDataset(ANTONYM_PAIRS, size=10, n_prepended=2, corrupted=False)

table = Table("Prompt", "Correct completion")
for seq, completion in zip(dataset.seqs, dataset.completions):
    table.add_row(str(seq), repr(completion))

rprint(table)
# %%
dataset = ICLDataset(ANTONYM_PAIRS, size=10, n_prepended=2, corrupted=True)

table = Table("Prompt", "Correct completion")
for seq, completions in zip(dataset.seqs, dataset.completions):
    table.add_row(str(seq), repr(completions))

rprint(table)
# %%
def calculate_h(model: LanguageModel, dataset: ICLDataset, layer: int = -1) -> tuple[list[str], Tensor]:
    '''
    Averages over the model's hidden representations on each of the prompts in `dataset` at layer `layer`, to produce
    a single vector `h`.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset whose prompts `dataset.prompts` you're extracting the activations from (at the last seq pos)
        layer: int
            the layer you're extracting activations from

    Returns:
        completions: list[str]
            list of the model's next-token predictions (i.e. the strings the model predicts to follow the last token)
        h: Tensor
            average hidden state tensor at final sequence position, of shape (d_model,)
    '''
    with model.trace(dataset.prompts, remote=REMOTE):
    # Save the model's hidden states
        hidden_states = model.transformer.h[layer].output[0][:, -1, :].mean(dim=0).save() 
        lm_heads_out = model.lm_head.output.save()
    print(lm_heads_out.shape)
    top_tokens = lm_heads_out[:, -1, :].argmax(dim=-1)
    print(top_tokens.shape)
    completions = [tokenizer.decode(token.item()) for token in top_tokens]
    return completions, hidden_states.value

# %%     
mean_hidden, completions = calculate_h(model, dataset, layer=-1)
completions

# %%
completions.shape
# %%

import tests

tests.test_calculate_h(calculate_h, model)

# %%


def display_model_completions_on_antonyms(
    model: LanguageModel,
    dataset: ICLDataset,
    completions: list[str],
    num_to_display: int = 20,
) -> None:
    table = Table("Prompt (tuple representation)", "Model's completion\n(green=correct)", "Correct completion", title="Model's antonym completions")

    for i in range(min(len(completions), num_to_display)):

        # Get model's completion, and correct completion
        completion = completions[i]
        correct_completion = dataset.completions[i]
        correct_completion_first_token = model.tokenizer.tokenize(correct_completion)[0].replace('Ġ', ' ')
        seq = dataset.seqs[i]

        # Color code the completion based on whether it's correct
        is_correct = (completion == correct_completion_first_token)
        completion = f"[b green]{repr(completion)}[/]" if is_correct else repr(completion)

        table.add_row(str(seq), completion, repr(correct_completion))

    rprint(table)


# Get uncorrupted dataset
dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=2)

# Getting it from layer 12, as in the description in section 2.1 of paper
model_completions, h = calculate_h(model, dataset, layer=12)

# Displaying the output
display_model_completions_on_antonyms(model, dataset, model_completions)




# %%

def intervene_with_h(
    model: LanguageModel,
    zero_shot_dataset: ICLDataset,
    h: Tensor,
    layer: int,
) -> tuple[list[str], list[str]]:
    '''
    Extracts the vector `h` using previously defined function, and intervenes by adding `h` to the
    residual stream of a set of generated zero-shot prompts.

    Inputs:
        model: the model we're using to generate completions
        zero_shot_dataset: the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        h: the `h`-vector we'll be adding to the residual stream
        layer: the layer we'll be extracting the `h`-vector from

    Returns:
        completions_zero_shot: list of string completions for the zero-shot prompts, without intervention
        completions_intervention: list of string completions for the zero-shot prompts, with h-intervention
    '''
    with model.trace(remote=REMOTE) as tracer:
    # Save the model's hidden states
        with tracer.invoke(zero_shot_dataset.prompts):
            hidden_states = model.transformer.h[layer].output[0].save() 
            wo_lm_heads_out = model.lm_head.output.save()

        with tracer.invoke(zero_shot_dataset.prompts):
            h_inter = model.transformer.h[layer].output[0] + h
            lm_heads_out = model.lm_head.output.save()

    top_tokens_wo = wo_lm_heads_out[:, -1, :].argmax(dim=-1)
    completions_wo = [tokenizer.decode(token.item()) for token in top_tokens_wo]

    top_tokens = lm_heads_out[:, -1, :].argmax(dim=-1)
    completions = [tokenizer.decode(token.item()) for token in top_tokens]
    return completions_wo, completions

tests.test_intervene_with_h(intervene_with_h, model, h, ANTONYM_PAIRS, remote=REMOTE)



# %%
