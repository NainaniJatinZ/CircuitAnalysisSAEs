# %%
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
from functools import partial
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, utils
from transformer_lens.hook_points import HookPoint
# from torchtyping import TensorType as TT

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)
import sys 
sys.path.append("../")
from utils import plot
from utils.ioi_dataset import IOIDataset

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
# %% IOI dataset 
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
# ioi_dataset.ioi_prompts
abc_dataset = (  # TODO seeded
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
    # .gen_flipped_prompts(("IO2", "RAND"))
    )
print(ioi_dataset.sentences[0])
print(abc_dataset.sentences[0])
# %% logit diff and ioi set up

def logits_to_ave_logit_diff_2(
    logits: Float[Tensor, "batch seq d_vocab"],
    ioi_dataset: IOIDataset = ioi_dataset,
    per_prompt=False
) -> Float[Tensor, "*batch"]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def _ioi_metric_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float,
        corrupted_logit_diff: float,
        ioi_dataset: IOIDataset,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
        return ((patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff))

def generate_data_and_caches(ioi_dataset, abc_dataset, verbose: bool = False, seed: int = 42):

    model.reset_hooks(including_permanent=True)

    ioi_logits_original = model(ioi_dataset.toks)
    abc_logits_original = model(abc_dataset.toks)

    ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original, ioi_dataset).item()
    abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original, ioi_dataset).item()

    if verbose:
        print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
        print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

    ioi_metric_noising = partial(
        _ioi_metric_noising,
        clean_logit_diff=ioi_average_logit_diff,
        corrupted_logit_diff=abc_average_logit_diff,
        ioi_dataset=ioi_dataset,
    )

    return ioi_metric_noising

ioi_metric_noising = generate_data_and_caches(ioi_dataset, abc_dataset, verbose=True)

# %%

# Metric = Callable[[TT["batch_and_pos_dims", "d_model"]], float]
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
    # print("Value: ", value)
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, ioi_dataset.toks.long(), ioi_metric_noising
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, abc_dataset.toks.long(), ioi_metric_noising
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))




# %%
import einops
def attr_patch_residual(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
):
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


# %%
clean_residual, residual_labels = clean_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=True
    )
residual_labels
# %%
ioi_dataset.toks.shape

# %%
ioi_dataset.word_idx.keys()
# %%
import torch
import einops

# Assuming residual_attr_dict is to be filled
residual_attr_dict = {}

# Extract clean, corrupted, and gradient residuals
clean_residual, residual_labels = clean_cache.accumulated_resid(
    -1, incl_mid=True, return_labels=True
)
corrupted_residual = corrupted_cache.accumulated_resid(
    -1, incl_mid=True, return_labels=False
)
corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(
    -1, incl_mid=True, return_labels=False
)

# Iterate through each key in the dictionary to calculate the residual attribution
for key in ['IO', 'IO-1', 'IO+1', 'S', 'S-1', 'S+1', 'S2', 'end', 'starts', 'punct']:
    # Get the indices corresponding to this key (one index per batch item)
    indices = torch.tensor(ioi_dataset.word_idx[key], dtype=torch.long)  # Assuming this is a list of length equal to batch size

    # Expand indices to align with the residual dimensions
    indices_expanded = indices.unsqueeze(0).expand(25, -1)  # Shape: [25, batch_size]

    # Select the specific indices from the token dimension (3rd index) using the provided indexing pattern
    selected_corrupted_grad_residual = corrupted_grad_residual[torch.arange(25).unsqueeze(1), torch.arange(100), indices_expanded]
    selected_clean_residual = clean_residual[torch.arange(25).unsqueeze(1), torch.arange(100), indices_expanded]
    selected_corrupted_residual = corrupted_residual[torch.arange(25).unsqueeze(1), torch.arange(100), indices_expanded]

    # Resulting tensors will have shape [25, 100, 768]

    # Perform the operation on the selected values
    residual_attr = einops.reduce(
        selected_corrupted_grad_residual * (selected_clean_residual - selected_corrupted_residual),
        "component batch d_model -> component",
        "sum",
    )

    # Store the result for the current key
    residual_attr_dict[key] = residual_attr

# Verify the shape of one of the keys to confirm
print(residual_attr_dict['IO'].shape)

# %%

import numpy as np

# Convert the residual_attr_dict to a list of numpy arrays
residual_attr_combined = []

# Collect residuals for each key and ensure they are in a form suitable for concatenation
for key in residual_attr_dict.keys():
    residual_attr_combined.append(residual_attr_dict[key].cpu().numpy())  # Convert to numpy if tensor

# Stack these arrays along a new axis to make a combined data matrix for visualization
# Resulting shape: [25, len(keys)] where 25 is the number of components and len(keys) is the number of categories
residual_attr_combined = np.stack(residual_attr_combined, axis=-1)

# Get the labels for the components and x-axis labels
component_labels = ["Component " + str(i) for i in range(residual_attr_combined.shape[0])]
x_labels = list(residual_attr_dict.keys())

# Use imshow function to visualize the combined result
fig = plot.imshow(
    residual_attr_combined,
    y=residual_labels,
    x=x_labels,
    yaxis="Component",
    xaxis="Word Index Category",
    title="Residual Attribution Patching",
    return_fig=True  # We need the figure object to further modify it
)

# Update the figure to add more specific labels if needed
fig.update_layout(
    xaxis_title="Word Index Category",
    yaxis_title="Component",
    xaxis=dict(tickmode='array', tickvals=list(range(len(x_labels))), ticktext=x_labels),
    yaxis=dict(tickmode='array', tickvals=list(range(len(residual_labels))), ticktext=residual_labels),
)

# Show the updated figure
fig.show()

# %%

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
    model, saes[10], ioi_dataset.toks, ioi_metric_noising
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(sae_clean_cache))
corrupted_value, sae_corrupted_cache, sae_corrupted_grad_cache = get_cache_fwd_and_bwd_sae(
    model, saes[10], abc_dataset.toks, ioi_metric_noising
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(sae_corrupted_cache))
print("Corrupted Gradients Cached:", len(sae_corrupted_grad_cache))


# %%
residual_attr_dict['end'][-5]
# residual_labels[-5]
# %%
hook_point = saes[10].cfg.hook_name + '.hook_sae_acts_post'

indices = ioi_dataset.word_idx["end"]
# indices
sae_clean_cache[hook_point][torch.arange(100), indices].shape



# %% layer 10 sae residual attribution

selected_corrupted_grad_residual = sae_corrupted_grad_cache[hook_point][torch.arange(100), indices]
selected_clean_residual = sae_clean_cache[hook_point][torch.arange(100), indices]
selected_corrupted_residual = sae_corrupted_grad_cache[hook_point][torch.arange(100), indices]

# Resulting tensors will have shape [25, 100, 768]

# Perform the operation on the selected values
residual_attr = einops.reduce(
    selected_corrupted_grad_residual * (selected_clean_residual - selected_corrupted_residual),
    "batch d_features -> d_features",
    "sum",
)
residual_attr.shape


#%%

top_feats = t.topk(residual_attr, 50)
top_feats.indices


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

_, _, feature_effects = patch_with_sae_features_with_hook(model, saes[10], sae_clean_cache, abc_dataset.toks, ioi_metric_noising, feature_list=top_feats.indices)
feature_effects



