# %%
import os 
import gc
import torch
os.chdir("/work/pi_jensen_umass_edu/jnainani_umass_edu/CircuitAnalysisSAEs")
import json
from sae_lens import SAE, HookedSAETransformer
from circ4latents import data_gen
from functools import partial
import einops

# Function to manage CUDA memory and clean up
def cleanup_cuda():
    torch.cuda.empty_cache()
    gc.collect()
# Load the config
with open("config.json", 'r') as file:
    config = json.load(file)
token = config.get('huggingface_token', None)
os.environ["HF_TOKEN"] = token

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

hf_cache = "/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub"
os.environ["HF_HOME"] = hf_cache

# Load the model
model = HookedSAETransformer.from_pretrained("google/gemma-2-9b", device=device, cache_dir=hf_cache)

layers= [7, 14, 21, 28, 40]
saes = [SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical", device=device)[0] for layer in layers]

# %%
# Generate samples
from tasks.error_detection.type2.data import generate_samples

selected_templates = [1] # Adjust this as needed
N = 50
samples = generate_samples(selected_templates, N)

selected_pos = {"i_start": [], "i_end": [], "end": []}
for i in range(N):
    str_tokens_clean = model.to_str_tokens(samples[0][i])
    str_tokens_corr = model.to_str_tokens(samples[1][i])
    diff_positions = [i for i, (a, b) in enumerate(zip(str_tokens_clean, str_tokens_corr)) if a != b]
    pos_end = len(str_tokens_clean) - 1
    selected_pos["i_start"].append(diff_positions[0])
    selected_pos["i_end"].append(diff_positions[-1])
    selected_pos["end"].append(pos_end)

# %%
# Define logit diff function
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]

# iterate over samples[3] and create the list of correct tokens by tokenizing each
correct_answers = [model.tokenizer.encode(sample, add_special_tokens=False)[0] for sample in samples[2]]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]


# %%


def logit_diff_fn(logits, selected_pos):
    err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
    no_err = logits[range(logits.size(0)), selected_pos, correct_answers]
    return (err - no_err).mean()

# Disable gradients for all parameters
for param in model.parameters():
    param.requires_grad_(False)

clean_tokens = model.to_tokens(samples[0])
corr_tokens = model.to_tokens(samples[1])

# # Compute logits for clean and corrupted samples
logits = model(clean_tokens)
clean_diff = logit_diff_fn(logits, selected_pos['end'])

logits = model(corr_tokens)
corr_diff = logit_diff_fn(logits, selected_pos['end'])

print(f"clean_diff: {clean_diff}")
print(f"corr_diff: {corr_diff}")

# Cleanup
del logits
cleanup_cuda()


# Define error type metric
def _err_type_metric(logits, clean_logit_diff, corr_logit_diff, end_positions):
    patched_logit_diff = logit_diff_fn(logits, end_positions)
    return (patched_logit_diff - corr_logit_diff) / (clean_logit_diff - corr_logit_diff)

err_metric_denoising = partial(_err_type_metric, clean_logit_diff=clean_diff, corr_logit_diff=corr_diff, end_positions=selected_pos['end'])

# %%

samples[0][0]


# %%

from transformer_lens.utils import test_prompt
prompt =""">>> var = "time" + 39 \n"""
test_prompt(prompt, "Traceback", model, prepend_space_to_answer=False)

# %%


def run_with_saes_filtered(tokens, filtered_ids, model, saes):
    # Ensure tokens are a torch.Tensor
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens).to(model.cfg.device)  # Move to the device of the model
    
    # Create a mask where True indicates positions to modify
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in filtered_ids:
        mask &= tokens != token_id

    # Expand the mask once, so it matches the shape [batch_size, seq_len, 1]
    mask_expanded = mask.unsqueeze(-1)  # Expand to allow broadcasting

    # Precompute mask-expansion before adding hooks to avoid repeated expansions inside the hook
    mask_expanded = mask_expanded.to(model.cfg.device)  # Move the mask to the same device as the model
    
    # For each SAE, add the appropriate hook
    for sae in saes:
        hook_point = sae.cfg.hook_name

        # Define the filtered hook function (optimized)
        def filtered_hook(act, hook, sae=sae, mask_expanded=mask_expanded):
            # Apply the SAE only where mask_expanded is True
            modified_act = sae(act)  # Call the SAE once
            # In-place update where the mask is True
            act = torch.where(mask_expanded, modified_act, act)
            return act

        # Add the hook to the model
        model.add_hook(hook_point, filtered_hook, dir='fwd')

    with torch.no_grad():
        # Run the model with the tokens
        logits = model(tokens)

    # Reset the hooks after computation to free memory
    model.reset_hooks()

    return logits

# %%
filtered_ids = [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id]
clean_sae = run_with_saes_filtered(clean_tokens, filtered_ids, model, saes)
clean_metric = logit_diff_fn(clean_sae, selected_pos["end"])

print(f"clean_metric: {clean_metric}")
corr_sae = run_with_saes_filtered(corr_tokens, filtered_ids, model, saes)
corr_metric = logit_diff_fn(corr_sae, selected_pos["end"])
print(f"corr_metric: {corr_metric}")
# %%
# get the top 5 tokens from the base log probabilities
top_5_token_indices = torch.topk(corr_sae.mean(0)[-1], 5).indices  # Assuming single batch, last token position
# print the decoded indices
print(model.tokenizer.decode(top_5_token_indices))
# %%
