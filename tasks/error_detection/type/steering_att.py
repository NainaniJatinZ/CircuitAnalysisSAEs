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
from tasks.error_detection.type.data import generate_samples

selected_templates = [2] # Adjust this as needed
N = 30
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

# Define logit diff function
traceback_token_id = model.tokenizer.encode("Traceback", add_special_tokens=False)[0]
trip_arrow_token_id = model.tokenizer.encode(">>>", add_special_tokens=False)[0]

def logit_diff_fn(logits, selected_pos):
    err = logits[range(logits.size(0)), selected_pos, :][:, traceback_token_id]
    no_err = logits[range(logits.size(0)), selected_pos, :][:, trip_arrow_token_id]
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
from transformer_lens.utils import test_prompt
model.reset_hooks()
prompt = """>>> print("energy" + 22)\n"""
test_prompt(prompt, "Traceback", model, prepend_space_to_answer=False)



# %%

with open('tasks/error_detection/type/out/second_dict_feats.json') as f:
    dict_feats = json.load(f)
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

# L7.9681, L14.14967,
_steering_hook = partial(
        steering_hook,
        sae=saes[1],
        latent_idx=14967,
        steering_coefficient=10,
    )
# model.add_sae(sae)
model.add_hook(saes[1].cfg.hook_name, _steering_hook, "fwd")
# model.add_hook(saes[0].cfg.hook_name, _steering_hook2, "fwd")
with torch.no_grad():
    logits = model(clean_tokens)
model.reset_hooks()
steered_diff = err_metric_denoising(logits)
# steered_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"steered_diff: {steered_diff}")


# %%
with torch.no_grad():
    logits = model(corr_tokens)
clean_diff = err_metric_denoising(logits)
# steered_diff = logit_diff_fn(logits, selected_pos['end'])
print(f"clean_diff: {clean_diff}")

# %%

from transformer_lens.utils import test_prompt
model.reset_hooks()
test_prompt(samples[0][5], "Traceback", model)


# %%
# Run the model without steering to get the base logits and log probabilities
with torch.no_grad():
    base_logits = model(clean_tokens)
    base_log_probs = base_logits.softmax(dim=-1)
    #torch.log_softmax(base_logits, dim=-1)
print(base_log_probs.shape)
# Get the top 5 tokens from the base log probabilities
top_5_token_indices = torch.topk(base_log_probs[1, -1], 5).indices  # Assuming single batch, last token position

print(f"Top 5 token indices: {top_5_token_indices}")
print(model.tokenizer.decode(top_5_token_indices))



# %%
# base_log_probs.shape
# base_log_probs.mean(0)[-1].shape
# # get the avg across the batch and then topk 
top_5_token_indices = torch.topk(base_log_probs.mean(0)[-1], 5).indices  # Assuming single batch, last token position

print(f"Top 5 token indices: {top_5_token_indices}")
print(model.tokenizer.decode(top_5_token_indices))

# %%
import matplotlib.pyplot as plt
# Define the steering hook function
def steering_hook(
    activations,
    hook,
    sae,
    latent_idx,
    steering_coefficient,
):
    return activations + steering_coefficient * sae.W_dec[latent_idx]

# Range of coefficients for steering
coefficients = range(-80, 80, 10)  # Adjust range and step as needed

# Dictionary to store log probs for each token at each coefficient value
log_probs_per_token = {token.item(): [] for token in top_5_token_indices}

# Iterate over coefficients, applying steering and recording log probs
for coeff in coefficients:
    _steering_hook = partial(
        steering_hook,
        sae=saes[3],
        latent_idx=4611,
        steering_coefficient=coeff,
    )
    model.add_hook(saes[3].cfg.hook_name, _steering_hook, "fwd")

    with torch.no_grad():
        steered_logits = model(clean_tokens)
        steered_log_probs = steered_logits.softmax(dim=-1)
        # torch.log_softmax(steered_logits, dim=-1)

    # Store log probs for the top 5 tokens at this coefficient
    for token_idx in top_5_token_indices:

        avg_logits = steered_log_probs.mean(0)
        token_log_prob = avg_logits[-1, token_idx].item()
        log_probs_per_token[token_idx.item()].append(token_log_prob)

    # Reset hooks after each coefficient to avoid interference
    model.reset_hooks()

# Plotting the results
plt.figure(figsize=(10, 6))
for token_idx, log_probs in log_probs_per_token.items():
    plt.plot(coefficients, log_probs, label=f'Token {model.tokenizer.decode(token_idx)}')

plt.xlabel("Steering Coefficient")
plt.ylabel("Log Probability")
plt.title("Log Probability of Top Tokens vs Steering Coefficient")
plt.legend()
plt.grid(True)
plt.show()
# %%
