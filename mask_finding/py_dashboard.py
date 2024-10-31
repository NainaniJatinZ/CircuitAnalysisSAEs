# %%
import os
import gc
import torch
os.chdir("/work/pi_jensen_umass_edu/jnainani_umass_edu/CircuitAnalysisSAEs")
import json
from sae_lens import SAE, HookedSAETransformer
from functools import partial
import einops

def cleanup_cuda():
   torch.cuda.empty_cache()
   gc.collect()

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

layers= [7, 14, 21, 40]
saes = [SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical", device=device)[0] for layer in layers]


# %%
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
data = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")


# %%
data['output']



# %%
import transformer_lens
SEQ_LEN = 128
tokenized_data = transformer_lens.utils.tokenize_and_concatenate(data, model.tokenizer, max_length=SEQ_LEN, column_name='output') # type: ignore
tokenized_data = tokenized_data.shuffle(42)

# Get the tokens as a tensor
all_tokens = tokenized_data["tokens"]
assert isinstance(all_tokens, torch.Tensor)

print(all_tokens.shape)


# %%
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


# %%

with open('mask_finding/mask.json') as f:
    mask = json.load(f)
mask

# %%
sae = saes[-1]
features = mask[sae.cfg.hook_name]
features

# %%

sae.fold_W_dec_norm()
# Configure visualization
config = SaeVisConfig(
    hook_point=sae.cfg.hook_name,
    features=features,
    minibatch_size_features=64,
    minibatch_size_tokens=256,
    device="cuda",
    dtype="bfloat16"
)

# Generate data
data = SaeVisRunner(config).run(encoder=sae, model=model, tokens=all_tokens)


# %%
# Save feature-centric visualization
from sae_dashboard.data_writing_fns import save_feature_centric_vis
save_feature_centric_vis(sae_vis_data=data, filename="feature_dashboard.html")
# %% getting 




