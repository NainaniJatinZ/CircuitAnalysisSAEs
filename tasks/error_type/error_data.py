# %%
import random
from typing import List, Tuple, Callable, Dict
import torch

# Define the template sets and values (moved here to be loaded automatically)
TEMPLATES_AND_VALUES = {
    1: {
        "clean_template": """print("{string_value} " + {int_value})
# When this code is executed, Python will raise a""",
        "corrupted_template": """print("{string_value} " + "{int_value}"
# When this code is executed, Python will raise a""",
        "values": {
            "string_value": ["age:", "height:", "value:", "name:", "score:", "distance:", "price:", "weight:", "rating:", "color:", "temperature:", "location:", "area:", "time:", "velocity:", "pressure:", "humidity:", "energy:", "volume:"],
            "int_value": [25, 100, 42, 7, 63, 150, 200, 10, 50, 120, 300, 500, 75, 80, 90, 15, 600, 45, 110, 130]
        }
    },
    2: {
        "clean_template": """my_list = [1, 2, 3]
my_list += {int_value}
# When this code is executed, Python will raise a""",
        "corrupted_template": """my_list = [1, 2, 3
my_list += {int_value}
# When this code is executed, Python will raise a""",
        "values": {
            "int_value": [5, 10, 100, 2, 20, 50, 75, 1, 33, 44, 99, 150, 250, 300, 400, 500, 600, 13, 21, 55]
        }
    }
}

# %%

def generate_prompts_by_template_numbers(template_numbers: List[int], N: int, seed: int = None) -> List[Tuple[str, str]]:
    """
    Generate a list of clean and corrupted prompts using selected template numbers.

    Args:
    - template_numbers: A list of template numbers to use.
    - N: Number of prompts to generate.
    - seed: Optional random seed for reproducibility.

    Returns:
    - A list of tuples (clean_prompt, corrupted_prompt)
    """
    clean_prompts = []
    corrupted_prompts = []

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    for template_num in template_numbers:
        template_set = TEMPLATES_AND_VALUES.get(template_num)
        if not template_set:
            raise ValueError(f"Template {template_num} does not exist.")
        
        clean_template = template_set["clean_template"]
        corrupted_template = template_set["corrupted_template"]
        values = template_set["values"]

        # For each template, randomly sample a set of values
        for _ in range(N):
            sampled_values = {key: random.choice(values[key]) for key in values.keys()}
            
            # Generate clean and corrupted prompts using the same sampled values
            clean_prompt = clean_template.format(**sampled_values)
            corrupted_prompt = corrupted_template.format(**sampled_values)
            
            clean_prompts.append(clean_prompt)
            corrupted_prompts.append(corrupted_prompt)

    # Combine clean and corrupted prompts and shuffle
    combined_prompts = list(zip(clean_prompts, corrupted_prompts))
    random.shuffle(combined_prompts)

    # Return only N randomly selected prompts
    return combined_prompts[:N]

def get_differing_positions(clean_tokens: List[int], corrupted_tokens: List[int], pad_token_id: int) -> List[int]:
    """
    Function to find the positions where the clean and corrupted tokens differ.

    Args:
    - clean_tokens: Tokenized clean prompt.
    - corrupted_tokens: Tokenized corrupted prompt.
    - pad_token_id: The token ID representing the padding token.

    Returns:
    - A list of indices where the clean and corrupted tokens differ.
    """
    differing_positions = []
    length = min(len(clean_tokens), len(corrupted_tokens))  # Compare up to the shorter length

    for i in range(length):
        if clean_tokens[i] != corrupted_tokens[i] and clean_tokens[i] != pad_token_id and corrupted_tokens[i] != pad_token_id:
            differing_positions.append(i)

    return differing_positions

# %% 

import torch
from typing import List

def tokenize_prompts(
    prompts: List[str],
    tokenizer: Callable,
    prepend_bos: bool = True,
    padding_side: str = "right",
    truncate: bool = False,
    max_length: int = None,
    move_to_device: bool = True,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Tokenizes the given list of prompts, with optional BOS token prepending and padding.

    Args:
    - prompts: A list of prompts (sentences).
    - tokenizer: The tokenizer object with `__call__` method for encoding.
    - prepend_bos: Whether to prepend a beginning-of-sequence (BOS) token.
    - padding_side: Side to apply padding ('right' or 'left').
    - truncate: Whether to truncate the tokens to a maximum length.
    - max_length: Maximum length to truncate the tokens, if truncation is enabled.
    - move_to_device: Whether to move the resulting token tensors to the given device.
    - device: The device to move the tensors to ('cpu', 'cuda', etc.).

    Returns:
    - A tensor of tokenized prompts.
    """
    # Optionally prepend BOS tokens
    if prepend_bos:
        prompts = [tokenizer.bos_token + prompt for prompt in prompts]

    # Tokenize with padding and truncation options
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=truncate,
        max_length=max_length,
    )["input_ids"]

    # Optionally move the token tensors to the specified device
    if move_to_device:
        tokens = tokens.to(device)
    
    return tokens


def get_end_positions(tokenized_prompts: List[List[int]], eos_token_id: int, pad_token_id: int) -> List[int]:
    """
    Function to find the end position of each tokenized prompt based on EOS or padding tokens.

    Args:
    - tokenized_prompts: A list of tokenized prompts.
    - eos_token_id: The token ID representing the end-of-sequence (EOS) token.
    - pad_token_id: The token ID representing the padding token.

    Returns:
    - end_positions: A list of integers indicating the end position for each tokenized prompt.
    """
    end_positions = []
    for tokens in tokenized_prompts:
        if eos_token_id in tokens:
            end_pos = tokens.index(eos_token_id)
        else:
            end_pos = len(tokens) - 1
            while end_pos >= 0 and tokens[end_pos] == pad_token_id:
                end_pos -= 1
        end_positions.append(end_pos)
    return end_positions


def create_dataset(N: int, template_numbers: List[int], tokenizer: Callable) -> Tuple[List[str], List[str], List[List[int]], List[List[int]], List[int], List[int], List[List[int]]]:
    """
    Function to generate a dataset of clean and corrupted prompts, tokenize them, and get their end positions and differing positions.

    Args:
    - N: Number of prompts to generate.
    - template_numbers: A list of template numbers to use for prompt generation.
    - tokenizer: The tokenizer object with `encode` method.

    Returns:
    - clean_prompts: A list of clean prompts.
    - corrupted_prompts: A list of corrupted prompts.
    - clean_tokens: Tokenized clean prompts.
    - corrupted_tokens: Tokenized corrupted prompts.
    - clean_end_positions: End positions of clean prompts.
    - corrupted_end_positions: End positions of corrupted prompts.
    - differing_positions: Positions where clean and corrupted tokens differ.
    """
    # Generate clean and corrupted prompts
    generated_prompts = generate_prompts_by_template_numbers(template_numbers, N)
    clean_prompts = [clean_prompt for clean_prompt, _ in generated_prompts]
    corrupted_prompts = [corr_prompt for _, corr_prompt in generated_prompts]

    # Tokenize the prompts
    clean_tokens = torch.Tensor(tokenizer(clean_prompts, padding=True).input_ids).type(
            torch.int
        )
    corrupted_tokens = torch.Tensor(tokenizer(corrupted_prompts, padding=True).input_ids).type(
            torch.int
        )

    # Get EOS and padding token IDs
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Get end positions for clean and corrupted prompts
    clean_end_positions = get_end_positions(clean_tokens, eos_token_id, pad_token_id)
    corrupted_end_positions = get_end_positions(corrupted_tokens, eos_token_id, pad_token_id)

    # Get differing positions between clean and corrupted tokens
    differing_positions = [
        get_differing_positions(clean, corrupt, pad_token_id)
        for clean, corrupt in zip(clean_tokens, corrupted_tokens)
    ]

    return clean_prompts, corrupted_prompts, clean_tokens, corrupted_tokens, clean_end_positions, corrupted_end_positions, differing_positions

# Example usage
# if __name__ == "__main__":
#     # Assuming you have a tokenizer object that has .encode and special token ids
#     from transformers import AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained("gpt2")

#     N = 10  # Number of prompts to generate
#     template_numbers = [1, 2]  # Template numbers to use

#     clean_prompts, corrupted_prompts, clean_tokens, corrupted_tokens, clean_end_positions, corrupted_end_positions = create_dataset(
#         N=N,
#         template_numbers=template_numbers,
#         tokenizer=tokenizer
#     )

#     # Output the results
#     print("Clean Prompts:", clean_prompts)
#     print("Corrupted Prompts:", corrupted_prompts)
#     print("Clean Tokens:", clean_tokens)
#     print("Corrupted Tokens:", corrupted_tokens)
#     print("Clean End Positions:", clean_end_positions)
#     print("Corrupted End Positions:", corrupted_end_positions)
