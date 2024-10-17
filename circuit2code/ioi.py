# %% 
def identify_ioi(sentence, name_indices):
    """
    Identify the indirect object (IOI) based on the sentence and the indices of names.
    
    Args:
    - sentence: The full sentence as a string.
    - name_indices: A list of indices where the names occur in the sentence.
    
    Returns:
    - The remaining name (IO) after removing duplicates.
    """
    # Step 1: Tokenize the sentence into words
    words = sentence.split()

    # Step 2: Extract the names from the provided indices
    names = [words[i] for i in name_indices]

    # Step 3: Count occurrences of each name and remove duplicates
    name_counts = {name: names.count(name) for name in names}

    remaining_names = [name for name, count in name_counts.items() if count == 1]

    # Step 4: Return the remaining name (if any)
    if remaining_names:
        return remaining_names[0]  # The indirect object (IO)
    else:
        return None

# Example sentence and the list of word indices where names appear
sentence = "When John and Mary went to the park, John gave a book to"
name_indices = [1, 3, 8]  # Indices of "John", "Mary", "John"

# Call the function with the provided sentence and indices
indirect_object = identify_ioi(sentence, name_indices)

print(f"The indirect object (IO) is: {indirect_object}")

# %%

sentence.split()
# %%
