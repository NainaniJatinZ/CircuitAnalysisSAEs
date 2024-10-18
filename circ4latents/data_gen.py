import random

# Define lists for code-like function names and non-code English phrases
code_like_names = ["print", "sum", "return", "fetch", "calculate", "process"]
english_phrases = ["hello", "greet", "hi", "say", "ask", "tell"]

# Define list of variable names (with inverted commas)
variable_names = ["'age'", "'score'", "'data'", "'value'", "'result'", "'input'"]

# Function to generate N clean and corrupted pairs based on the two templates
def generate_two_templates_pairs(n):
    pairs = []
    
    for _ in range(n):
        # Randomly pick a code-like function and an English phrase
        code_func = random.choice(code_like_names)
        english_phrase = random.choice(english_phrases)
        
        # Randomly pick a variable name with inverted commas
        var_name = random.choice(variable_names)
        
        # Template 1: code-like vs. non-code-like
        clean_1 = f"{code_func} {var_name}"
        corrupted_1 = f"{english_phrase} {var_name}"
        
        # Template 2: function call-like vs. plain
        clean_2 = f"{english_phrase}({var_name})"
        corrupted_2 = f"{english_phrase} {var_name}"
        
        # Append both pairs to the list
        pairs.append((clean_1, corrupted_1))
        pairs.append((clean_2, corrupted_2))
    
    return pairs

if __name__ == "__main__":
    # Generate and print N pairs
    def print_two_templates_pairs(n):
        pairs = generate_two_templates_pairs(n)
        for clean, corrupted in pairs:
            print(f"Clean: {clean}\nCorrupted: {corrupted}\n")

    # Example: Generate 5 pairs
    print_two_templates_pairs(5)


