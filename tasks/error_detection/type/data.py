# %%
import random

def activate_autoreload():
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    print("In IPython")
    print("Set autoreload")

def generate_random_integer():
    return random.randint(10, 99)

def generate_random_variable_name():
    variable_names = ['abc',
 'name',
 'value',
 'text',
 'age',
 'score',
 'distance',
 'price',
 'weight',
 'temperature',
 'time',
 'energy',
 'volume',
 'velocity',
 'pressure',
 'humidity']
    return random.choice(variable_names)

# Template functions
def template_with_type_error_1():
    variable_name = generate_random_variable_name()
    integer = generate_random_integer()
    
    type_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> print("{variable_name}" + {integer})
"""
    no_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> print("{variable_name}" + "{integer}")
"""
    return type_error, no_error

def template_with_type_error_2():
    variable_name = generate_random_variable_name()
    integer = generate_random_integer()
    
    type_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> var = "{variable_name}" + {integer} 
"""
    no_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> var = "{variable_name}" + "{integer}"
"""
    return type_error, no_error

def template_with_type_error_3():
    integer1 = generate_random_integer()
    integer2 = generate_random_integer()
    
    type_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> var = {integer1} + "{integer2}"
"""
    no_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> var = {integer1} + {integer2}
"""
    return type_error, no_error

def template_with_type_error_4():
    integer1 = generate_random_integer()
    integer2 = generate_random_integer()
    integer3 = generate_random_integer()
    
    type_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> numbers = [{integer1}, {integer2}, {integer3}]
>>> result = numbers + "{integer1}"
"""
    no_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> numbers = [{integer1}, {integer2}, {integer3}]
>>> result = numbers + [{integer1}]
"""
    return type_error, no_error

def template_with_type_error_5():
    integer1 = generate_random_integer()
    integer2 = generate_random_integer()
    
    type_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> def add_numbers(a, b):
        return a + b
>>> result = add_numbers({integer1}, "{integer2}")
"""
    no_error = f"""Type "help", "copyright", "credits" or "license" for more information.
>>> def add_numbers(a, b):
        return a + b
>>> result = add_numbers({integer1}, {integer2})
>>> print(result)
"""
    return type_error, no_error

# Generate random pair based on selected templates
def generate_random_type_error_pair(selected_templates):
    template_functions = {
        1: template_with_type_error_1,
        2: template_with_type_error_2,
        3: template_with_type_error_3,
        4: template_with_type_error_4,
        5: template_with_type_error_5
    }

    chosen_template_function = random.choice([template_functions[i] for i in selected_templates])
    return chosen_template_function()

# Function to generate N samples based on selected templates
def generate_samples(selected_templates, N):
    clean = []
    corr = []
    for i in range(N):
        type_error, no_error = generate_random_type_error_pair(selected_templates)
        # print(f"Sample {i+1} With TypeError:\n{type_error}")
        # print(f"Sample {i+1} Without Error:\n{no_error}")
        # print("-" * 80)
        clean.append(type_error)
        corr.append(no_error)
    return clean, corr
    
ipython = get_ipython()
if ipython is not None:
    print("In IPython")
    IN_IPYTHON = True
    activate_autoreload()
    # Code to automatically update the EasyTransformer code as its edited without restarting the kernel
    import tqdm.notebook as tqdm
else:
    print("Not in IPython")
    IN_IPYTHON = False
    import tqdm
    
# %%

if __name__ == "__main__":
    # User input: Select which templates to use (1-5) and how many samples to generate (N)
    selected_templates = [1] #, 2, 3]  
    N = 5  # Example: Generate 5 samples

    # Generate and display samples
    clean_prompts, corr_prompts = generate_samples(selected_templates, N)
    clean_prompts
    corr_prompts
# # User input: Select which templates to use (1-5) and how many samples to generate (N)
# selected_templates = [1] #, 2, 3]  
# N = 5  # Example: Generate 5 samples

# # Generate and display samples
# clean_prompts, corr_prompts = generate_samples(selected_templates, N)
# # %%
# clean_prompts
# # %%
# corr_prompts
# # %%
