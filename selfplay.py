from tqdm import tqdm
import torch
import re
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead

# Load models and tokenizer
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"  # replace with the model you prefer

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
ppo_model1 = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ppo_model2 = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
verifier_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# Configure PPO training
ppo_config = PPOConfig(
    batch_size=1,
    log_with="wandb",
    learning_rate=1e-5,
    ppo_epochs=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1
)

generation_kwargs = {
    'max_length': 10000,
    'num_return_sequences': 1,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.95,
    'do_sample': True
}
# Initialize PPO trainers
ppo_trainer1 = PPOTrainer(ppo_config, ppo_model1, tokenizer=tokenizer)
ppo_trainer2 = PPOTrainer(ppo_config, ppo_model2, tokenizer=tokenizer)
ppo_trainer3 = PPOTrainer(ppo_config, verifier_model, tokenizer=tokenizer)

dataset = load_dataset("TIGER-Lab/MathInstruct", split="train[:100]")

# Tokenize the dataset's "instruction" column
def tokenize_function(examples):
    return tokenizer(examples["instruction"], padding="max_length", truncation=True, return_tensors="pt")

# Map tokenization function
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create a DataLoader
train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True)

# Assign the dataloader to PPO trainers
ppo_trainer1.dataloader = train_dataloader
ppo_trainer2.dataloader = train_dataloader
ppo_trainer3.dataloader = train_dataloader

def get_device(model):
    """Helper function to get the device of a model"""
    return next(model.parameters()).device

def prover_generate(model, query, tokenizer, generation_kwargs):
    prover_prompt = """You are a careful reasoning agent. Your task is to help solve problems by providing clear logical steps. Each step will be evaluated for correctness.

For each step:
1. Analyze the current state
2. Provide ONE logical step in solving it
3. Enclose your step within <step></step> tags
4. Mark your step as either:
   - <correct> if correct
   - <incorrect><explain>explanation of the error</explain> if wrong
   - <neutral> if neither helpful nor wrong
5. Continue with next step until solution is complete
6. End with [EOS] when solved

Query: """
    
    # Handle the query properly whether it's a list or single string
    if isinstance(query, list):
        query = query[0]
    
    full_prompt = prover_prompt + query
    device = get_device(model)
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Create a copy of generation_kwargs to modify
    gen_kwargs = generation_kwargs.copy()
    
    # Set the EOS token IDs
    eos_token_ids = tokenizer.encode('</step>')[0]  # Take first token ID
    gen_kwargs['eos_token_id'] = eos_token_ids
    
    outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    response = response[len(full_prompt):].strip()
    return response

def sneaky_prover_generate(model, query, tokenizer, generation_kwargs):
    sneaky_prompt = """You are a reasoning agent that introduces subtle mistakes. Your task is to take the current problem state and previous correct reasoning, then provide an alternative step that contains a deliberately misleading error.

Your response should:
1. Look at the previous reasoning and current problem state
2. Provide ONE step that appears plausible but contains a subtle error
3. Enclose your step within <step></step> tags
4. Explain your deliberate mistake within <explain></explain> tags

Previous reasoning and Query: """
    
    if isinstance(query, list):
        query = query[0]
    
    full_prompt = sneaky_prompt + query
    device = get_device(model)
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    gen_kwargs = generation_kwargs.copy()
    eos_token_ids = tokenizer.encode('</explain>')[0]
    gen_kwargs['eos_token_id'] = eos_token_ids
    
    outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response[len(full_prompt):].strip()
    return response

def verifier_check(verifier_model, step_text, is_sneaky, sneaky_explanation, tokenizer, generation_kwargs):
    verifier_prompt = """You are a verification agent. Your task is to carefully analyze reasoning steps for any mistakes.

For each step you analyze, provide your verdict using ONE of these formats:
1. <correct> - if the step is completely correct
2. <incorrect><explain>detailed explanation of the error</explain> - if you find any mistake
3. <neutral> - if the step neither helps nor hinders the solution

Step to verify: """
    
    verifier_input = verifier_prompt + step_text
    device = get_device(verifier_model)
    inputs = tokenizer(verifier_input, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    gen_kwargs = generation_kwargs.copy()
    
    # Define stop tokens and get their IDs
    stop_tokens = ['</explain>', '<correct>', '<neutral>']
    stop_token_ids = []
    for token in stop_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            stop_token_ids.append(token_ids[0])
    
    gen_kwargs['eos_token_id'] = stop_token_ids[0]
    gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    try:
        outputs = verifier_model.generate(**inputs, **gen_kwargs)
        verifier_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        verifier_output = verifier_output[len(verifier_input):].strip()
        
        correct_match = re.search(r'<correct>', verifier_output)
        incorrect_match = re.search(r'<incorrect><explain>(.*?)</explain>', verifier_output, re.DOTALL)
        neutral_match = re.search(r'<neutral>', verifier_output)
        
        if is_sneaky:
            if incorrect_match:  # Verifier caught the mistake
                return -1.0, 1.0, verifier_output
            else:  # Verifier missed the mistake
                full_output = verifier_output + f"\n<incorrect><explain>{sneaky_explanation}</explain>"
                return 1.0, -1.0, full_output
        else:
            if correct_match:
                return 1.0, 0.0, verifier_output
            elif incorrect_match:
                return -1.0, 0.0, verifier_output
            elif neutral_match:
                return 0.0, 0.0, verifier_output
            else:
                return 0.0, 0.0, "Verifier output format error: " + verifier_output
                
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return 0.0, 0.0, f"Verification error: {str(e)}"


epochs = 1
for epoch in tqdm(range(epochs), desc="Training Progress"):
    for batch in tqdm(ppo_trainer1.dataloader, desc=f"Epoch {epoch + 1}"):
        query_tensors = batch["input_ids"]
        
        for query in query_tensors:
            response = tokenizer.decode(query)
            current_state = response  # Keep track of the problem state
            
            while True:
                # Step 1: Get regular prover step
                prover_step = prover_generate(ppo_model1, [current_state], tokenizer, generation_kwargs)
                
                # Step 2: Get sneaky step (providing it with the current state including previous reasoning)
                sneaky_full = sneaky_prover_generate(ppo_model2, [current_state], tokenizer, generation_kwargs)
                step_match = re.search(r'<step>(.*?)</step>', sneaky_full, re.DOTALL)
                explain_match = re.search(r'<explain>(.*?)</explain>', sneaky_full, re.DOTALL)
                
                if not (step_match and explain_match):
                    print("Error: Sneaky prover output malformed")
                    continue
                    
                sneaky_step = step_match.group(1)
                sneaky_explanation = explain_match.group(1)
                
                # Step 3: Verify both steps
                prover_reward, _, prover_verification = verifier_check(
                    verifier_model, 
                    f"<step>{prover_step}</step>", 
                    False, 
                    None, 
                    tokenizer, 
                    generation_kwargs
                )
                
                sneaky_reward, verifier_reward, sneaky_verification = verifier_check(
                    verifier_model,
                    f"<step>{sneaky_step}</step>",
                    True,
                    sneaky_explanation,
                    tokenizer,
                    generation_kwargs
                )
                
                # Update models with full context
                prover_stats = ppo_trainer1.step(
                    [tokenizer.encode(current_state)],  # Full context as input
                    tokenizer(current_state + f"\n<step>{prover_step}</step>", return_tensors="pt", padding=True).input_ids,  # Full response including new step
                    torch.tensor([prover_reward])
                )
                
                sneaky_stats = ppo_trainer2.step(
                    [tokenizer.encode(current_state)],  # Full context as input
                    tokenizer(current_state + f"\n<step>{sneaky_step}</step>\n<explain>{sneaky_explanation}</explain>", return_tensors="pt", padding=True).input_ids,  # Full response including sneaky step and explanation
                    torch.tensor([sneaky_reward])
                )
                
                verifier_stats = ppo_trainer3.step(
                    [tokenizer.encode(current_state + f"\n<step>{sneaky_step}</step>")],  # Full context including step to verify
                    tokenizer(current_state + f"\n<step>{sneaky_step}</step>\n{sneaky_verification}", return_tensors="pt", padding=True).input_ids,  # Full response including verification
                    torch.tensor([verifier_reward])
                )

                # Train verifier with sneaky's explanation when it failed to catch the mistake
                if '<incorrect>' not in sneaky_verification:
                    explanation_stats = ppo_trainer3.step(
                        [tokenizer.encode(current_state + f"\n<step>{sneaky_step}</step>")],  # Full context including step to verify
                        tokenizer(current_state + f"\n<step>{sneaky_step}</step>\n<incorrect><explain>{sneaky_explanation}</explain>", return_tensors="pt", padding=True).input_ids,  # Full response including correct verification
                        torch.tensor([1.0])
                    )
                    
                    verifier_stats.update({f"explanation_{k}": v for k, v in explanation_stats.items()})

                # Update current_state for next iteration AFTER all training steps
                current_state += f"\n<step>{prover_step}</step>\n{prover_verification}\n"
                
                # Check if solution is complete
                if '[EOS]' in prover_step:
                    break

# Save the trained models
ppo_trainer1.save_pretrained("ppo_model_prover")
ppo_trainer2.save_pretrained("ppo_model_sneaky_prover")
ppo_trainer3.save_pretrained("ppo_model_verifier")
