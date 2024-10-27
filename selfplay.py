from tqdm import tqdm
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig

# Load models and tokenizer
model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # replace with the model you prefer

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
ppo_model1 = AutoModelForCausalLM.from_pretrained(model_name)
ppo_model2 = AutoModelForCausalLM.from_pretrained(model_name)
verifier_model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure PPO training
ppo_config = PPOConfig(batch_size=1, ppo_epochs=4, clip_range=0.2, lr=1e-5, model_name=model_name)

# Initialize PPO trainers
ppo_trainer1 = PPOTrainer(ppo_model1, tokenizer, **ppo_config.to_dict())
ppo_trainer2 = PPOTrainer(ppo_model2, tokenizer, **ppo_config.to_dict())
ppo_trainer3 = PPOTrainer(verifier_model, tokenizer, **ppo_config.to_dict())

dataset = load_dataset("TIGER-Lab/MathInstruct")

# Tokenize the dataset's "instruction" column
def tokenize_function(examples):
    return tokenizer(examples["instruction"], padding="max_length", truncation=True, return_tensors="pt")

# Map tokenization function
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create a DataLoader
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=1, shuffle=True)

# Assign the dataloader to PPO trainers
ppo_trainer1.dataloader = train_dataloader
ppo_trainer2.dataloader = train_dataloader
ppo_trainer3.dataloader = train_dataloader

def prover_generate(model, query_tensors, tokenizer, generation_kwargs):
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

Example:
Query: What is 15 + 27?
<step>First, let's add the ones digits: 5 + 7 = 13. We write down 3 and carry the 1.</step>
<incorrect><explain>5 + 7 = 12, not 13. This affects the entire calculation.</explain>

<step>Let me correct that. Adding the ones digits: 5 + 7 = 12. We write down 2 and carry the 1.</step>
<correct>

<step>Now we add the tens: 1 (carried) + 1 + 2 = 4, giving us 42 as our final answer.</step>
<correct>

<step>The problem is solved. 15 + 27 = 42</step>
[EOS]

Remember to be clear and precise in your reasoning. Each step should be logical and help progress toward the solution.
Query: """
    
    full_prompts = [prover_prompt + tokenizer.decode(q) for q in query_tensors]
    full_prompt_tensors = tokenizer(full_prompts, return_tensors="pt", padding=True).input_ids
    
    eos_token_ids = tokenizer.encode('</step>') + tokenizer.encode('[EOS]')
    if 'eos_token_id' in generation_kwargs:
        if isinstance(generation_kwargs['eos_token_id'], int):
            generation_kwargs['eos_token_id'] = [generation_kwargs['eos_token_id']] + eos_token_ids
        else:
            generation_kwargs['eos_token_id'] = generation_kwargs['eos_token_id'] + eos_token_ids
    else:
        generation_kwargs['eos_token_id'] = eos_token_ids
    
    response_tensors = model.generate(full_prompt_tensors, **generation_kwargs)
    responses = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in response_tensors]
    
    return responses

def sneaky_prover_generate(model, query_tensors, tokenizer, generation_kwargs):
    sneaky_prompt = """You are a reasoning agent that introduces subtle mistakes. Your task is to take the current problem state and previous correct reasoning, then provide an alternative step that contains a deliberately misleading error.

Your response should:
1. Look at the previous reasoning and current problem state
2. Provide ONE step that appears plausible but contains a subtle error
3. Enclose your step within <step></step> tags
4. Explain your deliberate mistake within <explain></explain> tags

Example:
Previous reasoning and Query:
Query: What is 15 + 27?
<step>First, let's add the ones digits: 5 + 7 = 12. We write down 2 and carry the 1.</step> <correct>

<step>Now we add the tens: 1 + 1 + 2 = 3, giving us 32 as our final answer.</step>
<explain>I deliberately ignored the carried 1 when adding the tens, making it seem like a simple oversight.</explain>

Previous reasoning and Query: """

    
    full_prompts = [sneaky_prompt + tokenizer.decode(q) for q in query_tensors]
    full_prompt_tensors = tokenizer(full_prompts, return_tensors="pt", padding=True).input_ids
    
    eos_token_ids = tokenizer.encode('</explain>')
    if 'eos_token_id' in generation_kwargs:
        if isinstance(generation_kwargs['eos_token_id'], int):
            generation_kwargs['eos_token_id'] = [generation_kwargs['eos_token_id']] + eos_token_ids
        else:
            generation_kwargs['eos_token_id'] = generation_kwargs['eos_token_id'] + eos_token_ids
    else:
        generation_kwargs['eos_token_id'] = eos_token_ids
    
    response_tensors = model.generate(full_prompt_tensors, **generation_kwargs)
    responses = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in response_tensors]
    
    return responses

def verifier_check(verifier_model, step_text, is_sneaky, sneaky_explanation, tokenizer, generation_kwargs):
    verifier_prompt = """You are a verification agent. Your task is to carefully analyze reasoning steps for any mistakes.

For each step you analyze, provide your verdict using ONE of these formats:
1. <correct> - if the step is completely correct
2. <incorrect><explain>detailed explanation of the error</explain> - if you find any mistake
3. <neutral> - if the step neither helps nor hinders the solution

Note: When you use <correct> or <neutral>, no explanation is needed.
When you use <incorrect>, always include an explanation of the error within <explain></explain> tags.

Step to verify:
"""
    
    verifier_input = verifier_prompt + step_text
    verifier_input_tensors = tokenizer(verifier_input, return_tensors="pt").input_ids
    
    stop_token_ids = tokenizer.encode(['</explain>', '<correct>', '<neutral>'], add_special_tokens=False)
    if 'eos_token_id' in generation_kwargs:
        if isinstance(generation_kwargs['eos_token_id'], int):
            generation_kwargs['eos_token_id'] = [generation_kwargs['eos_token_id']] + stop_token_ids
        else:
            generation_kwargs['eos_token_id'] = generation_kwargs['eos_token_id'] + stop_token_ids
    else:
        generation_kwargs['eos_token_id'] = stop_token_ids
    
    verifier_output_tensors = verifier_model.generate(verifier_input_tensors, **generation_kwargs)
    verifier_output = tokenizer.decode(verifier_output_tensors[0], skip_special_tokens=True)
    
    # Parse verification result
    correct_match = re.search(r'<correct>', verifier_output)
    incorrect_match = re.search(r'<incorrect><explain>(.*?)</explain>', verifier_output, re.DOTALL)
    neutral_match = re.search(r'<neutral>', verifier_output)
    
    if is_sneaky:
        if incorrect_match:  # Verifier caught the mistake
            verifier_reward = 1.0
            sneaky_reward = -1.0
            return sneaky_reward, verifier_reward, verifier_output
        else:  # Verifier missed the mistake
            verifier_reward = -1.0
            sneaky_reward = 1.0
            # Add the sneaky prover's explanation after the verification attempt
            full_output = verifier_output + f"\n<incorrect><explain>{sneaky_explanation}</explain>"
            return sneaky_reward, verifier_reward, full_output
    else:
        if correct_match:
            return 1.0, 0.0, verifier_output  # Positive reward for prover
        elif incorrect_match:
            return -1.0, 0.0, verifier_output  # Negative reward for prover
        elif neutral_match:
            return 0.0, 0.0, verifier_output  # No reward for neutral step
        else:
            return 0.0, 0.0, "Verifier output format error: " + verifier_output

epochs = 1
for epoch in tqdm(range(epochs), desc="Training Progress"):
    for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch + 1}"):
        query_tensors = batch["input_ids"]
        
        for query in query_tensors:
            response = tokenizer.decode(query)
            current_state = response  # Keep track of the problem state
            
            while True:
                # Step 1: Get regular prover step
                prover_step = prover_generate(ppo_model1, [current_state], tokenizer, generation_kwargs)[0]
                
                # Step 2: Get sneaky step (providing it with the current state including previous reasoning)
                sneaky_full = sneaky_prover_generate(ppo_model2, [current_state], tokenizer, generation_kwargs)[0]
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
