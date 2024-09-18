from tqdm import tqdm
import torch
import re

def prover_generate(model, query_tensors, tokenizer, generation_kwargs):
    prover_prompt = """Provide the next step in responding to the following query. 
    End your response with a double newline (\n\n).
    Query: """
    
    full_prompts = [prover_prompt + tokenizer.decode(q) for q in query_tensors]
    full_prompt_tensors = tokenizer(full_prompts, return_tensors="pt", padding=True).input_ids
    
    # Add '\n\n' as a stop token
    if 'eos_token_id' in generation_kwargs:
        generation_kwargs['eos_token_id'] = [generation_kwargs['eos_token_id'], tokenizer.encode('\n\n')[-1]]
    else:
        generation_kwargs['eos_token_id'] = tokenizer.encode('\n\n')[-1]
    
    response_tensors = model.generate(full_prompt_tensors, **generation_kwargs)
    responses = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in response_tensors]
    
    # Remove the trailing '\n\n' if present
    responses = [r[:-2] if r.endswith('\n\n') else r for r in responses]
    
    return responses

def sneaky_prover_generate(model, query_tensors, tokenizer, generation_kwargs):
    sneaky_prompt = """Provide the next step in responding to the following query, but introduce a subtle mistake in your reasoning or calculation. 
    The mistake should be hard to notice at first glance. 
    End your response with a double newline (\n\n).
    Query: """
    
    full_prompts = [sneaky_prompt + tokenizer.decode(q) for q in query_tensors]
    full_prompt_tensors = tokenizer(full_prompts, return_tensors="pt", padding=True).input_ids
    
    # Add '\n\n' as a stop token
    if 'eos_token_id' in generation_kwargs:
        generation_kwargs['eos_token_id'] = [generation_kwargs['eos_token_id'], tokenizer.encode('\n\n')[-1]]
    else:
        generation_kwargs['eos_token_id'] = tokenizer.encode('\n\n')[-1]
    
    sneaky_response_tensors = model.generate(full_prompt_tensors, **generation_kwargs)
    sneaky_responses = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in sneaky_response_tensors]
    
    # Remove the trailing '\n\n' if present
    sneaky_responses = [r[:-2] if r.endswith('\n\n') else r for r in sneaky_responses]
    
    return sneaky_responses

def verifier_check(verifier_model, response, tokenizer, generation_kwargs, is_sneaky):
    verifier_prompt = f"""Analyze the last step in reasoning in the following response for any mistakes in reasoning, logic, or calculations. 
    First, add one of these tags to your response:
    <wrong> if you find a mistake
    <correct> if the step is correct
    <neutral> if the step has no effect on the reasoning
    
    If you use the <wrong> tag, explain why the step is wrong after the tag.
    If you use <correct> or <neutral>, stop immediately after the tag.

    Reasoning to verify:
    {response}

    Your analysis:"""
    
    verifier_input_tensors = tokenizer(verifier_prompt, return_tensors="pt").input_ids
    verifier_output_tensors = verifier_model.generate(verifier_input_tensors, **generation_kwargs)
    verifier_output = tokenizer.decode(verifier_output_tensors[0], skip_special_tokens=True)
    
    # Extract the tag and explanation (if any)
    match = re.match(r'<(wrong|correct|neutral)>\s*(.*)', verifier_output, re.DOTALL)
    if match:
        tag, explanation = match.groups()
        if tag == 'wrong':
            return 1, verifier_output  # Mistake found
        elif tag == 'correct':
            return -1, verifier_output  # No mistake found
        else:  # neutral
            return 0, verifier_output  # No effect
    else:
        return 0, "Verifier output format error: " + verifier_output  # Error in verifier output format

epochs = 10
for epoch in tqdm(range(epochs), desc="Training Progress"):
    for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch + 1}"):
        query_tensors = batch["input_ids"]
        
        #### Step 1: Get single-step response from Prover (PPO Model)
        prover_responses = prover_generate(ppo_model1, query_tensors, tokenizer, generation_kwargs)
        batch["prover_response"] = prover_responses
        
        #### Step 2: Sneaky Prover generates a single-step response with a subtle mistake
        sneaky_responses = sneaky_prover_generate(ppo_model2, query_tensors, tokenizer, generation_kwargs)
        batch["sneaky_response"] = sneaky_responses
        
        #### Step 3: Verifier checks both the Prover's and Sneaky Prover's responses
        verifier_results_prover = [verifier_check(verifier_model, r, tokenizer, generation_kwargs, is_sneaky=False) for r in prover_responses]
        verifier_results_sneaky = [verifier_check(verifier_model, r, tokenizer, generation_kwargs, is_sneaky=True) for r in sneaky_responses]
        
        #### Step 4: Compute reward scores
        prover_scores, prover_outputs = zip(*verifier_results_prover)
        sneaky_scores, sneaky_outputs = zip(*verifier_results_sneaky)
        
        # Reward the prover for correct responses and penalize for mistakes
        prover_rewards = [-score for score in prover_scores]
        
        # Reward the sneaky prover for mistakes that weren't caught (when verifier says correct)
        sneaky_rewards = [1 if score == -1 else -1 for score in sneaky_scores]
        verifier_rewards = [1 if score == 1 else -1 for score in sneaky_scores]
        #### Step 5: Run PPO steps
        prover_stats = ppo_trainer1.step(query_tensors, tokenizer(prover_responses, return_tensors="pt", padding=True).input_ids, torch.tensor(prover_rewards))
        sneaky_stats = ppo_trainer2.step(query_tensors, tokenizer(sneaky_responses, return_tensors="pt", padding=True).input_ids, torch.tensor(sneaky_rewards))
        verifier_stats = ppo_trainer3.step(query_tensors, tokenizer(sneaky_responses, return_tensors="pt", padding=True).input_ids, torch.tensor(verifier_rewards))
        
        
        ppo_trainer.log_stats({**prover_stats, **{"sneaky_" + k: v for k, v in sneaky_stats.items()}}, batch, torch.tensor(prover_rewards + sneaky_rewards))

#### Save models
ppo_trainer.save_pretrained("ppo_model_with_single_step_prover")