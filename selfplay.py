from tqdm import tqdm
import torch
import re

def prover_generate(model, query_tensors, tokenizer, generation_kwargs):
    prover_prompt = """Respond to the query. Reason step by step. Split steps with a double newline (\n\n).
    After finishing the reasoning end your response with a reasoning summary.
    Query: """
    
    full_prompts = [prover_prompt + tokenizer.decode(q) for q in query_tensors]
    full_prompt_tensors = tokenizer(full_prompts, return_tensors="pt", padding=True).input_ids
    
    # Add '\n\n' and '[EOS]' as stop tokens
    eos_token_ids = tokenizer.encode('\n\n') + tokenizer.encode('[EOS]')
    if 'eos_token_id' in generation_kwargs:
        if isinstance(generation_kwargs['eos_token_id'], int):
            generation_kwargs['eos_token_id'] = [generation_kwargs['eos_token_id']] + eos_token_ids
        else:
            generation_kwargs['eos_token_id'] = generation_kwargs['eos_token_id'] + eos_token_ids
    else:
        generation_kwargs['eos_token_id'] = eos_token_ids
    
    response_tensors = model.generate(full_prompt_tensors, **generation_kwargs)
    responses = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in response_tensors]
    
    # Remove the trailing '\n\n' if present, but keep '[EOS]' if it's there
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

def verifier_check(verifier_model, response, tokenizer, generation_kwargs):
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
    stop_token_ids = tokenizer.encode(['<neutral>', '<correct>'], add_special_tokens=False)
    if 'eos_token_id' in generation_kwargs:
        if isinstance(generation_kwargs['eos_token_id'], int):
            generation_kwargs['eos_token_id'] = [generation_kwargs['eos_token_id']] + stop_token_ids
        else:
            generation_kwargs['eos_token_id'] = generation_kwargs['eos_token_id'] + stop_token_ids
    else:
        generation_kwargs['eos_token_id'] = stop_token_ids
        
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

epochs = 1
for epoch in tqdm(range(epochs), desc="Training Progress"):
    for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch + 1}"):
        query_tensors = batch["input_ids"]
        
        all_prover_rewards = []
        all_sneaky_rewards = []
        all_verifier_rewards = []

        for query in query_tensors:
            response = tokenizer.decode(query)
            
            while True:
                sneaky_response = response
                #### Step 1: Get single-step response from Prover (PPO Model)
                prover_step = prover_generate(ppo_model1, [response], tokenizer, generation_kwargs)[0]
                response += prover_step + "\n"
                
                #### Step 2: Sneaky Prover generates a single-step response with a subtle mistake
                sneaky_step = sneaky_prover_generate(ppo_model2, [sneaky_response], tokenizer, generation_kwargs)[0]
                sneaky_response += sneaky_step + "\n"
                
                #### Step 3: Verifier checks both the Prover's and Sneaky Prover's responses
                prover_score, prover_output = verifier_check(verifier_model, prover_step, tokenizer, generation_kwargs)
                sneaky_score, sneaky_output = verifier_check(verifier_model, sneaky_step, tokenizer, generation_kwargs)
                
                #### Step 4: Compute reward scores
                prover_reward = -prover_score
                sneaky_reward = 1 if sneaky_score == -1 else -1 if sneaky_score == 1
                verifier_reward = 1 if sneaky_score == 1 else -1
                
                all_prover_rewards.append(prover_reward)
                all_sneaky_rewards.append(sneaky_reward)
                all_verifier_rewards.append(verifier_reward)
                
                #### Step 5: Run PPO steps
                prover_stats = ppo_trainer1.step([tokenizer.encode(response)], tokenizer(prover_step, return_tensors="pt", padding=True).input_ids, torch.tensor([prover_reward]))
                sneaky_stats = ppo_trainer2.step([tokenizer.encode(sneaky_response)], tokenizer(sneaky_step, return_tensors="pt", padding=True).input_ids, torch.tensor([sneaky_reward]))
                verifier_stats = ppo_trainer3.step([tokenizer.encode(sneaky_step)], tokenizer(sneaky_output, return_tensors="pt", padding=True).input_ids, torch.tensor([verifier_reward]))
                
                # Check if the prover has finished
                if '[EOS]' in prover_step:
                    break
                
                # If the step was wrong, provide feedback to the prover
                if prover_score == 1:
                    response += f"\nFeedback: {prover_output}\n"
        
        # Log stats for the entire batch
        ppo_trainer.log_stats({
            **prover_stats, 
            **{"sneaky_" + k: v for k, v in sneaky_stats.items()},
            **{"verifier_" + k: v for k, v in verifier_stats.items()}
        }, batch, torch.tensor(all_prover_rewards + all_sneaky_rewards + all_verifier_rewards))

#### Save models
ppo_trainer1.save_pretrained("ppo_model_prover")
ppo_trainer2.save_pretrained("ppo_model_sneaky_prover")
ppo_trainer3.save_pretrained("ppo_model_verifier")
