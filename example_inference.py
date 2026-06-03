#!/usr/bin/env python3
"""
E2C (Explore-Execute Chain) - Simple Inference Example

This script demonstrates how to use E2C models for reasoning tasks.
The model will first explore possible solution paths, then execute the chosen path.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path


def run_inference(model_path: str, problem: str, max_tokens: int = 2048, temperature: float = 0.7, subfolder: str = None):
    """
    Run E2C inference on a given problem.
    
    Args:
        model_path: HuggingFace model ID or local path
        problem: Problem statement to solve
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        subfolder: Optional subfolder within the HF repo (e.g. "Qwen3-8B-E2C-SFT-RL")
    
    Returns:
        Generated response string
    """
    print("=" * 70)
    print("Loading E2C Model...")
    print("=" * 70)
    
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if subfolder:
        load_kwargs["subfolder"] = subfolder

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, **({"subfolder": subfolder} if subfolder else {})
    )
    
    print(f"Model loaded: {model_path}")
    print(f"Device: {model.device}")
    print()
    
    # Prepare prompt
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("=" * 70)
    print("Problem:")
    print("=" * 70)
    print(problem)
    print()
    
    # Generate response
    print("=" * 70)
    print("Generating E2C Reasoning...")
    print("=" * 70)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Try to extract just the assistant's response
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response


def parse_e2c_response(response: str):
    """
    Parse E2C response to extract exploration and execution phases.
    
    Args:
        response: Generated response string
        
    Returns:
        Tuple of (exploration, execution)
    """
    exploration = ""
    execution = ""
    
    # Try to extract exploration phase
    if "<EXPLORATION>" in response and "</EXPLORATION>" in response:
        exploration = response.split("<EXPLORATION>")[1].split("</EXPLORATION>")[0].strip()
    elif "<EXPLORATION>" in response and "<EXECUTION>" in response:
        exploration = response.split("<EXPLORATION>")[1].split("<EXECUTION>")[0].strip()
    
    # Try to extract execution phase
    if "<EXECUTION>" in response:
        if "</EXECUTION>" in response:
            execution = response.split("<EXECUTION>")[1].split("</EXECUTION>")[0].strip()
        else:
            execution = response.split("<EXECUTION>")[1].strip()
    
    return exploration, execution


def main():
    parser = argparse.ArgumentParser(description="E2C Inference Example")
    parser.add_argument(
        "--model_path",
        type=str,
        default="TingheOliver/Explore-Execute-Chain-Qwen",
        help="Path to E2C model"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Problem to solve (optional, uses example if not provided)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within the HF repo (e.g. Qwen3-8B-E2C-SFT-RL or Qwen3-4B-E2C-SFT-RL)"
    )
    
    args = parser.parse_args()
    
    # Use example problem if none provided
    if args.problem is None:
        args.problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    
    # Skip local-path check for HuggingFace model IDs (contain '/')
    model_path = Path(args.model_path)
    if "/" not in args.model_path and not model_path.exists():
        print(f"Error: model not found at {args.model_path}")
        print("\nTo use a released model:")
        print("  python example_inference.py --model_path TingheOliver/Explore-Execute-Chain-Qwen")
        return
    
    # Run inference
    try:
        response = run_inference(
            model_path=args.model_path,
            problem=args.problem,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            subfolder=args.subfolder,
        )
        
        # Parse and display response
        exploration, execution = parse_e2c_response(response)
        
        print()
        print("=" * 70)
        print("E2C Response:")
        print("=" * 70)
        
        if exploration:
            print("\nEXPLORATION PHASE:")
            print("-" * 70)
            print(exploration)
            print()

        if execution:
            print("EXECUTION PHASE:")
            print("-" * 70)
            print(execution)
            print()

        if not exploration and not execution:
            print("\nFULL RESPONSE:")
            print("-" * 70)
            print(response)
            print()

        print("=" * 70)
        print("Done.")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

