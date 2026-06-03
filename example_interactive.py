#!/usr/bin/env python3
"""
E2C Interactive Demo

Select from example problems or enter your own to see E2C reasoning in action.
"""

import json
import sys
from pathlib import Path
from example_inference import run_inference, parse_e2c_response


def load_example_problems():
    problem_file = Path("example_problems.json")
    if not problem_file.exists():
        return []
    with open(problem_file, 'r') as f:
        return json.load(f)


def display_menu(problems):
    print("\n" + "=" * 70)
    print("E2C Interactive Demo")
    print("=" * 70)
    print()
    for i, prob in enumerate(problems, 1):
        print(f"{i}. [{prob['category']}] {prob['difficulty']}")
        preview = prob['problem'][:80] + "..." if len(prob['problem']) > 80 else prob['problem']
        print(f"   {preview}")
        print()
    print(f"{len(problems) + 1}. Enter your own problem")
    print(f"{len(problems) + 2}. Exit")
    print()


def main():
    model_path_str = "TingheOliver/Explore-Execute-Chain-Qwen"
    subfolder_str = "Qwen3-8B-E2C-SFT-RL"   # set to None for local paths
    if "/" not in model_path_str and not Path(model_path_str).exists():
        print("Error: E2C model not found at", model_path_str)
        print("\nTo use the released model from HuggingFace:")
        print("  python example_interactive.py  (uses TingheOliver/Explore-Execute-Chain-Qwen by default)")
        return
    model_path = model_path_str

    problems = load_example_problems()
    if not problems:
        print("Warning: example_problems.json not found, custom input only")

    print("\nWelcome to the E2C interactive demo.")

    while True:
        if problems:
            display_menu(problems)
            try:
                choice = input("Select an option (1-{}): ".format(len(problems) + 2))
                choice = int(choice)
            except (ValueError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if choice == len(problems) + 2:
                print("Goodbye!")
                break
            elif choice == len(problems) + 1:
                print("\nEnter your problem (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                problem = "\n".join(lines[:-1])
            elif 1 <= choice <= len(problems):
                selected = problems[choice - 1]
                print(f"\nSelected: {selected['category']} - {selected['difficulty']}")
                print(f"Expected answer: {selected['answer']}")
                problem = selected['problem']
            else:
                print("Invalid choice, please try again.")
                continue
        else:
            print("\nEnter your problem (press Enter twice to finish):")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    return
            problem = "\n".join(lines[:-1])
            if not problem.strip():
                print("Empty problem, exiting.")
                break

        try:
            print("\nRunning E2C inference...")
            response = run_inference(
                model_path=model_path,
                problem=problem,
                max_tokens=2048,
                temperature=0.7,
                subfolder=subfolder_str,
            )
            exploration, execution = parse_e2c_response(response)

            print("\n" + "=" * 70)
            if exploration:
                print("EXPLORATION (planning):")
                print("-" * 70)
                print(exploration)
                print()
            if execution:
                print("EXECUTION (detailed reasoning):")
                print("-" * 70)
                print(execution)
                print()
            if not exploration and not execution:
                print("FULL RESPONSE:")
                print("-" * 70)
                print(response)
                print()
            print("=" * 70)

            if problems:
                cont = input("\nTry another problem? (y/n): ").lower()
                if cont != 'y':
                    print("Goodbye!")
                    break
            else:
                break

        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
            if problems:
                cont = input("\nTry again? (y/n): ").lower()
                if cont != 'y':
                    break
            else:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        sys.exit(0)
