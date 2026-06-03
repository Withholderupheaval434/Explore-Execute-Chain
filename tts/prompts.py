# Prompts (Appendix A.6)
from typing import List, Optional

QUESTION_SUFFIX = "Provide the final answer in the boxed{}. Please reasoning step-by-step."
SOLUTION_PREFIX = "<EXPLORATION>"

LLM_JUDGE_PROMPT = """Role: You are an expert mathematical reasoner and an impartial judge. Your task is to evaluate several proposed plans for solving a given math problem and identify the single best one.

Input:
• Problem: {problem}

• Candidate Plans:
{plans}

Instructions:
1. Carefully analyze the problem and each of the K candidate plans.
2. Assess the plans based on their logical soundness, potential for success, and efficiency.
3. Select the single best plan that is most likely to lead to a correct and complete solution.

Output Format: Output only the full text of the single best plan you have selected. Do not add any extra commentary, explanation, or formatting."""


LLM_JUDGE_READY_PROMPT = """Role: You are an expert mathematical reasoner and an impartial judge. Your task is to evaluate several proposed plans for solving a given math problem, identify the single best one, and decide whether that plan is already complete enough to derive the final numerical answer.

Input:
• Problem: {problem}

• Candidate Plans:
{plans}

Instructions:
1. Carefully analyze the problem and each of the candidate plans.
2. Assess the plans based on their logical soundness, potential for success, and efficiency.
3. Select the single best plan that is most likely to lead to a correct and complete solution.
4. Decide: Is this best plan already complete enough to derive the final numerical answer without further exploration? If yes, the reasoning is sufficient to execute and get the answer. If no, more exploration (another layer) would help.

Output Format (strict):
- On the first line, output exactly one word: READY or CONTINUE.
  - READY: the best plan is complete enough to derive the final answer now.
  - CONTINUE: the best plan is promising but more reasoning steps are needed; another layer of exploration is recommended.
- Starting from the next line, output the full text of the single best plan you have selected, with no extra commentary."""


def format_llm_judge_prompt(problem: str, plans: List[str]) -> str:
    plans_text = "\n".join(f"Plan {i+1}:\n{p.strip()}" for i, p in enumerate(plans))
    return LLM_JUDGE_PROMPT.format(problem=problem, plans=plans_text)


def format_llm_judge_ready_prompt(problem: str, plans: List[str]) -> str:
    """Judge that outputs READY/CONTINUE on first line, then best plan (for adaptive-depth ToT)."""
    plans_text = "\n".join(f"Plan {i+1}:\n{p.strip()}" for i, p in enumerate(plans))
    return LLM_JUDGE_READY_PROMPT.format(problem=problem, plans=plans_text)


def format_refine_exploration_prompt(
    problem: str,
    exploration: str,
    execution_excerpt: str,
    verify_feedback: str = "",
    hallucination_feedback: str = "",
) -> str:
    """
    Build refine prompt; use string concat so LaTeX/braces in exploration or excerpt
    never break str.format().

    ``verify_feedback``: non-empty when a verification pass judged the answer wrong.
    ``hallucination_feedback``: non-empty when the execution self-negated a formula
        (e.g. "but this is not a standard identity") — contains the exact formula(s)
        that must NOT be used in the revised plan.
    """
    p = (problem or "").strip()
    e = (exploration or "").strip()
    x = (execution_excerpt or "").strip()
    v = (verify_feedback or "").strip()
    h = (hallucination_feedback or "").strip()
    result = (
        "Role: You are an expert mathematician. A solution attempt used the exploration "
        "plan below but did not finish with a valid final answer in \\boxed{...} "
        "(or the attempt stalled / repeated).\n\n"
        "Problem:\n"
        + p
        + "\n\nCurrent exploration plan (baseline — revise this, do not restart from "
        "scratch unless it is unusable):\n"
        + e
        + "\n\nRecent execution output (excerpt; may be incomplete or looping):\n"
        + x
    )
    if h:
        result += (
            "\n\n*** HALLUCINATION WARNING ***\n"
            + h
        )
    if v:
        result += (
            "\n\nVerification feedback (a self-check found the answer above to be incorrect):\n"
            + v
        )
    result += (
        "\n\nInstructions:\n"
        "1. Identify what is missing, wrong, or blocking a clean finish. Common causes:\n"
        "   - An unjustified assumption (e.g. assigning a value to a variable not given "
        "in the problem). The revised plan must derive every unknown from the problem "
        "conditions; no free assumptions.\n"
        "   - A 'known result' shortcut used instead of an explicit derivation. The "
        "revised plan must spell out the derivation.\n"
        "   - A hallucinated formula that the model itself rejected (see HALLUCINATION "
        "WARNING above, if present). The revised plan must NOT reuse those formulas and "
        "must suggest an entirely different mathematical approach.\n"
        "   - If the problem asks for the minimum / maximum / smallest / all solutions, "
        "the plan must enumerate *every* candidate (e.g. all roots from Hensel's Lemma, "
        "all cases) before selecting the answer.\n"
        "   - Algebra / calculus slip: locate the exact incorrect step.\n"
        "2. Produce a **revised exploration plan only**: concrete, numbered steps the "
        "executor can follow. Keep correct ideas from the current plan; fix or extend "
        "only what is needed.\n"
        "3. Do not write the full final solution with \\boxed{} here — only the "
        "revised plan.\n\n"
        "Output format: Output only the revised plan text, with no title lines or "
        "meta-commentary."
    )
    return result


def build_verify_prompt(question: str, execution_output: str, answer: str, tokenizer) -> str:
    """
    Lightweight self-verification prompt.

    Ask the model to substitute the proposed answer back into the problem conditions
    and decide whether it is correct.  Output format: first non-empty line must be
    exactly CORRECT or WRONG; then a brief explanation.
    """
    excerpt = execution_output.strip()
    if len(excerpt) > 2000:
        excerpt = excerpt[-2000:]
    content = (
        "You are a mathematical verifier. Your ONLY job is to check whether the "
        "proposed answer is correct — do NOT re-solve the problem from scratch.\n\n"
        "Problem:\n"
        + question.strip()
        + "\n\nProposed answer: "
        + str(answer).strip()
        + "\n\nSolution excerpt (the reasoning that produced this answer):\n"
        + excerpt
        + "\n\nVerification task:\n"
        "1. Substitute the proposed answer back into the key conditions or equations "
        "of the problem and check they are all satisfied.\n"
        "2. Identify whether any critical derivation step was skipped or replaced by "
        "an unjustified assumption or shortcut.\n"
        "3. YOUR RESPONSE MUST START WITH EXACTLY ONE WORD ON ITS OWN LINE: "
        "either CORRECT or WRONG. Do not write any other text before this word. "
        "After that word, give a brief explanation (2–3 sentences).\n\n"
        "Example of correct response format:\n"
        "CORRECT\n"
        "Substituting the answer back satisfies all given conditions. "
        "The derivation is complete and no unjustified steps were found.\n\n"
        "Example of wrong response format:\n"
        "WRONG\n"
        "The answer does not satisfy [condition X]. "
        "Step Y made an unjustified assumption."
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_e2c_prompt(question: str, exploration: str, tokenizer, question_suffix: str = None) -> str:
    qs = question_suffix or QUESTION_SUFFIX
    content = question + " " + qs
    if exploration and exploration.strip():
        content = content + "\n\nGuideline (follow exactly):\n" + exploration.strip()
    content = content + (
        "\n\nExecution rules (must follow):\n"
        "- Follow every step of the guideline in order. Do NOT skip steps or switch to a different approach midway.\n"
        "- Complete every algebraic and calculus derivation in full. Do not replace a calculation with geometric intuition or an informal shortcut.\n"
        "- If a derivation feels complex, work through it explicitly step by step anyway.\n"
        "\nYou must end your solution with the final answer in the form \\boxed{your_answer}."
    )
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<EXPLORATION>" + (exploration.strip() or "") + "</EXPLORATION><EXECUTION>"


def build_targeted_reexplore_prompt(
    question: str,
    stuck_keywords: List[str],
    tokenizer,
    question_suffix: Optional[str] = None,
) -> str:
    """Exploration prompt with an explicit negative constraint on the stuck point.

    Used as a last resort when all K original exploration plans share the same
    mathematical bottleneck.  The stuck_keywords list contains short phrases
    extracted from the failed execution (e.g. ["ap · ad = ab · ac"]) so the
    model knows exactly which approach to avoid.
    """
    qs = question_suffix or QUESTION_SUFFIX
    kw_desc = "; ".join(f'"{kw}"' for kw in stuck_keywords[:3]) if stuck_keywords else ""
    warning = ""
    if kw_desc:
        warning = (
            "\n\nIMPORTANT: Previous solution attempts got stuck trying to use the "
            "following approach(es): " + kw_desc + ". "
            "Do NOT use these in your plan. "
            "Instead, propose a completely different mathematical strategy "
            "(e.g. similar triangles, angle chasing, trigonometric identities, "
            "coordinate geometry, cross-ratio / harmonic conjugate, Menelaus, "
            "Ceva, or any other approach that avoids the blocked step)."
        )
    content = question + " " + qs + warning
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + SOLUTION_PREFIX


def build_exploration_prompt(question: str, tokenizer, question_suffix: str = None) -> str:
    qs = question_suffix or QUESTION_SUFFIX
    content = question + " " + qs
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + SOLUTION_PREFIX


def build_full_cot_prompt(question: str, tokenizer, question_suffix: str = None) -> str:
    qs = question_suffix or QUESTION_SUFFIX
    content = question + " " + qs
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + SOLUTION_PREFIX


def build_boxed_only_prompt(question: str, tokenizer) -> str:
    """
    Rescue prompt: ask the model to output only the final answer in \\boxed{...}
    (no reasoning, no extra text).
    """
    content = (
        question.strip()
        + "\n\nProvide only the final answer in the exact form \\boxed{your_answer}."
        + " Do not write any reasoning or extra text."
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_boxed_only_from_exploration_prompt(
    question: str, exploration: str, tokenizer
) -> str:
    """
    Rescue prompt conditioned on exploration: still require only \\boxed{...}.
    """
    guide = (exploration or "").strip()
    content = question.strip() + (
        "\n\nUse the following exploration as guidance (but do not copy it)."
        + ("\nGuidance:\n" + guide if guide else "")
        + "\n\nNow output only the final answer in the exact form \\boxed{your_answer}."
        + " Do not write any reasoning or extra text."
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
