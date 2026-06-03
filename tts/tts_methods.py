# TTS methods
import random
import re
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional
from util.reward import boxed_evaluate, check_answer_match, extract_boxed_content
from prompts import (
    build_exploration_prompt,
    build_e2c_prompt,
    build_full_cot_prompt,
    build_boxed_only_from_exploration_prompt,
    build_boxed_only_prompt,
    format_llm_judge_prompt,
    format_llm_judge_ready_prompt,
    format_refine_exploration_prompt,
    build_verify_prompt,
    build_targeted_reexplore_prompt,
)

# E2C special token IDs (Qwen3)
TOKEN_EXPLORATION_END = 151672   # </EXPLORATION>
TOKEN_EXECUTION_END = 151674     # </EXECUTION>


def _trim_generated_row_ids(row: torch.Tensor, tokenizer) -> List[int]:
    """
    Remove trailing batch-padding from one row of *new* token ids.

    HF batched generate pads shorter rows to the same length; those positions inflate
    both decode strings and naive numel() counts. When pad_id == eos_id, alignment
    padding is duplicate trailing eos tokens — trim to a single trailing eos.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    ids = row.tolist()
    if not ids:
        return ids
    if pad_id is not None and eos_id is not None and pad_id == eos_id:
        while len(ids) > 1 and ids[-1] == eos_id and ids[-2] == eos_id:
            ids.pop()
    elif pad_id is not None and pad_id != eos_id:
        while ids and ids[-1] == pad_id:
            ids.pop()
    return ids


def _count_effective_new_tokens(new_ids: torch.Tensor, tokenizer) -> int:
    """Sum of per-row lengths after removing trailing batch padding (not naive numel)."""
    total = 0
    for i in range(new_ids.size(0)):
        ids = _trim_generated_row_ids(new_ids[i], tokenizer)
        total += len(ids)
    return total


# Lowercase markers that signal the model has moved from planning to executing.
# When any of these appear in the generated exploration text, everything from
# that point onwards is execution noise that should be stripped out.
_EXPLORATION_EXEC_MARKERS = [
    "i need to carefully and step-by-step execute",
    "i need to carefully execute",
    "let me carefully execute",
    "now let me execute",
    "let's execute this plan",
    "now i will execute",
    "executing the plan",
    "step-by-step execution:",
    "step by step execution:",
    "<execution>",
]


def _strip_execution_leak(text: str) -> str:
    """Remove or skip execution content that leaked into an exploration string.

    Two cases:
    - Marker appears at the *start* (< 40 chars in): the real plan content
      follows the header — skip past the marker and any separator lines so
      only the numbered plan steps remain.
    - Marker appears *mid-text* (>= 40 chars in): everything from the marker
      onwards is execution noise — truncate there.
    """
    # Work iteratively because skipping a leading header may expose another one.
    for _ in range(5):  # at most 5 passes to handle stacked headers
        low = text.lower()
        earliest_mid = len(text)
        leading_skip = None
        for marker in _EXPLORATION_EXEC_MARKERS:
            idx = low.find(marker)
            if idx == -1:
                continue
            if idx < 40:
                # Leading header: jump past the marker and any dashes/whitespace
                after = idx + len(marker)
                while after < len(low) and low[after] in " \t\n\r-":
                    after += 1
                if leading_skip is None or after < leading_skip:
                    leading_skip = after
            elif idx < earliest_mid:
                earliest_mid = idx
        if leading_skip is not None:
            # Skip the header; then loop to strip any further headers
            text = text[leading_skip:].strip()
        elif earliest_mid < len(text):
            # Truncate at mid-text execution marker
            text = text[:earliest_mid].strip()
            break
        else:
            break  # Nothing to strip
    return text.strip()


def _extract_exploration(text: str) -> str:
    if "</EXPLORATION>" in text:
        end = text.index("</EXPLORATION>")
        if "<EXPLORATION>" in text:
            start = text.index("<EXPLORATION>") + len("<EXPLORATION>")
            if start < end:
                return _strip_execution_leak(text[start:end].strip())
        # Prompt ends with "<EXPLORATION>"; continuation has no opening tag in new_ids
        return _strip_execution_leak(text[:end].strip())
    if "<EXPLORATION>" in text and "<EXECUTION>" in text:
        start = text.index("<EXPLORATION>") + len("<EXPLORATION>")
        end = text.index("<EXECUTION>")
        return _strip_execution_leak(text[start:end].strip())
    return _strip_execution_leak(text.strip())


def _extract_answer(text: str) -> str:
    contents = extract_boxed_content(text)
    return contents[-1] if contents else ""


def _extract_final_answer(text: str) -> str:
    """Extract the boxed answer from an execution output.

    Two guards compared to the plain ``_extract_answer``:

    1. *Trailing-text guard*: refuses a ``\\boxed{}`` with more than 500 chars
       following it — those are mid-reasoning guesses, not final answers.

    2. *Minimum-reasoning guard*: refuses a ``\\boxed{}`` that appears before
       300 chars of preceding text — the model gave up immediately and output
       a number without doing any work (e.g. after a confused plan enumeration).
    """
    contents = extract_boxed_content(text)
    if not contents:
        return ""
    # Locate the last \boxed{ occurrence.
    last_idx = text.rfind("\\boxed{")
    if last_idx == -1:
        return contents[-1]
    # Guard 2: require at least 300 chars of reasoning before the answer.
    if last_idx < 300:
        return ""
    # Walk forward to find the matching closing brace.
    depth = 0
    close_idx = last_idx
    for i in range(last_idx, min(last_idx + 500, len(text))):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    # Guard 1: reject if substantial reasoning continues after this boxed.
    text_after = text[close_idx + 1:].strip()
    if len(text_after) > 500:
        return ""
    return contents[-1]


def _plans_are_too_similar(old: str, new: str, threshold: float = 0.85) -> bool:
    """Return True when the revised plan carries essentially no new content.

    Measured as word-level Jaccard overlap: if the new plan's vocabulary is
    covered >``threshold`` by the old plan, the refinement stalled and we
    should not waste the next execution on an identical plan.
    """
    if not old or not new:
        return False
    old_words = set(old.lower().split())
    new_words = set(new.lower().split())
    if not new_words:
        return True
    overlap = len(old_words & new_words) / len(new_words)
    return overlap > threshold


def _majority_vote(answers: List[str]) -> str:
    if not answers:
        return ""
    from collections import Counter
    valid = [a for a in answers if a and a.strip()]
    if not valid:
        return answers[0] if answers else ""
    return Counter(valid).most_common(1)[0][0]


def _weighted_majority(answers: List[str], weights: List[float]) -> str:
    if not answers or not weights or len(answers) != len(weights):
        return _majority_vote(answers)
    from collections import defaultdict
    scores = defaultdict(float)
    for a, w in zip(answers, weights):
        if a and a.strip():
            scores[a.strip()] += w
    return max(scores, key=scores.get) if scores else (answers[0] if answers else "")


# ---------- LoopGuard helpers ----------

# Keywords that indicate the model abandoned the planned approach mid-execution.
# When any of these appear in an execution output that carries a boxed answer,
# the answer may have been reached via an unjustified shortcut and warrants
# a verification pass.
# ---------- Self-negation / hallucination patterns ----------
# (Lazy-execution string matching was removed: common phrases like
# "instead, let", "actually, let", "it is well-known" appear in legitimate
# rigorous proofs and caused too many false verify triggers.  Verification is
# now always run when an answer is found — the verify model itself is the
# right semantic judge of whether execution was faithful.)


# Phrases the model uses when it tries a formula, realises it is wrong, but
# cannot find the correct replacement.  These indicate a "hallucination loop":
# the model invented a plausible-sounding but incorrect mathematical identity
# and keeps cycling between attempting and self-rejecting it.
_SELF_NEGATION_PATTERNS = [
    "but this is not a standard identity",
    "but this is not a valid",
    "but this is not correct",
    "but this is incorrect",
    "but this is not a standard",
    "which is not a standard identity",
    "which is not correct",
    "which is incorrect",
    "this is not a valid use of",
    "this is not a standard result",
    "this formula is not correct",
    "this formula is incorrect",
    "this is not a valid formula",
    "is not a valid identity",
]

# How many characters of context to capture around a self-negation hit
# when building the feedback snippet for the refine prompt.
_HALLUCINATION_CONTEXT_CHARS = 300


def _extract_hallucination_feedback(text: str) -> str:
    """Scan execution output for self-negation patterns and return a feedback
    string that tells the refine model exactly which formula/approach was
    hallucinated so the revised plan explicitly avoids it.

    Returns an empty string when no self-negation is detected.
    """
    low = text.lower()
    snippets: List[str] = []
    seen: set = set()
    for pat in _SELF_NEGATION_PATTERNS:
        idx = low.find(pat)
        while idx != -1:
            # Grab a window of text around the hit for context.
            start = max(0, idx - _HALLUCINATION_CONTEXT_CHARS)
            end = min(len(text), idx + len(pat) + _HALLUCINATION_CONTEXT_CHARS)
            snippet = text[start:end].strip()
            # Deduplicate near-identical snippets.
            key = snippet[:80].lower()
            if key not in seen:
                seen.add(key)
                snippets.append(snippet)
            idx = low.find(pat, idx + 1)
        if len(snippets) >= 3:
            break  # enough context; avoid bloating the prompt

    if not snippets:
        return ""
    joined = "\n---\n".join(snippets)
    return (
        "The execution attempted the following approach(es) but explicitly "
        "self-rejected them as mathematically incorrect:\n"
        + joined
        + "\n\nDo NOT repeat any of the above formulas or reasoning steps in "
        "the revised plan. The revised plan must take a completely different "
        "approach to establish the missing relationship."
    )


def _extract_stuck_keywords(execution_text: str) -> List[str]:
    """Extract short keyword phrases identifying the specific step where the
    model looped.

    Strategy: for each self-negation hit, take the ~150 chars *before* the
    negation pattern (the text that was being negated) and extract the last
    sentence-like fragment as the keyword.  This gives us a compact,
    matchable representation of the stuck approach (e.g. "ap · ad = ab · ac").
    """
    low = execution_text.lower()
    keywords: List[str] = []
    seen: set = set()
    for pat in _SELF_NEGATION_PATTERNS:
        idx = low.find(pat)
        if idx == -1:
            continue
        before_start = max(0, idx - 150)
        before_text = execution_text[before_start:idx].strip()
        # Take the last sentence / clause (split on . , ; or newline)
        parts = re.split(r"[.,;\n]", before_text)
        parts = [p.strip() for p in parts if len(p.strip()) >= 6]
        if parts:
            kw = parts[-1].lower()[:80]
            if kw and kw not in seen:
                seen.add(kw)
                keywords.append(kw)
    return keywords


def _plan_avoids_stuck_keywords(plan: str, stuck_keywords: List[str]) -> bool:
    """Return True if the plan does not contain any of the stuck-point phrases.

    Used to prefer plans that approach the problem differently from the one
    that caused the hallucination loop.
    """
    if not stuck_keywords:
        return True
    plan_lower = plan.lower()
    return not any(kw in plan_lower for kw in stuck_keywords)


def _normalize_for_loops(s: str) -> str:
    # Reduce whitespace variance when doing heuristic repetition checks.
    return re.sub(r"\s+", " ", s.strip())


def _tail_sentences(text: str, max_chars: int = 1200, max_sents: int = 8) -> List[str]:
    t = _normalize_for_loops(text)[-max_chars:]
    # Sentence/segment split: tuned for mixed punctuation + newlines.
    parts = re.split(r"[\n。！？!?;；]+", t)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts[-max_sents:]


def _process_degeneracy_score(text: str) -> float:
    """
    Heuristic score for "looping / repeating the same sentence".
    Higher = more degenerate.
    """
    sents = _tail_sentences(text, max_chars=1400, max_sents=10)
    if len(sents) < 4:
        return 0.0
    last = sents[-8:]

    # Specialized ABAB cycle detector (the dominant failure mode observed):
    # A,B,A,B,... where adjacent sentences differ, but 2-step repeats persist.
    # Guardrails reduce false positives on normal reasoning progression.
    if len(last) >= 6:
        uniq_count = len(set(last))
        if uniq_count <= 3:
            cycle_hits = 0
            valid_pairs = 0
            for i in range(2, len(last)):
                a = last[i]
                b = last[i - 2]
                prev = last[i - 1]
                if not a or not b:
                    continue
                valid_pairs += 1
                same2 = (a == b) or (a in b) or (b in a)
                diff1 = a != prev
                if same2 and diff1:
                    cycle_hits += 1
            if valid_pairs >= 4 and cycle_hits >= 3:
                # Strong ABAB signal; return very high score directly.
                return 0.97

    uniq = len(set(last))
    repeat_ratio = 1.0 - (uniq / max(1, len(last)))
    # Also punish consecutive duplicates (common in copy loops).
    consec = 0
    for i in range(1, len(last)):
        if last[i] == last[i - 1]:
            consec += 1
    # Map to [0, ~2], then clamp.
    score = repeat_ratio + (consec / max(1, len(last)))
    return max(0.0, min(1.0, score))


def _is_process_degenerate(text: str, threshold: float = 0.65) -> bool:
    return _process_degeneracy_score(text) >= threshold


def _is_result_degenerate(text: str) -> bool:
    # Result degeneracy: model didn't produce a boxed answer (or can't extract one).
    return not _extract_answer(text)


def _has_repeated_block_loop(text: str) -> bool:
    """
    Detect high-confidence long-form template loops (low false positives):
    repeated multi-sentence blocks appearing 3+ times near the tail.
    """
    t = _normalize_for_loops(text)
    if len(t) < 900:
        return False
    # Strict repeated block patterns from the tail window.
    tail = t[-1800:]
    # 2-sentence block
    sents = _tail_sentences(tail, max_chars=1800, max_sents=12)
    if len(sents) >= 8:
        pairs = [" || ".join(sents[i : i + 2]) for i in range(0, len(sents) - 1)]
        nontrivial = [p for p in pairs if len(p) >= 80]
        if nontrivial:
            from collections import Counter
            most = Counter(nontrivial).most_common(1)[0][1]
            if most >= 3:
                return True
    # 1-sentence long block repeated many times
    long_sents = [s for s in sents if len(s) >= 120]
    if long_sents:
        from collections import Counter
        most = Counter(long_sents).most_common(1)[0][1]
        if most >= 4:
            return True
    return False


def _has_paragraph_repeat_loop(
    text: str,
    min_para_len: int = 20,
    min_repeats: int = 3,
    tail_paras: int = 40,
) -> bool:
    """
    Detect paragraph-level repetition that sentence-level checks miss.

    Root cause they address: ``_normalize_for_loops`` collapses all ``\\n``
    into spaces *before* sentence splitting, so multi-paragraph English prose
    becomes one giant "sentence" and ABAB / block detectors are blind to it.

    This function intentionally works on the *raw* (un-normalised) text so that
    paragraph boundaries are preserved.

    Detected patterns
    -----------------
    * Single paragraph repeating ≥ min_repeats times in the tail.
    * Any block of 2–4 consecutive paragraphs repeating ≥ min_repeats times.
    """
    from collections import Counter

    # --- split into paragraphs (prefer blank-line breaks, fall back to newlines)
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paras) < min_repeats * 2:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    paras = [p for p in paras if len(p) >= min_para_len]
    if len(paras) < min_repeats * 2:
        return False

    tail = paras[-tail_paras:]
    n = len(tail)

    # single-paragraph repetition
    if Counter(tail).most_common(1)[0][1] >= min_repeats:
        return True

    # block repetition (2–4 consecutive paragraphs)
    for bsz in range(2, min(5, n // min_repeats + 1)):
        blocks = [" ||| ".join(tail[i : i + bsz]) for i in range(n - bsz + 1)]
        if blocks and Counter(blocks).most_common(1)[0][1] >= min_repeats:
            return True

    return False


def _has_inline_repeat_loop(text: str, window: int = 600, min_frag: int = 8, min_hits: int = 6) -> bool:
    """
    Detect intra-line / intra-formula repetitions that sentence-level heuristics miss.

    Examples caught:
      ``16 \\cdot 31.5 = 16 \\cdot 31.5 = 16 \\cdot 31.5 = ...``
      ``= \\frac{11}{25} = \\frac{11}{25} = ...``

    Also catches sentence-level self-negation loops such as:
      "Let's use X ... but this is not standard." repeated 4+ times.

    Strategy: tokenise the tail on ``=`` / ``\\\\`` for formula fragments, then run
    a second pass on sentence boundaries for longer natural-language repetitions.
    """
    from collections import Counter
    t = _normalize_for_loops(text)
    if len(t) < window:
        return False

    # --- pass 1: formula-fragment repetition (original logic) ---
    tail = t[-window:]
    frags = re.split(r"[=\\]+", tail)
    frags = [f.strip() for f in frags if f and f.strip()]
    if len(frags) >= min_hits:
        counts: Counter = Counter()
        prev = None
        run = 1
        for frag in frags:
            if len(frag) < min_frag:
                prev = None
                run = 1
                continue
            if frag == prev:
                run += 1
                if run >= min_hits:
                    return True
            else:
                run = 1
            prev = frag
            counts[frag] += 1
        for frag, cnt in counts.items():
            if len(frag) >= min_frag and cnt >= min_hits + 2:
                return True

    # --- pass 2: sentence-level repetition (wider window, lower threshold) ---
    # Catches natural-language loops like "but this is not a standard identity" × N.
    sent_window = 1400
    tail2 = t[-sent_window:]
    # Split on sentence-ending punctuation or newlines.
    sents = re.split(r"[.!?\n]+", tail2)
    sents = [s.strip() for s in sents if len(s.strip()) >= 30]
    if sents:
        sent_counts: Counter = Counter(sents)
        if sent_counts.most_common(1)[0][1] >= 4:
            return True

    return False


def _generate_single_with_chunk_guard(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
    chunk_tokens: int = 256,
    stop_token_ids: Optional[List[int]] = None,
) -> Tuple[str, int, bool]:
    """
    Incremental generation for one prompt; stop early on high-confidence loop.
    Returns: (text, tokens, stopped_by_loop)
    """
    total_tokens = 0
    acc = ""
    stopped_by_loop = False
    remaining = max_tokens
    while remaining > 0:
        step = min(chunk_tokens, remaining)
        texts, t = generate(
            model,
            tokenizer,
            [prompt + acc],
            step,
            temperature=temperature,
            do_sample=do_sample,
            stop_token_ids=stop_token_ids,
            device=device,
        )
        total_tokens += t
        piece = texts[0] if texts else ""
        if not piece:
            break
        acc += piece
        # Only cut on strong evidence to avoid hurting normal reasoning.
        if (
            _has_repeated_block_loop(acc)
            or _has_inline_repeat_loop(acc)
            or _has_paragraph_repeat_loop(acc)
            or _is_process_degenerate(acc, threshold=0.93)
        ):
            stopped_by_loop = True
            break
        remaining -= step
    return acc, total_tokens, stopped_by_loop


def generate_with_process_loop_guard(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    stop_token_ids: Optional[List[int]],
    device: str,
    max_retries: int = 1,
    threshold: float = 0.95,
) -> Tuple[List[str], int]:
    """
    LoopGuard for process-level repetition:
    - run generation
    - detect degenerate outputs with a heuristic
    - retry only degenerate prompts with more conservative decoding
    """
    # Stage-0 decoding config: keep it close to baseline to reduce accuracy drop.
    repetition_penalty0 = 1.01
    ngram0 = None
    temp0 = temperature

    texts, total_tokens = generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens,
        temperature=temp0,
        do_sample=do_sample,
        stop_token_ids=stop_token_ids,
        device=device,
        repetition_penalty=repetition_penalty0,
        no_repeat_ngram_size=ngram0,
    )

    for _ in range(max_retries):
        deg_flags = [_is_process_degenerate(t, threshold=threshold) for t in texts]
        if not any(deg_flags):
            break
        idxs = [i for i, f in enumerate(deg_flags) if f]
        regen_prompts = [prompts[i] for i in idxs]

        repetition_penalty1 = 1.08
        ngram1 = 5
        # If sampling, reduce temp to avoid getting stuck in a template.
        temp1 = min(temperature, 0.7) if do_sample else 1.0

        regen_texts, extra_tokens = generate(
            model,
            tokenizer,
            regen_prompts,
            max_new_tokens,
            temperature=temp1,
            do_sample=do_sample,
            stop_token_ids=stop_token_ids,
            device=device,
            repetition_penalty=repetition_penalty1,
            no_repeat_ngram_size=ngram1,
        )
        total_tokens += extra_tokens
        # Replace only degenerate positions.
        for local_j, global_i in enumerate(idxs):
            texts[global_i] = regen_texts[local_j]

    return texts, total_tokens


def _generate_boxed_only_from_question(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int,
    device: str,
) -> Tuple[str, int]:
    prompt = build_boxed_only_prompt(question, tokenizer)
    texts, t = generate(
        model,
        tokenizer,
        [prompt],
        max_new_tokens,
        temperature=0.0,
        do_sample=False,
        stop_token_ids=None,
        device=device,
        repetition_penalty=1.2,
        no_repeat_ngram_size=6,
    )
    return texts[0] if texts else "", t


def _generate_boxed_only_from_exploration(
    model,
    tokenizer,
    question: str,
    exploration: str,
    max_new_tokens: int,
    device: str,
) -> Tuple[str, int]:
    prompt = build_boxed_only_from_exploration_prompt(
        question, exploration, tokenizer
    )
    texts, t = generate(
        model,
        tokenizer,
        [prompt],
        max_new_tokens,
        temperature=0.0,
        do_sample=False,
        stop_token_ids=None,
        device=device,
        repetition_penalty=1.2,
        no_repeat_ngram_size=6,
    )
    return texts[0] if texts else "", t


def generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float = 0.9,
    do_sample: bool = True,
    stop_token_ids: Optional[List[int]] = None,
    device: str = "cuda",
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Tuple[List[str], int]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=16384,
    ).to(device)
    eos_ids = [tokenizer.eos_token_id]
    if stop_token_ids:
        eos_ids.extend(stop_token_ids)
    with torch.no_grad():
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            repetition_penalty=repetition_penalty,
        )
        if no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["top_p"] = 0.95 if do_sample else 1.0

        out = model.generate(**gen_kwargs)
    new_ids = out[:, inputs.input_ids.shape[1]:]
    total_tokens = _count_effective_new_tokens(new_ids, tokenizer)
    texts = []
    for i in range(new_ids.size(0)):
        ids = _trim_generated_row_ids(new_ids[i], tokenizer)
        if not ids:
            texts.append("")
        else:
            texts.append(
                tokenizer.decode(
                    ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    return texts, total_tokens


def sample_explorations(
    model, tokenizer, question: str, K: int, max_explore_tokens: int,
    temperature: float, device: str
) -> Tuple[List[str], int]:
    prompt = build_exploration_prompt(question, tokenizer)
    prompts = [prompt] * K
    texts, total = generate_with_process_loop_guard(
        model, tokenizer, prompts, max_explore_tokens,
        temperature=temperature, do_sample=True,
        stop_token_ids=[TOKEN_EXPLORATION_END],
        device=device,
        threshold=0.9,
    )
    plans = [_extract_exploration(t) for t in texts]
    return plans, total


def _targeted_reexplore(
    model,
    tokenizer,
    question: str,
    stuck_keywords: List[str],
    max_explore_tokens: int,
    device: str,
) -> Tuple[List[str], int]:
    """Sample a single new exploration plan with an explicit negative constraint
    on the approach that caused the model to loop.

    Called only as a last resort — when all K original plans share the same
    stuck point and no unused alternative is available.  Uses a higher
    temperature than normal exploration to encourage genuinely different paths.
    """
    prompt = build_targeted_reexplore_prompt(question, stuck_keywords, tokenizer)
    texts, total = generate_with_process_loop_guard(
        model, tokenizer, [prompt], max_explore_tokens,
        temperature=0.95,
        do_sample=True,
        stop_token_ids=[TOKEN_EXPLORATION_END],
        device=device,
        threshold=0.9,
    )
    plans = [_extract_exploration(t) for t in texts]
    return [p for p in plans if p.strip()], total


def run_execution(
    model, tokenizer, question: str, exploration: str, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    prompt = build_e2c_prompt(question, exploration, tokenizer)
    do_sample = temperature > 0
    out_text, total, stopped_by_loop = _generate_single_with_chunk_guard(
        model,
        tokenizer,
        prompt,
        max_exec_tokens,
        temperature,
        do_sample,
        device,
        stop_token_ids=[TOKEN_EXECUTION_END],
    )
    # Only rescue on strong looping or empty answer extraction; keep original as default.
    if stopped_by_loop or _is_result_degenerate(out_text):
        rescue_tokens = min(512, max(128, max_exec_tokens // 8)) if max_exec_tokens > 0 else 128
        rescue_text, t_res = _generate_boxed_only_from_exploration(
            model,
            tokenizer,
            question,
            exploration,
            rescue_tokens,
            device,
        )
        total += t_res
        if _extract_answer(rescue_text):
            return rescue_text, total
    return out_text, total


def run_full_cot(
    model, tokenizer, question: str, max_tokens: int,
    temperature: float, do_sample: bool, device: str
) -> Tuple[str, int]:
    prompt = build_full_cot_prompt(question, tokenizer)
    out_text, total, stopped_by_loop = _generate_single_with_chunk_guard(
        model,
        tokenizer,
        prompt,
        max_tokens,
        temperature,
        do_sample,
        device,
    )
    if stopped_by_loop or _is_result_degenerate(out_text):
        rescue_tokens = min(128, max_tokens // 4) if max_tokens > 0 else 64
        rescue_text, t_res = _generate_boxed_only_from_question(
            model,
            tokenizer,
            question,
            rescue_tokens,
            device,
        )
        total += t_res
        if _extract_answer(rescue_text):
            return rescue_text, total
    return out_text, total


# ---------- TTS Methods ----------

def greedy_cot(model, tokenizer, question: str, max_tokens: int, device: str) -> Tuple[str, int]:
    return run_full_cot(model, tokenizer, question, max_tokens, 0.0, False, device)


def self_consistency(
    model, tokenizer, question: str, N: int, max_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    prompt = build_full_cot_prompt(question, tokenizer)
    prompts = [prompt] * N
    texts, total = generate(
        model, tokenizer, prompts, max_tokens,
        temperature=temperature, do_sample=True,
        stop_token_ids=None,
        device=device,
    )
    valid_answers: List[str] = []
    for t in texts:
        if _is_process_degenerate(t, threshold=0.85):
            continue
        ans = _extract_answer(t)
        if ans:
            valid_answers.append(ans)

    if not valid_answers:
        rescue_tokens = min(128, max_tokens // 4) if max_tokens > 0 else 64
        rescue_text, t_res = _generate_boxed_only_from_question(
            model, tokenizer, question, rescue_tokens, device
        )
        total += t_res
        if _extract_answer(rescue_text):
            valid_answers = [_extract_answer(rescue_text)]

    return _majority_vote(valid_answers), total


def e2c_select_lm_judge(
    model, tokenizer, question: str, K: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    judge_prompt = format_llm_judge_prompt(question, plans)
    messages = [{"role": "user", "content": judge_prompt}]
    judge_inp = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chosen, t2 = generate(
        model,
        tokenizer,
        [judge_inp],
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        stop_token_ids=None,
        device=device,
    )
    best_plan = _extract_exploration(chosen[0]) if chosen else plans[0]
    if not best_plan.strip():
        best_plan = plans[0]
    # If the judged plan itself looks degenerate (looping templates), pick
    # an alternative non-degenerate plan from the candidate pool.
    if _is_process_degenerate(best_plan, threshold=0.8):
        for p in plans:
            if p and p.strip() and (not _is_process_degenerate(p, threshold=0.8)):
                best_plan = p
                break
    out, t3 = run_execution(
        model, tokenizer, question, best_plan, max_exec_tokens, 0.0, device
    )
    return _extract_answer(out), t1 + t2 + t3


def _run_verify(
    model,
    tokenizer,
    question: str,
    execution_output: str,
    answer: str,
    max_verify_tokens: int,
    device: str,
) -> Tuple[str, bool, int]:
    """
    Lightweight self-verification pass.

    Ask the model to substitute the proposed answer back into the problem and
    decide if it is correct.  Returns (verify_text, is_correct, tokens_used).

    ``is_correct`` is True only when the model's first non-empty line starts
    with "CORRECT" (case-insensitive).  Any other response (WRONG, empty, or
    malformed) is treated as a failed verification to err on the side of caution.
    """
    verify_prompt = build_verify_prompt(question, execution_output, answer, tokenizer)
    texts, t = generate(
        model,
        tokenizer,
        [verify_prompt],
        max_verify_tokens,
        temperature=0.0,
        do_sample=False,
        stop_token_ids=None,
        device=device,
    )
    verify_text = (texts[0] if texts else "").strip()
    first_line = ""
    for line in verify_text.split("\n"):
        stripped = line.strip()
        if stripped:
            first_line = stripped.upper()
            break
    is_correct = first_line.startswith("CORRECT")
    return verify_text, is_correct, t


def _refine_exploration_after_execution(
    model,
    tokenizer,
    question: str,
    exploration: str,
    execution_output: str,
    max_refine_tokens: int,
    device: str,
    verify_feedback: str = "",
    hallucination_feedback: str = "",
) -> Tuple[str, int, str]:
    """Revise exploration given a failed or incomplete execution (ReAct-style outer loop).

    If ``verify_feedback`` is non-empty, the refine prompt includes the
    verification diagnosis.  If ``hallucination_feedback`` is non-empty, the
    refine prompt includes the self-negated formula(s) that must be avoided.

    Returns (revised_plan, token_count, raw_model_output).
    """
    excerpt = execution_output.strip()
    if len(excerpt) > 3500:
        excerpt = excerpt[-3500:]
    guide = (exploration or "").strip()
    if len(guide) > 5000:
        guide = guide[:5000]
    user_content = format_refine_exploration_prompt(
        question, guide, excerpt, verify_feedback, hallucination_feedback
    )
    messages = [{"role": "user", "content": user_content}]
    refine_inp = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    texts, t = generate(
        model,
        tokenizer,
        [refine_inp],
        max_refine_tokens,
        temperature=0.3,
        do_sample=True,
        stop_token_ids=None,
        device=device,
    )
    raw = (texts[0] if texts else "").strip()
    new_plan = _extract_exploration(raw) if raw else ""
    if not new_plan.strip():
        # Fallback: use raw but still strip any execution-header prefix
        new_plan = _strip_execution_leak(raw)
    max_plan_chars = 6000
    if len(new_plan) > max_plan_chars:
        new_plan = new_plan[:max_plan_chars]
    return new_plan.strip(), t, raw


def e2c_react_loop(
    model,
    tokenizer,
    question: str,
    K: int,
    max_explore_tokens: int,
    max_exec_tokens: int,
    temperature: float,
    device: str,
    max_refine_rounds: int = 3,
    max_refine_tokens: int = 768,
    max_verify_tokens: int = 512,  # kept for API compatibility, no longer used
    trace: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int]:
    """
    E2C + execute–refine outer loop (ReAct-like feedback on exploration):
    1. Sample K explorations, LM-judge picks one plan (same as e2c_select_lm_judge).
    2. run_execution with that plan.
    3. If a \\boxed{} answer is found → return it immediately (trust the model).
    4. If no answer is found (execution stalled / stuck) → refine the exploration
       plan using hallucination feedback and retry.
    5. Stop when an answer is produced or max_refine_rounds is exhausted.

    If ``trace`` is a dict, it is filled with full intermediate outputs for JSON logging.
    """
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        if trace is not None:
            trace.clear()
            trace["question"] = question
            trace["exploration_plans"] = []
            trace["tokens_explore"] = t1
            trace["error"] = "no_plans"
        return "", t1
    judge_prompt = format_llm_judge_prompt(question, plans)
    messages = [{"role": "user", "content": judge_prompt}]
    judge_inp = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chosen, t2 = generate(
        model,
        tokenizer,
        [judge_inp],
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        stop_token_ids=None,
        device=device,
    )
    judge_raw = (chosen[0] if chosen else "") or ""
    best_plan = _extract_exploration(judge_raw) if chosen else plans[0]
    if not best_plan.strip():
        best_plan = plans[0]
    if _is_process_degenerate(best_plan, threshold=0.8):
        for p in plans:
            if p and p.strip() and (not _is_process_degenerate(p, threshold=0.8)):
                best_plan = p
                break

    if trace is not None:
        trace.clear()
        trace["question"] = question
        trace["exploration_plans"] = list(plans)
        trace["judge_raw"] = judge_raw
        trace["selected_plan"] = best_plan
        trace["tokens_explore"] = t1
        trace["tokens_judge"] = t2
        trace["rounds"] = []

    total_tokens = t1 + t2
    exploration = best_plan
    last_out = ""
    # Keep a queue of unused original plans to fall back to on stall.
    # Plans are deduplicated against the selected plan by vocabulary similarity.
    _unused_plans = [
        p for p in plans
        if p.strip() and not _plans_are_too_similar(best_plan, p, threshold=0.75)
    ]

    for round_idx in range(max_refine_rounds + 1):
        out, t_e = run_execution(
            model, tokenizer, question, exploration, max_exec_tokens, 0.0, device
        )
        total_tokens += t_e
        last_out = out
        ans = _extract_final_answer(out)
        round_info: Dict[str, Any] = {
            "round_index": round_idx,
            "exploration_used": exploration,
            "execution_output": out,
            "extracted_answer": ans or "",
            "tokens_execution": t_e,
        }
        if trace is not None:
            trace["rounds"].append(round_info)

        if ans:
            # Answer found — return immediately, no verification gate.
            if trace is not None:
                trace["final_answer"] = ans
                trace["total_tokens"] = total_tokens
            return ans, total_tokens

        if round_idx >= max_refine_rounds:
            break

        # No answer produced — execution stalled or stuck; refine the plan.
        hallucination_feedback = _extract_hallucination_feedback(out)
        if trace is not None and trace["rounds"] and hallucination_feedback:
            round_info["hallucination_detected"] = True

        exploration_new, t_r, refine_raw = _refine_exploration_after_execution(
            model,
            tokenizer,
            question,
            exploration,
            out,
            max_refine_tokens,
            device,
            verify_feedback="",
            hallucination_feedback=hallucination_feedback,
        )
        total_tokens += t_r
        new_plan = exploration_new.strip()
        stalled = _plans_are_too_similar(exploration, new_plan)
        if trace is not None and trace["rounds"]:
            trace["rounds"][-1]["refine_raw"] = refine_raw
            trace["rounds"][-1]["refined_plan"] = new_plan
            trace["rounds"][-1]["tokens_refine"] = t_r
            if stalled:
                trace["rounds"][-1]["refine_stalled"] = True

        # --- Next-plan selection (execution produced no answer) ---
        #
        # A. Hallucination detected (model self-negated a formula):
        #    Bypass refine's output — find an unused plan that AVOIDS the stuck point.
        #    A1. Unused plan that avoids stuck keywords → use it.
        #    A2. No such plan but refine gave something new → use refine's plan
        #        (it carries the hallucination warning).
        #    A3. Refine also stalled → any remaining unused plan.
        #    A4. All plans exhausted → targeted re-exploration with negative constraint.
        #
        # B. No hallucination, refine gave a new plan → use it.
        #
        # C. No hallucination, refine stalled → any remaining unused plan.

        stuck_kws: List[str] = (
            _extract_stuck_keywords(out) if hallucination_feedback else []
        )

        if stuck_kws:
            # --- Branch A: hallucination-driven path switch ---
            chosen_fallback: str = ""
            chosen_idx: int = -1
            for i, p in enumerate(_unused_plans):
                if _plan_avoids_stuck_keywords(p, stuck_kws):
                    chosen_fallback = p
                    chosen_idx = i
                    break

            if chosen_idx >= 0:
                # A1: found a plan that goes around the stuck point
                _unused_plans.pop(chosen_idx)
                exploration = chosen_fallback
                if trace is not None and trace["rounds"]:
                    trace["rounds"][-1]["hallucination_bypass_plan"] = chosen_fallback
                    trace["rounds"][-1]["fallback_avoids_stuck"] = True
            elif new_plan and not stalled:
                # A2: no bypass plan, but refine gave a new plan that already
                #     carries the hallucination warning — use it
                exploration = new_plan
            elif _unused_plans:
                # A3: refine stalled too; try any remaining unused plan
                fallback = _unused_plans.pop(0)
                exploration = fallback
                if trace is not None and trace["rounds"]:
                    trace["rounds"][-1]["stall_fallback_plan"] = fallback
            elif round_idx < max_refine_rounds:
                # A4: all original plans exhausted — targeted re-exploration
                new_plans, t_re = _targeted_reexplore(
                    model, tokenizer, question, stuck_kws,
                    max_explore_tokens, device,
                )
                total_tokens += t_re
                if new_plans:
                    exploration = new_plans[0]
                    if trace is not None and trace["rounds"]:
                        trace["rounds"][-1]["targeted_reexplore_plan"] = exploration
                        trace["rounds"][-1]["tokens_reexplore"] = t_re

        elif new_plan and not stalled:
            # --- Branch B: clean refine, no hallucination ---
            exploration = new_plan

        elif stalled:
            # --- Branch C: plain stall, no hallucination signal ---
            if _unused_plans:
                fallback = _unused_plans.pop(0)
                exploration = fallback
                if trace is not None and trace["rounds"]:
                    trace["rounds"][-1]["stall_fallback_plan"] = fallback

    # Prefer the stricter final-answer extractor; fall back to any boxed if none found.
    final_ans = _extract_final_answer(last_out) or _extract_answer(last_out) or ""
    if trace is not None:
        trace["final_answer"] = final_ans
        trace["total_tokens"] = total_tokens
        trace["last_execution_output"] = last_out
    return final_ans, total_tokens


def e2c_select_semantic_cluster(
    model, tokenizer, question: str, K: int, M: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str,
    encoder=None,
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    if encoder is None:
        try:
            from util.embedding import get_encoder
            encoder = get_encoder(backend="auto")
        except Exception:
            # Fallback: random selection
            centroids = plans[:min(M, len(plans))]
            weights = [1.0] * len(centroids)
            answers = []
            total_exec = 0
            for c in centroids:
                out, t = run_execution(model, tokenizer, question, c, max_exec_tokens, 0.0, device)
                answers.append(_extract_answer(out))
                total_exec += t
            return _weighted_majority(answers, weights), t1 + total_exec
    emb = encoder.encode(plans)
    # L2-normalize for cosine similarity (paper A.4)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-8)
    from sklearn.cluster import KMeans
    n_clusters = min(M, len(plans))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(emb)
    labels = kmeans.labels_
    centroid_plans = []
    weights = []
    for i in range(n_clusters):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        center = kmeans.cluster_centers_[i]
        dists = np.linalg.norm(emb[idx] - center, axis=1)
        best_local = idx[np.argmin(dists)]
        centroid_plans.append(plans[best_local])
        weights.append(len(idx))
    answers = []
    total_exec = 0
    for p in centroid_plans:
        out, t = run_execution(model, tokenizer, question, p, max_exec_tokens, 0.0, device)
        answers.append(_extract_answer(out))
        total_exec += t
    return _weighted_majority(answers, weights), t1 + total_exec


def e2c_sc(
    model, tokenizer, question: str, K: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    answers = []
    total_exec = 0
    for p in plans:
        out, t = run_execution(model, tokenizer, question, p, max_exec_tokens, 0.0, device)
        answers.append(_extract_answer(out))
        total_exec += t
    return _majority_vote(answers), t1 + total_exec


def e2c_rp(
    model, tokenizer, question: str, K: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    p = random.choice(plans)
    out, t2 = run_execution(model, tokenizer, question, p, max_exec_tokens, 0.0, device)
    return _extract_answer(out), t1 + t2


def _tot_explore(
    model, tokenizer, question: str, N: int, thought_tokens: int,
    temperature: float, device: str,
) -> Tuple[List[str], int]:
    """
    Run only the ToT tree expansion; return leaf nodes (full reasoning prefixes)
    and total tokens used. Used by e2c_tot for Explore phase.
    """
    import math
    if N <= 0:
        return [], 0
    branch_factor = 2
    depth = max(1, math.ceil(math.log(N) / math.log(branch_factor)))
    total_tokens = 0
    root_prompt = build_full_cot_prompt(question, tokenizer)
    current_nodes: List[str] = [root_prompt]
    for _ in range(depth):
        next_nodes: List[str] = []
        prompts = []
        parent_indices = []
        for idx, node in enumerate(current_nodes):
            if len(prompts) >= N * branch_factor:
                break
            for _ in range(branch_factor):
                prompts.append(node)
                parent_indices.append(idx)
        if not prompts:
            break
        continuations, t = generate_with_process_loop_guard(
            model,
            tokenizer,
            prompts,
            thought_tokens,
            temperature=temperature,
            do_sample=True,
            stop_token_ids=None,
            device=device,
            max_retries=1,
            threshold=0.8,
        )
        total_tokens += t
        for cont, p_idx in zip(continuations, parent_indices):
            child = current_nodes[p_idx] + cont
            next_nodes.append(child)
        if not next_nodes:
            break
        current_nodes = next_nodes[:N]
    return current_nodes, total_tokens


def tree_of_thoughts(
    model, tokenizer, question: str, N: int, max_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    """
    Tree-of-Thoughts:
    - Maintain a reasoning tree where each node is a partial CoT trajectory.
    - At each depth, expand every node into `branch_factor` children by *continuing*
      the existing reasoning (not restarting from scratch).
    - After a fixed search depth, take the leaf nodes and let the model complete
      each leaf into a full solution, then majority-vote over boxed answers.

    N: approximate leaf budget / beam width (controls how many nodes we keep).
    """
    import math

    if N <= 0:
        return "", 0

    # Branching factor and search depth
    branch_factor = 2
    # Depth grows ~log_b(N), but at least 1
    depth = max(1, math.ceil(math.log(N) / math.log(branch_factor)))

    # Token budget per *expansion* step; keep it relatively small so we can
    # explore multiple levels before spending full budget in the final stage.
    thought_tokens = min(256, max_tokens // 4)

    total_tokens = 0

    # Each "node" is a full text prefix that the model will continue from.
    # Start with a single root node: the full CoT-style prompt, no reasoning yet.
    root_prompt = build_full_cot_prompt(question, tokenizer)
    current_nodes: List[str] = [root_prompt]

    # ---- Exploration: grow the reasoning tree ----
    for _ in range(depth):
        next_nodes: List[str] = []
        prompts = []
        parent_indices = []

        # Prepare prompts: for each current node, sample `branch_factor` continuations
        for idx, node in enumerate(current_nodes):
            # Global cap: don't create far more children than we will ever keep
            if len(prompts) >= N * branch_factor:
                break
            for _ in range(branch_factor):
                prompts.append(node)
                parent_indices.append(idx)

        if not prompts:
            break

        # Generate continuations in a single batched call
        continuations, t = generate_with_process_loop_guard(
            model,
            tokenizer,
            prompts,
            thought_tokens,
            temperature=temperature,
            do_sample=True,
            stop_token_ids=None,
            device=device,
            max_retries=1,
            threshold=0.8,
        )
        total_tokens += t

        # Reconstruct full node texts (parent prefix + continuation)
        for cont, p_idx in zip(continuations, parent_indices):
            parent = current_nodes[p_idx]
            child = parent + cont
            next_nodes.append(child)

        if not next_nodes:
            break

        # Beam-like pruning: keep at most N nodes (no complex scoring for now)
        current_nodes = next_nodes[:N]

    if not current_nodes:
        return "", total_tokens

    # ---- Execution: from each leaf node, let the model complete to full answer ----
    answers: List[str] = []
    for leaf in current_nodes[:N]:
        # Continue from the leaf's reasoning; do not restart from scratch.
        full_texts, t = generate_with_process_loop_guard(
            model,
            tokenizer,
            [leaf],
            max_tokens,
            temperature=temperature,
            do_sample=True,
            stop_token_ids=None,
            device=device,
            max_retries=1,
            threshold=0.8,
        )
        total_tokens += t
        if full_texts:
            full_output = leaf + full_texts[0]
            ans = _extract_answer(full_output)
            if not ans:
                rescue_tokens = min(128, max_tokens // 4) if max_tokens > 0 else 64
                rescue_text, t_res = _generate_boxed_only_from_question(
                    model, tokenizer, question, rescue_tokens, device
                )
                total_tokens += t_res
                ans = _extract_answer(rescue_text)
            answers.append(ans)

    return _majority_vote(answers), total_tokens


def e2c_tot(
    model, tokenizer, question: str, K: int, M: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str,
    encoder=None,
) -> Tuple[str, int]:
    """
    E2C + ToT: Use ToT to produce diverse reasoning leaves (Explore), then E2C-style
    contrast (semantic cluster or random fallback) to select representative plans,
    then Execute each with build_e2c_prompt and weighted majority vote.
    """
    thought_tokens = min(256, max_explore_tokens // 2)
    leaves, t1 = _tot_explore(
        model, tokenizer, question, K, thought_tokens, temperature, device
    )
    if not leaves:
        return "", t1

    root_prompt = build_full_cot_prompt(question, tokenizer)
    # Extract reasoning from each leaf (strip the initial prompt)
    max_plan_chars = 6000
    plans = []
    for leaf in leaves:
        if leaf.startswith(root_prompt):
            plan = leaf[len(root_prompt):].strip()
        else:
            plan = leaf.strip()
        if plan:
            plans.append(plan[:max_plan_chars] if len(plan) > max_plan_chars else plan)
    if not plans:
        return "", t1

    if encoder is None:
        try:
            from util.embedding import get_encoder
            encoder = get_encoder(backend="auto")
        except Exception:
            encoder = None
    if encoder is None:
        centroids = plans[: min(M, len(plans))]
        weights = [1.0] * len(centroids)
        answers = []
        total_exec = 0
        for c in centroids:
            out, t = run_execution(model, tokenizer, question, c, max_exec_tokens, 0.0, device)
            answers.append(_extract_answer(out))
            total_exec += t
        return _weighted_majority(answers, weights), t1 + total_exec

    emb = encoder.encode(plans)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-8)
    from sklearn.cluster import KMeans
    n_clusters = min(M, len(plans))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(emb)
    labels = kmeans.labels_
    centroid_plans = []
    weights = []
    for i in range(n_clusters):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        center = kmeans.cluster_centers_[i]
        dists = np.linalg.norm(emb[idx] - center, axis=1)
        best_local = idx[np.argmin(dists)]
        centroid_plans.append(plans[best_local])
        weights.append(len(idx))
    answers = []
    total_exec = 0
    for p in centroid_plans:
        out, t = run_execution(model, tokenizer, question, p, max_exec_tokens, 0.0, device)
        answers.append(_extract_answer(out))
        total_exec += t
    return _weighted_majority(answers, weights), t1 + total_exec


def e2c_tot_lm_judge(
    model, tokenizer, question: str, K: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str,
) -> Tuple[str, int]:
    """
    E2C + ToT with LM judge: Use ToT to produce diverse reasoning leaves (Explore),
    then LM judge to select the single best plan (Contrast), then Execute once.
    """
    thought_tokens = min(256, max_explore_tokens // 2)
    leaves, t1 = _tot_explore(
        model, tokenizer, question, K, thought_tokens, temperature, device
    )
    if not leaves:
        return "", t1

    root_prompt = build_full_cot_prompt(question, tokenizer)
    max_plan_chars = 6000
    plans = []
    for leaf in leaves:
        if leaf.startswith(root_prompt):
            plan = leaf[len(root_prompt):].strip()
        else:
            plan = leaf.strip()
        if plan:
            plans.append(plan[:max_plan_chars] if len(plan) > max_plan_chars else plan)
    if not plans:
        return "", t1

    judge_prompt = format_llm_judge_prompt(question, plans)
    messages = [{"role": "user", "content": judge_prompt}]
    judge_inp = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chosen, t2 = generate(
        model,
        tokenizer,
        [judge_inp],
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        stop_token_ids=None,
        device=device,
    )
    best_plan = _extract_exploration(chosen[0]) if chosen else plans[0]
    if not best_plan.strip():
        best_plan = plans[0]
    if _is_process_degenerate(best_plan, threshold=0.8):
        for p in plans:
            if p and p.strip() and (not _is_process_degenerate(p, threshold=0.8)):
                best_plan = p
                break
    out, t3 = run_execution(
        model, tokenizer, question, best_plan, max_exec_tokens, 0.0, device
    )
    return _extract_answer(out), t1 + t2 + t3


def e2c_tot_layered(
    model, tokenizer, question: str,
    branch_factor: int,
    max_depth: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str,
) -> Tuple[str, int]:
    """
    E2C + ToT with per-layer LM judge (atomic explore → judge → [execute or next layer]).
    - Each parent has exactly branch_factor children (exploration only, no execution).
    - Judge outputs READY/CONTINUE: READY = execute and stop; CONTINUE = expand selected node (or execute if at max_depth).
    - Depth is adaptive: stop when judge says READY, or when depth >= max_depth (safety cap).
    - Budget in config = branch_factor (e.g. 4 = 4 children per parent).
    """
    if branch_factor <= 0 or max_depth <= 0:
        return "", 0
    root_prompt = build_full_cot_prompt(question, tokenizer)
    current_parents: List[str] = [root_prompt]
    total_tokens = 0
    thought_tokens = min(256, max_explore_tokens // 2)
    max_plan_chars = 6000

    for depth in range(max_depth):
        next_leaves: List[str] = []
        for parent in current_parents:
            prompts = [parent] * branch_factor
            continuations, t = generate_with_process_loop_guard(
                model,
                tokenizer,
                prompts,
                thought_tokens,
                temperature=temperature,
                do_sample=True,
                stop_token_ids=None,
                device=device,
                max_retries=1,
                threshold=0.8,
            )
            total_tokens += t
            for c in continuations:
                next_leaves.append(parent + c)
        if not next_leaves:
            break

        plans: List[str] = []
        for leaf in next_leaves:
            if leaf.startswith(root_prompt):
                plan = leaf[len(root_prompt):].strip()
            else:
                plan = leaf.strip()
            plans.append(plan[:max_plan_chars] if len(plan) > max_plan_chars else plan)

        judge_prompt = format_llm_judge_ready_prompt(question, plans)
        messages = [{"role": "user", "content": judge_prompt}]
        judge_inp = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chosen, t = generate(
            model,
            tokenizer,
            [judge_inp],
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            stop_token_ids=None,
            device=device,
        )
        total_tokens += t
        raw_judge = (chosen[0] or "").strip()
        first_line, _, rest = raw_judge.partition("\n")
        first_line = first_line.strip().upper()
        is_ready = "READY" in first_line
        best_plan = rest.strip() if rest else _extract_exploration(chosen[0]) if chosen else ""
        if not best_plan.strip():
            best_plan = plans[0]
        plan_degen_flags = [_is_process_degenerate(p, threshold=0.8) for p in plans]

        selected_leaf = None
        selected_idx = None
        best_plan_s = best_plan.strip()
        for i, p in enumerate(plans):
            ps = p.strip()
            if ps == best_plan_s or (best_plan_s in ps) or (ps in best_plan_s):
                selected_leaf = next_leaves[i]
                selected_idx = i
                break
        if selected_leaf is None:
            selected_leaf = next_leaves[0]
            selected_idx = 0

        # If judge wants to READY but the chosen plan looks looping/repetitive,
        # don't stop the whole search. Keep exploring with an alternative candidate.
        if is_ready and depth < max_depth - 1:
            if selected_idx is not None and plan_degen_flags[selected_idx]:
                is_ready = False
                for j in range(len(plans)):
                    if not plan_degen_flags[j] and plans[j].strip():
                        selected_leaf = next_leaves[j]
                        selected_idx = j
                        best_plan = plans[j]
                        break

        should_execute = is_ready or (depth >= max_depth - 1)
        if should_execute:
            plan_for_exec = best_plan if best_plan.strip() else (
                selected_leaf[len(root_prompt):].strip() if selected_leaf.startswith(root_prompt) else selected_leaf
            )
            out, t = run_execution(
                model, tokenizer, question, plan_for_exec, max_exec_tokens, 0.0, device
            )
            total_tokens += t
            return _extract_answer(out), total_tokens

        current_parents = [selected_leaf]

    if current_parents:
        leaf = current_parents[0]
        plan = leaf[len(root_prompt):].strip() if leaf.startswith(root_prompt) else leaf.strip()
        plan = plan[:max_plan_chars] if len(plan) > max_plan_chars else plan
        out, t = run_execution(model, tokenizer, question, plan, max_exec_tokens, 0.0, device)
        total_tokens += t
        return _extract_answer(out), total_tokens
    return "", total_tokens


def forest_of_thought(
    model, tokenizer, question: str, N: int, max_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    # FoT: N trees, 2-step expand each, then one full chain per tree, vote
    thought_tokens = min(256, max_tokens // 4)
    total_tokens = 0
    representatives = []
    for _ in range(N):
        node = question
        for _ in range(2):
            prompt = build_full_cot_prompt(question, tokenizer)
            texts, t = generate_with_process_loop_guard(
                model,
                tokenizer,
                [prompt],
                thought_tokens,
                temperature=temperature,
                do_sample=True,
                stop_token_ids=None,
                device=device,
                max_retries=1,
                threshold=0.8,
            )
            total_tokens += t
            node = texts[0] if texts else node
        representatives.append(node)
    answers = []
    for rep in representatives:
        prompt = build_full_cot_prompt(question, tokenizer)
        texts, t = generate_with_process_loop_guard(
            model,
            tokenizer,
            [prompt],
            max_tokens,
            temperature=temperature,
            do_sample=True,
            stop_token_ids=None,
            device=device,
            max_retries=1,
            threshold=0.8,
        )
        total_tokens += t
        text0 = texts[0] if texts else ""
        ans = _extract_answer(text0)
        if not ans:
            rescue_tokens = min(128, max_tokens // 4) if max_tokens > 0 else 64
            rescue_text, t_res = _generate_boxed_only_from_question(
                model, tokenizer, question, rescue_tokens, device
            )
            total_tokens += t_res
            ans = _extract_answer(rescue_text)
        answers.append(ans)
    return _majority_vote(answers), total_tokens


def evaluate_predictions(predictions: List[str], ground_truths: List[str]) -> float:
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        succ, _ = boxed_evaluate(pred, gt)
        if not succ and pred and "boxed" not in pred:
            succ = check_answer_match(pred, gt)
        if succ:
            correct += 1
    return correct / len(predictions) * 100.0 if predictions else 0.0
