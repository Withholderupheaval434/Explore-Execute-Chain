"""
Answer normalization and extraction for math evaluation.
From E2C paper and OpenQA standards (AIME, MATH).
"""
import re

# Constants for normalization (from E2C/OpenQA)
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
    "\\dots",
]


def keep_lowercase_and_digits(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isascii() and ch.isalnum())


def keep_only_digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def calculate_frac(s: str):
    pattern = r"frac\{(-?\d+)\}\{(-?\d+)\}"
    match = re.search(pattern, s)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        if den != 0:
            return num / den
    return None


def normalize_final_answer(final_answer: str):
    """Normalize a final answer string for math evaluation."""
    final_answer = str(final_answer).split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    if "frac" in final_answer:
        float_answer = calculate_frac(final_answer)
        final_answer = keep_only_digits(final_answer)
    else:
        final_answer = keep_lowercase_and_digits(final_answer)
        float_answer = None
    final_answer = final_answer.lstrip("0")
    if final_answer == "":
        final_answer = "0"
    return final_answer, float_answer


def extract_boxed_content(s: str) -> list:
    """Extract all content inside \\boxed{}."""
    results = []
    brace_level = 0
    start_index = -1
    i = 0
    while i < len(s):
        if s[i : i + 6] == "boxed{":
            if brace_level == 0:
                start_index = i + 6
            brace_level += 1
            i += 6
            continue
        if brace_level > 0:
            if s[i] == "{":
                brace_level += 1
            elif s[i] == "}":
                brace_level -= 1
                if brace_level == 0:
                    results.append(s[start_index:i])
                    start_index = -1
        i += 1
    return results


def boxed_evaluate(pred: str, gt) -> tuple:
    """Check if prediction matches ground truth. Returns (correct, pred_answer)."""
    gt_norm, p_float = normalize_final_answer(gt)
    for pre_answer in extract_boxed_content(pred):
        pre_norm, q_float = normalize_final_answer(pre_answer)
        if pre_norm == gt_norm or (q_float is not None and p_float is not None and abs(q_float - p_float) < 1e-6):
            return True, pre_answer
    return False, ""


def check_answer_match(pred_answer: str, gt) -> bool:
    """Check if extracted answer string matches gt (for voting outputs)."""
    gt_norm, p_float = normalize_final_answer(gt)
    pre_norm, q_float = normalize_final_answer(pred_answer)
    if pre_norm == gt_norm:
        return True
    if q_float is not None and p_float is not None and abs(q_float - p_float) < 1e-6:
        return True
    return False
