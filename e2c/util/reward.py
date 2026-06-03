import re

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
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def keep_lowercase_and_digits(s: str) -> str:
    return ''.join(ch.lower() for ch in s if ch.isascii() and ch.isalnum())


def keep_only_digits(s: str) -> str:
    return ''.join(ch for ch in s if ch.isdigit())


def calculate_frac(s: str) -> float:
    """Parse a simple frac{a}{b} expression and return a/b, or None."""
    pattern = r'frac\{(-?\d+)\}\{(-?\d+)\}'
    match = re.search(pattern, s)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator != 0:
            return numerator / denominator
    return None


def shift_numbered_list(text: str, k: int) -> str:
    """Shift all numbered-list markers in text by k."""
    def replacer(match):
        num = int(match.group(1))
        return f"{num + k}. "
    return re.sub(r'\b(\d+)\.\s', replacer, text)


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    if 'frac' in final_answer:
        float_answer = calculate_frac(final_answer)
        final_answer = keep_only_digits(final_answer)
    else:
        final_answer = keep_lowercase_and_digits(final_answer)
        float_answer = None

    final_answer = final_answer.lstrip('0')
    if final_answer == '':
        final_answer = '0'
    return final_answer, float_answer


def extract_boxed_content(s: str) -> list:
    """Return a list of all content found inside \\boxed{} blocks."""
    results = []
    brace_level = 0
    start_index = -1
    i = 0
    while i < len(s):
        if s[i:i+6] == "boxed{":
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
