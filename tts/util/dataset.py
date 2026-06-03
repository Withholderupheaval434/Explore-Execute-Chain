"""Dataset loading for AIME and math benchmarks."""
import os
from datasets import Dataset


DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
AIME2024_PATH = os.path.join(DEFAULT_BASE_DIR, "aime2024", "aime2024.parquet")
MAX_TOKENS_AIME = 8192


def _standardize_item(item: dict, idx: int) -> dict:
    question = item.get("question", item.get("Question", item.get("problem", item.get("prompt", ""))))
    answer = item.get("answer", item.get("Answer", item.get("response", "")))
    if not question:
        raise ValueError(f"Item {idx} missing 'question' field")
    return {"question": str(question), "answer": str(answer).strip() if answer else ""}


def load_aime2024(base_dir: str = None) -> list:
    """Load AIME 2024 benchmark. Returns list of {question, answer}."""
    base_dir = base_dir or DEFAULT_BASE_DIR
    path = os.path.join(base_dir, "aime2024", "aime2024.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AIME 2024 dataset not found: {path}\n"
            "Please download it. See data/aime2024/README.md"
        )
    ds = Dataset.from_parquet(path)
    data_list = ds.to_list()
    return [_standardize_item(item, i) for i, item in enumerate(data_list)]
