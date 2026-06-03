#!/usr/bin/env python
"""Download AIME 2024 dataset to data/aime2024/"""
import os
from pathlib import Path

def main():
    out_dir = Path(__file__).resolve().parent.parent / "data" / "aime2024"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aime2024.parquet"

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return

    print("Downloading HuggingFaceH4/aime_2024...")
    ds = load_dataset("HuggingFaceH4/aime_2024")
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    # Normalize columns to question, answer
    col_map = {"problem": "question", "Problem": "question", "prompt": "question",
               "Answer": "answer", "response": "answer"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "question" not in df.columns:
        df["question"] = df.iloc[:, 0]
    if "answer" not in df.columns:
        df["answer"] = df.iloc[:, 1] if df.shape[1] > 1 else ""
    df = df[["question", "answer"]].copy()
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
