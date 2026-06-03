"""
Split the raw SFT parquet file into train/val sets.
Called by scripts/prepare_all_data.sh.

Input:  data/raw/sft/e2c-sft.parquet
Output: data/processed/sft/e2c-sft-train.parquet
        data/processed/sft/e2c-sft-val.parquet
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_file = input_dir / "e2c-sft.parquet"
    if not sft_file.exists():
        raise FileNotFoundError(f"SFT data not found: {sft_file}")

    print(f"Loading {sft_file} ...")
    df = pd.read_parquet(sft_file)
    print(f"  Total samples: {len(df)}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * args.train_ratio)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]

    train_out = output_dir / "e2c-sft-train.parquet"
    val_out = output_dir / "e2c-sft-val.parquet"

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    print(f"  Train: {len(train_df)} samples -> {train_out}")
    print(f"  Val:   {len(val_df)} samples -> {val_out}")
    print("SFT data preparation complete.")


if __name__ == "__main__":
    main()
