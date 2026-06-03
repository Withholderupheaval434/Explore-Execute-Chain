"""
Split the raw RL parquet file into train/val sets.
Called by scripts/prepare_all_data.sh.

Input:  data/raw/rl/e2c-rl.parquet
        data/raw/rl/e2c-rl-valid.parquet  (used as val directly if present)
Output: data/processed/rl/e2c-rl-train.parquet
        data/processed/rl/e2c-rl-val.parquet
"""

import argparse
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

    train_file = input_dir / "e2c-rl.parquet"
    valid_file = input_dir / "e2c-rl-valid.parquet"

    if not train_file.exists():
        raise FileNotFoundError(f"RL data not found: {train_file}")

    print(f"Loading {train_file} ...")
    train_df = pd.read_parquet(train_file)

    train_out = output_dir / "e2c-rl-train.parquet"
    val_out = output_dir / "e2c-rl-val.parquet"

    if valid_file.exists():
        # Use the dedicated validation split
        print(f"Loading validation data from {valid_file} ...")
        val_df = pd.read_parquet(valid_file)
    else:
        # Fall back to a random split
        print("No dedicated validation file found; splitting from train data.")
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(train_df) * args.train_ratio)
        val_df = train_df.iloc[split:]
        train_df = train_df.iloc[:split]

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    print(f"  Train: {len(train_df)} samples -> {train_out}")
    print(f"  Val:   {len(val_df)} samples -> {val_out}")
    print("RL data preparation complete.")


if __name__ == "__main__":
    main()
