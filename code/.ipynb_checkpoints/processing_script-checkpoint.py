"""
processing_script.py
SageMaker Processing job: cleans and encodes the frequency dataset.

Input:  /opt/ml/processing/input/freMTPL2freq.csv
Output: /opt/ml/processing/output/train/freq_train.csv
        /opt/ml/processing/output/test/freq_test.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--brand-threshold", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = "/opt/ml/processing/input/freMTPL2freq.csv"
    train_output_dir = "/opt/ml/processing/output/train"
    test_output_dir = "/opt/ml/processing/output/test"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # ---- Load ----
    df_freq = pd.read_csv(input_path)
    print("Loaded shape:", df_freq.shape)

    # ---- Type consistency ----
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)

    # ---- Cleaning: drop Area, log-transform Density ----
    df_freq = df_freq.drop(columns=["Area"])
    df_freq["log_density"] = np.log1p(df_freq["Density"])
    df_freq = df_freq.drop(columns=["Density"])

    # ---- Cap ClaimNb outliers, filter Exposure ----
    df_freq["ClaimNb"] = df_freq["ClaimNb"].clip(upper=5)
    df_freq = df_freq[df_freq["Exposure"] <= 1.0]
    print("After cleaning/filtering:", df_freq.shape)

    # ---- VehBrand grouping: rare brands -> "Other" ----
    brand_counts = df_freq["VehBrand"].value_counts(normalize=True)
    rare_brands = brand_counts[brand_counts < args.brand_threshold].index.tolist()
    print("Brands grouped as 'Other':", rare_brands)
    df_freq["VehBrand_grouped"] = df_freq["VehBrand"].apply(
        lambda x: "Other" if x in rare_brands else x
    )

    # ---- VehGas binary ----
    df_freq["VehGas_binary"] = (df_freq["VehGas"] == "Diesel").astype(int)

    # ---- One-hot encoding ----
    df_encoded = pd.get_dummies(
        df_freq, columns=["VehBrand_grouped", "Region"], drop_first=True
    )
    df_encoded = df_encoded.drop(columns=["VehBrand", "VehGas"])

    dummy_cols = [
        c for c in df_encoded.columns
        if c.startswith("VehBrand_grouped_") or c.startswith("Region_")
    ]
    df_encoded[dummy_cols] = df_encoded[dummy_cols].astype(int)

    print("Final encoded shape:", df_encoded.shape)

    # ---- Train/test split ----
    freq_train, freq_test = train_test_split(
        df_encoded, test_size=args.test_size, random_state=args.random_state
    )
    print("Train shape:", freq_train.shape, "Test shape:", freq_test.shape)

    # ---- Save outputs ----
    freq_train.to_csv(os.path.join(train_output_dir, "freq_train.csv"), index=False)
    freq_test.to_csv(os.path.join(test_output_dir, "freq_test.csv"), index=False)
    print("Saved train/test CSVs.")


if __name__ == "__main__":
    main()