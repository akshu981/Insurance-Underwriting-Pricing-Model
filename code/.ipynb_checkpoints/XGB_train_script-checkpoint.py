"""
training_script_xgb.py
SageMaker Training job: fits the challenger XGBoost model on processed frequency data.

Input (channel "train"): /opt/ml/input/data/train/freq_train.csv
Output: /opt/ml/model/xgb_model.json
        /opt/ml/model/feature_cols.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-channel", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--num-round", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load processed training data ----
    train_path = os.path.join(args.train_channel, "freq_train.csv")
    freq_train = pd.read_csv(train_path)
    print("Loaded train shape:", freq_train.shape)

    # ---- Feature set: raw continuous vars + existing dummies, no binning needed ----
    exclude_cols = ["IDpol", "ClaimNb", "Exposure"]
    feature_cols = [c for c in freq_train.columns if c not in exclude_cols]
    print("Feature count:", len(feature_cols))
    print("Features:", feature_cols)

    X_train = freq_train[feature_cols]
    y_train = freq_train["ClaimNb"]
    base_margin_train = np.log(freq_train["Exposure"])

    # ---- Build DMatrix with exposure as base_margin (same role as GLM's offset) ----
    dtrain = xgb.DMatrix(X_train, label=y_train, base_margin=base_margin_train)

    # ---- Train ----
    params = {
        "objective": "count:poisson",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "eval_metric": "poisson-nloglik",
    }
    booster = xgb.train(params, dtrain, num_boost_round=args.num_round)

    print("Training complete.")

    # ---- Save artifacts ----
    os.makedirs(args.model_dir, exist_ok=True)

    booster.save_model(os.path.join(args.model_dir, "xgb_model.json"))

    with open(os.path.join(args.model_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print("Saved model artifacts to", args.model_dir)


if __name__ == "__main__":
    main()