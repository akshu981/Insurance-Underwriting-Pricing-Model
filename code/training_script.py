"""
training_script.py
SageMaker Training job: fits the champion GLM (v3) on processed frequency data.

Input (SageMaker Training channel "train"): /opt/ml/input/data/train/freq_train.csv
Output: /opt/ml/model/glm_model.pkl
        /opt/ml/model/bin_edges.pkl
        /opt/ml/model/feature_cols.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-channel", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    return parser.parse_args()


def get_tree_bins(train_df, col, max_leaf_nodes, min_samples_leaf):
    X = train_df[[col]].values
    y = train_df["ClaimNb"] / train_df["Exposure"]
    w = train_df["Exposure"].values
    tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
    tree.fit(X, y, sample_weight=w)
    thresholds = sorted(set(tree.tree_.threshold[tree.tree_.threshold != -2]))
    return [-np.inf] + thresholds + [np.inf]


def main():
    args = parse_args()

    # ---- Load processed training data ----
    train_path = os.path.join(args.train_channel, "freq_train.csv")
    freq_train = pd.read_csv(train_path)
    print("Loaded train shape:", freq_train.shape)

    # ---- Recompute tree-based bin edges fresh on this training data ----
    bin_vars = ["VehPower", "VehAge", "DrivAge", "BonusMalus"]
    bin_params = {
        "VehPower": dict(max_leaf_nodes=6, min_samples_leaf=0.02),
        "VehAge": dict(max_leaf_nodes=6, min_samples_leaf=0.02),
        "DrivAge": dict(max_leaf_nodes=7, min_samples_leaf=0.02),
        "BonusMalus": dict(max_leaf_nodes=6, min_samples_leaf=0.04),
    }

    bins = {}
    for col, params in bin_params.items():
        edges = get_tree_bins(freq_train, col, **params)
        bins[col] = edges
        freq_train[f"{col}_bin"] = pd.cut(freq_train[col], bins=edges)
        print(col, "-> bin edges:", edges)

    # ---- Build dummy columns for the bins ----
    bin_dummy_cols = []
    for col in bin_vars:
        dummies = pd.get_dummies(freq_train[f"{col}_bin"], prefix=col, drop_first=True).astype(int)
        freq_train = pd.concat([freq_train, dummies], axis=1)
        bin_dummy_cols += list(dummies.columns)

    # ---- Interaction term ----
    freq_train["BonusMalus_x_DrivAge"] = freq_train["BonusMalus"].astype(float) * freq_train["DrivAge"].astype(float)

    # ---- Final feature set: everything except raw bin_vars, IDs, and target columns ----
    exclude_cols = ["IDpol", "ClaimNb", "Exposure"] + bin_vars + [f"{c}_bin" for c in bin_vars]
    other_cols = [c for c in freq_train.columns if c not in exclude_cols and c not in bin_dummy_cols
                  and c != "BonusMalus_x_DrivAge"]
    feature_cols = other_cols + bin_dummy_cols + ["BonusMalus_x_DrivAge"]

    print("Final feature count:", len(feature_cols))

    # ---- Fit Poisson GLM ----
    X_train = sm.add_constant(freq_train[feature_cols])
    glm_model = sm.GLM(
        freq_train["ClaimNb"],
        X_train,
        family=sm.families.Poisson(),
        offset=np.log(freq_train["Exposure"]),
    ).fit()

    print("Deviance:", glm_model.deviance)
    print(glm_model.summary())

    # ---- Save artifacts ----
    os.makedirs(args.model_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, "glm_model.pkl"), "wb") as f:
        pickle.dump(glm_model, f)

    with open(os.path.join(args.model_dir, "bin_edges.pkl"), "wb") as f:
        pickle.dump(bins, f)

    with open(os.path.join(args.model_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print("Saved model artifacts to", args.model_dir)


if __name__ == "__main__":
    main()