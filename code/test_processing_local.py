"""
test_processing_local.py
Local test of processing_script.py's logic — run this before wiring up SageMaker.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---- Local paths (adjust if your folder structure differs) ----
bucket = "insurance-pricing-model-akshayaa"
input_path = f"s3://{bucket}/raw/freMTPL2freq.csv"  # adjust to wherever your local raw CSV lives
train_output_path = "freq_train_test_output.csv"
test_output_path = "freq_test_test_output.csv"

test_size = 0.2
random_state = 42
brand_threshold = 0.02

# ---- Load ----
df_freq = pd.read_csv(input_path)
print("Loaded shape:", df_freq.shape)

# ---- Type consistency ----
df_freq["IDpol"] = df_freq["IDpol"].astype(int)

# ---- Cleaning ----
df_freq = df_freq.drop(columns=["Area"])
df_freq["log_density"] = np.log1p(df_freq["Density"])
df_freq = df_freq.drop(columns=["Density"])

df_freq["ClaimNb"] = df_freq["ClaimNb"].clip(upper=5)
df_freq = df_freq[df_freq["Exposure"] <= 1.0]
print("After cleaning/filtering:", df_freq.shape)

# ---- VehBrand grouping ----
brand_counts = df_freq["VehBrand"].value_counts(normalize=True)
rare_brands = brand_counts[brand_counts < brand_threshold].index.tolist()
print("Brands grouped as 'Other':", rare_brands)
df_freq["VehBrand_grouped"] = df_freq["VehBrand"].apply(
    lambda x: "Other" if x in rare_brands else x
)

# ---- VehGas binary ----
df_freq["VehGas_binary"] = (df_freq["VehGas"] == "Diesel").astype(int)

# ---- Encoding ----
df_encoded = pd.get_dummies(df_freq, columns=["VehBrand_grouped", "Region"], drop_first=True)
df_encoded = df_encoded.drop(columns=["VehBrand", "VehGas"])

dummy_cols = [c for c in df_encoded.columns if c.startswith("VehBrand_grouped_") or c.startswith("Region_")]
df_encoded[dummy_cols] = df_encoded[dummy_cols].astype(int)

print("Final encoded shape:", df_encoded.shape)

# ---- Split ----
freq_train, freq_test = train_test_split(df_encoded, test_size=test_size, random_state=random_state)
print("Train shape:", freq_train.shape, "Test shape:", freq_test.shape)

# ---- Save ----
freq_train.to_csv(train_output_path, index=False)
freq_test.to_csv(test_output_path, index=False)
print("Saved local test outputs.")