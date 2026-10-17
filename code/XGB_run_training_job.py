"""
XGB_run_training_job.py
Launches training_script_xgb.py as a SageMaker Training job (local mode).
"""

from sagemaker.xgboost.estimator import XGBoost

# ---- Config ----
role_arn = "arn:aws:iam::890742572547:role/insurance-pricing-sagemaker-role"
bucket = "insurance-pricing-model-akshayaa"

train_s3 = f"s3://{bucket}/processed/train"
model_output_s3 = f"s3://{bucket}/models"

# ---- Define the estimator ----
xgb_estimator = XGBoost(
    entry_point="XGB_train_script.py",
    source_dir=".",
    role=role_arn,
    framework_version="1.7-1",
    instance_type="local",
    instance_count=1,
    base_job_name="insurance-xgb-challenger",
    output_path=model_output_s3,
    hyperparameters={
        "max-depth": 5,
        "eta": 0.2,
        "num-round": 100,
    },
)

# ---- Run the job ----
xgb_estimator.fit({"train": train_s3})

print("Training job complete.")
print("Model artifacts saved under:", model_output_s3)