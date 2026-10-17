"""
run_training_job.py
Launches training_script.py as a SageMaker Training job (local mode).
"""

from sagemaker.sklearn.estimator import SKLearn

# ---- Config ----
role_arn = "arn:aws:iam::890742572547:role/insurance-pricing-sagemaker-role"
bucket = "insurance-pricing-model-akshayaa"

train_s3 = f"s3://{bucket}/processed/train"
model_output_s3 = f"s3://{bucket}/models"

# ---- Define the estimator ----
sklearn_estimator = SKLearn(
    entry_point="training_script.py",
    source_dir=".",  # looks for requirements.txt alongside training_script.py here
    role=role_arn,
    framework_version="1.2-1",
    instance_type="local",   # Docker local mode, same as the processing job
    instance_count=1,
    base_job_name="insurance-glm-champion",
    output_path=model_output_s3,
)

# ---- Run the job ----
sklearn_estimator.fit({"train": train_s3})

print("Training job complete.")
print("Model artifacts saved under:", model_output_s3)