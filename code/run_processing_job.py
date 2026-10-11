"""
run_processing_job.py
Launches processing_script.py as a real SageMaker Processing job.
"""

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# ---- Config ----
role_arn = "arn:aws:iam::890742572547:role/insurance-pricing-sagemaker-role"
bucket = "insurance-pricing-model-akshayaa"
region = "us-east-1"

input_s3 = f"s3://{bucket}/raw/freMTPL2freq.csv"
output_train_s3 = f"s3://{bucket}/processed/train"
output_test_s3 = f"s3://{bucket}/processed/test"

# ---- Define the processor ----
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role_arn,
    instance_type="local",
    instance_count=1,
    base_job_name="insurance-freq-cleaning",
)

# ---- Run the job ----
sklearn_processor.run(
    code="processing_script.py",
    inputs=[
        ProcessingInput(
            source=input_s3,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/train",
            destination=output_train_s3,
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/test",
            destination=output_test_s3,
        ),
    ],
)

print("Processing job complete.")
print("Train output:", output_train_s3)
print("Test output:", output_test_s3)