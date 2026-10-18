# Insurance-Underwriting-Pricing-Model

A production-oriented machine learning project that predicts auto insurance claim frequency and severity to support underwriting and pricing decisions, built with production ML practices in mind from data pipeline to deployment, monitoring, and fairness auditing.

Why this project: Insurers are under increasing regulatory pressure to make pricing models explainable and free of discriminatory bias. This project goes beyond a single trained model to demonstrate the fuller lifecycle a real insurer requires: validated data, a defensible modeling approach, fairness testing, and ongoing drift monitoring.

Approach:

- Predicts claim frequency and severity using the French Motor Third-Party Liability dataset (freMTPL2freq/sev). <br>
- Compares a baseline GLM (champion) against an XGBoost model (challenger) using actuarial-standard validation (Gini coefficient, lift charts) rather than naive A/B testing. <br>
- Includes an explainability layer (SHAP) and fairness testing (disparate impact analysis) before any model is promoted to production. <br>
- Monitors for data drift over time, with automated retraining triggers when performance degrades. <br>

Tech stack: Python (pandas, scikit-learn, XGBoost), AWS (S3, SageMaker), GitHub Actions for CI/CD


# MLOps Architecture & Pipeline

<img width="500" height="676" alt="Image" src="https://github.com/user-attachments/assets/652a337b-6e2c-44e5-9c24-b854b3ca1c21" /> 

## Architecture Overview

This project utilizes a **Local-First MLOps architecture** with Cloud Backup and Artifact Storage. The primary goal of this design is **cost-efficiency**, specifically to avoid incurring high costs from provisioning cloud-based AWS SageMaker compute instances during the development and experimentation phases.

Because we are running the heavy compute locally, the flow is: **JupyterLab -> Local Docker -> AWS Cloud (S3)**. 

If we were using a fully cloud-native setup, the flow would be: *JupyterLab -> AWS Cloud (SageMaker) -> AWS Cloud (S3).*

---

## Component Breakdown
**1. JupyterLab & Desktop Docker (Local Control & Execution)**
JupyterLab (Control Tower): Acts as the primary environment where code is written, modified, and tracked via the Local Git Repo (which pushes backups to GitHub).

Command Prompt (The Bridge): We use terminal commands (e.g., run_training_job.py) in the JupyterLab environment to tell Docker to start executing. Docker Desktop (Engine Room) runs isolated containers. It ensures that our application runs identically regardless of the machine. In this project, Docker executes the code inside an isolated environment and simulates AWS SageMaker jobs locally. Since this local processing relies on the computer's CPU, it is highly effective for sequential tasks like our GLM and XGBoost models.


**2. Docker and AWS (Cloud Interaction)**
IAM Role: Docker uses AWS credentials via an IAM Role to securely authenticate and communicate with the AWS Cloud.

Amazon S3 Bucket: Acts as our centralized cloud storage. The local Docker container:
- READS the raw data from S3 to train the model.
- Once training is complete, it WRITES the finished model artifacts (e.g., insurance-glm-champion... and insurance-xgb...) back into the /models folder in S3.

## Summary of the Data Flow
Develop: Write code in JupyterLab and commit changes to Git.

Execute: Use the Command Prompt to trigger the training script.

Train: Docker Desktop runs the code locally in an isolated container, saving compute costs.

Store: Docker authenticates via IAM Role and pushes the completed model artifacts to AWS S3.
