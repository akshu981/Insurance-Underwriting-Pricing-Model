# Insurance-Underwriting-Pricing-Model

A production-oriented machine learning project that predicts auto insurance claim frequency and severity to support underwriting and pricing decisions, built with production ML practices in mind from data pipeline to deployment, monitoring, and fairness auditing.

Why this project: Insurers are under increasing regulatory pressure to make pricing models explainable and free of discriminatory bias. This project goes beyond a single trained model to demonstrate the fuller lifecycle a real insurer requires: validated data, a defensible modeling approach, fairness testing, and ongoing drift monitoring.

Approach:

- Predicts claim frequency and severity using the French Motor Third-Party Liability dataset (freMTPL2freq/sev). <br>
- Compares a baseline GLM ("champion") against an XGBoost model ("challenger") using actuarial-standard validation (Gini coefficient, lift charts) rather than naive A/B testing. <br>
- Includes an explainability layer (SHAP) and fairness testing (disparate impact analysis) before any model is promoted to production. <br>
- Monitors for data drift over time, with automated retraining triggers when performance degrades. <br>

Tech stack: Python (pandas, scikit-learn, XGBoost), AWS (S3, SageMaker), GitHub Actions for CI/CD
