# DIGITAL-CREDIT-SCORING
This repository contains a machine learning-powered platform for rural credit risk assessment and explainable credit scoring. The project is designed to improve financial inclusion for smallholder farmers and rural borrowers who often lack formal credit profiles.

Rural Credit Scoring Dashboard – Trustworthy AI for Financial Inclusion
📢 Project Overview
This repository presents an end-to-end, machine learning-powered dashboard for rural credit risk assessment. The solution applies Trustworthy AI principles—fairness, transparency, explainability, robustness, and privacy—to predict repayment probability, assign a credit score, and explain lending risk for rural borrowers, especially farmers. Our goal is to help financial institutions make ethical, data-driven credit decisions that support inclusion for communities underserved by traditional methods.

📦 Repository Structure
App.py, app1.py — Main Streamlit dashboard/application files. Run either to launch the user interface.

rural_credit_dataset_mixed.csv — Core dataset simulating real borrower profiles (demographics, agriculture, finance, digital behaviour).

xgb_model.pkl — Pre-trained XGBoost model for loan repayment prediction.

scaler.pkl, column_transformer.pkl — Data transformation objects (scaling and encoding features for consistency).

graphs_paper1.py, newplot.jpg — Visualisation code and sample output for use in reports or presentations.

Untitled.ipynb — Jupyter notebook for full exploratory data analysis, feature engineering, and model training.

image.jpeg, image.jpg — Dashboard screenshots and demo visuals for documentation/README use.

🚀 How to Deploy and Use the Dashboard
Clone the Repository

Install Required Python Packages

You’ll need: streamlit, pandas, scikit-learn, xgboost, matplotlib

bash
pip install streamlit pandas scikit-learn xgboost matplotlib
Optionally, add a requirements.txt file for quick environment setup.

Run the Dashboard

bash
streamlit run App.py
Visit the local URL provided (usually http://localhost:8501).

Enter sample borrower details in the sidebar to receive a credit score, repayment probability, and risk classification instantly.

Analyze which factors influence the result via the feature importance graph.

Explore and Extend

Use the provided notebook (Untitled.ipynb) for custom data exploration, retraining, or model tweaking.

Replace or retrain models (xgb_model.pkl) to update prediction logic as needed.

Study included scripts and images for adapting the dashboard for your own presentation or paper.

💡 Key Features
Trustworthy AI: Demonstrates clear, interpretable risk assessment based on ethical AI standards.

Feature Importance: Users see why the model assigns a score—enabling transparency and trust.

Rural Focus: Accounts for agricultural context, digital behaviour, and economic diversity in lending risk.

Ready to Use: Pre-trained models and full dataset included—no wait time to test or demo.

👩‍💻 Example Use Case
Input a Maharashtra farmer’s details, land size, income, crop, loan requested, and see output: repayment probability, credit score, and risk category.

The dashboard’s graph will highlight the most influential variables—e.g., land size and annual income.

Use for research demos, bank testing, or comparative studies in financial AI.

📈 Extending the Project
Add new models to App.py or retrain using the Jupyter notebook.

Expand the dataset with real-world data for robust credit scoring.

Integrate live Monte Carlo scenarios or SHAP explainability modules.

<img width="568" height="801" alt="Screenshot 2025-11-12 234117" src="https://github.com/user-attachments/assets/257ea536-0eea-4692-b3d1-7dce02604e19" />


<img width="598" height="797" alt="Screenshot 2025-11-12 234309" src="https://github.com/user-attachments/assets/633b3234-ca51-49dc-b4c6-8aedf6d59f16" />


<img width="563" height="154" alt="Screenshot 2025-11-12 234422" src="https://github.com/user-attachments/assets/654f33ff-0dcc-4f7f-9bcb-feb775c0baef" />


<img width="544" height="403" alt="Screenshot 2025-11-12 234509" src="https://github.com/user-attachments/assets/2cc914a8-5a89-41b0-8b91-b35bab12a193" />
