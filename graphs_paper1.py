import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('rural_credit_dataset_mixed.csv')
df = df.fillna(0)

# Drop identifier column if present
if 'borrowerid' in df.columns:
    df = df.drop(columns=['borrowerid'])

# Define features and target (replace with the actual target column name if needed)
X = df.drop('loanrepaymentstatus', axis=1)
y = df['loanrepaymentstatus']
catcols = ['gender', 'educationlevel', 'villagestate', 'occupationtype', 'irrigationtype', 'croptype']
numcols = [col for col in X.columns if col not in catcols]

# Convert categorical columns to string
for col in catcols:
    X[col] = X[col].astype(str)

# One-hot encode categorical variables
ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(X[catcols])
X_num = X[numcols].values
X_all = pd.DataFrame(
    np.hstack([X_ohe, X_num]),
    columns=list(ohe.get_feature_names_out(catcols)) + numcols
)

# Scale numerical variables
scaler = StandardScaler()
X_all[numcols] = scaler.fit_transform(X_all[numcols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Fit logistic regression
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)
probs = logmodel.predict_proba(X_test)[:,1]

# ROC curve
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve with large font
plt.figure(figsize=(9,9))
plt.plot(fpr, tpr, color='blue', linewidth=4, label=f'Logistic Regression\nAUC = {roc_auc:.2f}', zorder=10)
plt.plot([0,1], [0,1], 'k--', linewidth=3, alpha=0.7, zorder=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate\n(Positive label: 1)', fontsize=28, weight='bold')
plt.ylabel('True Positive Rate\n(Positive label: 1)', fontsize=28, weight='bold')
plt.title('ROC Curve - Logistic Regression (Loan Repayment Prediction)', fontsize=32, weight='bold')
plt.legend(loc='lower right', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig('roc_logistic_visible.png', dpi=200)
plt.show()
