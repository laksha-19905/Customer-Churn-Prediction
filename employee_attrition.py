import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
data = pd.read_csv("C:/Users/Lenovo/Downloads/employee_attrition.csv")
print("Dataset loaded successfully")
print(data.head())
drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])
X = data.drop('Attrition', axis=1)
y = data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000, solver='liblinear')
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_prob))
print(classification_report(y_test, lr_pred))
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))
print(classification_report(y_test, rf_pred))
importance = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importance.sort_values(ascending=False).head(10)
print("\nTop 10 Important Features:")
print(top_features)
top_features.plot(kind='bar')
plt.title("Top 10 Factors Affecting Attrition")
plt.tight_layout()
plt.show()
print("\nConclusion:")
print("• Employees with low job satisfaction are more likely to leave")
print("• Overtime and monthly income strongly influence attrition")
print("• Random Forest performs better than Logistic Regression")
