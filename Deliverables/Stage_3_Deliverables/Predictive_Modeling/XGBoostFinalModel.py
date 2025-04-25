import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------------------------------
# 1. Load and Preprocess Data
file_path = r"C:\Users\sthab\OneDrive\Documents\data\Resampled_Training_Data.csv"

df = pd.read_csv(file_path)
print(df.info())
print(df.head())

X = df.drop(columns=['Churn_1'])
y = df['Churn_1']

scaler = StandardScaler()
X[['tenure', 'MonthlyCharges']] = scaler.fit_transform(X[['tenure', 'MonthlyCharges']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ----------------------------------------
# XGBoost with initial parameters
xgb_model = XGBClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8, gamma=0.2,
    reg_lambda=1.0, reg_alpha=0.5, eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# ------------------------------------------
# 3. 5-Fold Cross-Validation with XGBoost
# ------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')
print("5-Fold Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy: {:.4f}".format(cv_scores.mean()))
print("CV Accuracy Standard Deviation: {:.4f}".format(cv_scores.std()))

# ------------------------------------------
# 4. Hyperparameter Tuning for XGBoost
# ------------------------------------------
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [0.5, 1.0, 2.0],
    'reg_alpha': [0, 0.5, 1.0]
}

xgb_for_tuning = XGBClassifier(eval_metric="logloss", random_state=42)
grid_search = GridSearchCV(estimator=xgb_for_tuning, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
# Train final XGBoost model with best parameters
best_params = grid_search.best_params_
xgb_optimized = XGBClassifier(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    reg_lambda=best_params['reg_lambda'],
    reg_alpha=best_params['reg_alpha'],
    eval_metric="logloss",
    random_state=42
)
xgb_optimized.fit(X_train, y_train)
y_pred_optimized = xgb_optimized.predict(X_test)
print("Optimized XGBoost Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("Optimized XGBoost Classification Report:\n", classification_report(y_test, y_pred_optimized))

# ------------------------------------------
# 5. Performance Metrics and Visualization
# ------------------------------------------
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Optimized XGBoost")
plt.show()

# Feature Importance
importances = xgb_optimized.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
plt.title("Feature Importance in XGBoost Model")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
