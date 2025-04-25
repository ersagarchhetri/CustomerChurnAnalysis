import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\sthab\OneDrive\Documents\data\Resampled_Training_Data.csv"
data = pd.read_csv(file_path)

# View first 5 rows
print(data.head())

# Check shape and column names
print("\nDataset shape:", data.shape)
print("\nColumn names:", data.columns.tolist())

# Prepare features and target
X = data.drop("Churn_1", axis=1)
y = data["Churn_1"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
accuracy_lr = accuracy_score(y_test, lr.predict(X_test))

#Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy_knn = accuracy_score(y_test, knn.predict(X_test))

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
accuracy_rf = accuracy_score(y_test, rf.predict(X_test))

# Train XGBoost
xgb = XGBClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8,
    colsample_bytree=0.8, gamma=0.2, reg_lambda=1.0, reg_alpha=0.5, eval_metric='logloss'
)
xgb.fit(X_train, y_train)
accuracy_xgb = accuracy_score(y_test, xgb.predict(X_test))



# Store model accuracies
models = {
    "Logistic Regression": accuracy_lr,
    "Random Forest": accuracy_rf,
    "XGBoost": accuracy_xgb,
    "K-Nearest Neighbors (KNN)": accuracy_knn,
}

# Print model accuracies
print("\n🔹 Model Accuracies:")
for model, acc in models.items():
    print(f"{model}: {acc:.2f}")

# Find the best model
best_model = max(models, key=models.get)
print("\n🏆 Best Model:", best_model, "with accuracy of", models[best_model])

# Feature importance plot (for XGBoost only)
feature_importance = xgb.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in XGBoost Model")
plt.show()
