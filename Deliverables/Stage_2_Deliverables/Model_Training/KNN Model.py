import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier  
file_path = r"C:\Users\sthab\OneDrive\Documents\data\Resampled_Training_Data.csv"

# Load dataset
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

#Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy_knn = accuracy_score(y_test, knn.predict(X_test))




