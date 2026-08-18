
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Wifi.csv')

# Create a copy for processing
data = df.copy()

# Handle missing values
# For categorical columns with missing values, fill with the most frequent value
data['Dorm'].fillna(data['Dorm'].mode()[0], inplace=True)
data['Locations'].fillna(data['Locations'].mode()[0], inplace=True)

# Separate features and target
X = data.drop('TechCenter', axis=1)
y = data['TechCenter']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
y_encoded = LabelEncoder().fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train a classification model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Print evaluation metrics
print("=" * 60)
print("CLASSIFICATION MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Get the original target values for display (first 10 from test set)
y_test_original = df.iloc[X_test.index]['TechCenter'].values
y_pred_original = np.where(y_pred == 0, 'No', 'Yes')

print("\n" + "=" * 60)
print("FIRST 10 PREDICTIONS VS ACTUAL VALUES")
print("=" * 60)
print(f"{'Index':<8} {'Predicted':<15} {'Actual':<15}")
print("-" * 40)
for i in range(min(10, len(y_pred_original))):
    print(f"{i:<8} {y_pred_original[i]:<15} {y_test_original[i]:<15}")
