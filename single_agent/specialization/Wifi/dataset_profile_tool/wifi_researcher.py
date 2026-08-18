
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

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Handle missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Fill missing values in categorical columns with mode
df['Dorm'].fillna(df['Dorm'].mode()[0], inplace=True)
df['Locations'].fillna(df['Locations'].mode()[0], inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('TechCenter', axis=1)
y = df['TechCenter']

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# Create a copy of X for preprocessing
X_processed = X.copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

print("\nProcessed features shape:", X_processed.shape)
print("Processed features columns:", X_processed.columns.tolist())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classification model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*50)

# Print classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Print first 10 predictions and actual values
print("\n" + "="*50)
print("FIRST 10 PREDICTIONS vs ACTUAL VALUES")
print("="*50)
results_df = pd.DataFrame({
    'Predicted': le_target.inverse_transform(y_pred[:10]),
    'Actual': le_target.inverse_transform(y_test[:10])
})
print(results_df.to_string(index=True))
print("="*50)
