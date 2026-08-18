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

# Display dataset info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Separate features and target
X = df.drop('TechCenter', axis=1)
y = df['TechCenter']

# Handle missing values in categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Fill missing values in categorical columns with 'Unknown'
for col in categorical_cols:
    if X[col].isna().sum() > 0:
        X[col].fillna('Unknown', inplace=True)

# Fill missing values in numerical columns with median
for col in numerical_cols:
    if X[col].isna().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Split data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print("\n" + "="*60)
print("CLASSIFICATION MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Get first 10 predictions
print("\n" + "="*60)
print("FIRST 10 PREDICTIONS (Predicted vs Actual)")
print("="*60)
print(f"{'Index':<8} {'Predicted':<15} {'Actual':<15}")
print("-"*40)
for i in range(min(10, len(y_test))):
    predicted = le_target.inverse_transform([y_pred[i]])[0]
    actual = le_target.inverse_transform([y_test[i]])[0]
    print(f"{i:<8} {predicted:<15} {actual:<15}")

print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"Total samples: {len(df)}")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Model used: Random Forest Classifier")
print(f"Number of trees: 100")