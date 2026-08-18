
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
print("\nDataset info:")
print(df.info())

# Handle missing values by filling with mode for categorical columns
df['Dorm'].fillna(df['Dorm'].mode()[0], inplace=True)
df['Locations'].fillna(df['Locations'].mode()[0], inplace=True)

# Separate target and features
X = df.drop('TechCenter', axis=1)
y = df['TechCenter']

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create a copy for preprocessing
X_processed = X.copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

# Split data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train classification model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("\n" + "="*60)
print("MODEL EVALUATION METRICS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Print classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Print first 10 predictions and actual values
print("\n" + "="*60)
print("FIRST 10 PREDICTIONS VS ACTUAL VALUES")
print("="*60)
print(f"{'Index':<8} {'Predicted':<15} {'Actual':<15}")
print("-" * 38)

for i in range(min(10, len(y_test))):
    predicted = le_target.inverse_transform([y_pred[i]])[0]
    actual = le_target.inverse_transform([y_test[i]])[0]
    print(f"{i:<8} {predicted:<15} {actual:<15}")

print("\n" + "="*60)
print("TRAINING AND TEST SET SIZES")
print("="*60)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Total samples: {len(df)}")
