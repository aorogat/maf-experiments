
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PROFILE DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING AND PROFILING DATA")
print("="*80)

# Load the dataset
df = pd.read_csv('Yelp_Merged.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Verify delimiter and basic info
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values Count:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CLEANING DATA")
print("="*80)

# Target variable
target_col = 'stars'

# Check target column
print(f"\nTarget Column '{target_col}' Info:")
print(f"  Data Type: {df[target_col].dtype}")
print(f"  Missing Count: {df[target_col].isnull().sum()}")
print(f"  Unique Values: {sorted(df[target_col].unique())}")
print(f"  Value Distribution:\n{df[target_col].value_counts().sort_index()}")

# Remove rows where target is missing or invalid
initial_rows = len(df)
df = df[df[target_col].notna()].copy()
df = df[df[target_col].isin([1.0, 2.0, 3.0, 4.0, 5.0])].copy()
cleaned_rows = len(df)
print(f"\nRemoved {initial_rows - cleaned_rows} rows with invalid/missing target values")
print(f"Dataset shape after cleaning: {df.shape}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING (IMPUTATION, ENCODING, SCALING)")
print("="*80)

# Separate features and target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Target Shape: {y.shape}")

# Identify feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric Features Count: {len(numeric_features)}")
print(f"Categorical Features Count: {len(categorical_features)}")
print(f"Categorical Features: {categorical_features}")

# Step 3a: Impute Missing Values
print("\n--- Imputing Missing Values ---")

# Impute numeric features with mean
for col in numeric_features:
    missing_count = X[col].isnull().sum()
    if missing_count > 0:
        mean_value = X[col].mean()
        X[col].fillna(mean_value, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mean ({mean_value:.4f})")

# Impute categorical features with most frequent value
for col in categorical_features:
    missing_count = X[col].isnull().sum()
    if missing_count > 0:
        most_frequent = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
        X[col].fillna(most_frequent, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mode ({most_frequent})")

print(f"\nMissing values after imputation: {X.isnull().sum().sum()}")

# Step 3b: Encode Categorical Features
print("\n--- Encoding Categorical Features ---")

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: encoded with {len(le.classes_)} unique values")

# Step 3c: Scale Numerical Features
print("\n--- Scaling Numerical Features ---")

scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])
print(f"  Scaled {len(numeric_features)} numeric features using StandardScaler")

print(f"\nFinal Feature Matrix Shape: {X.shape}")
print(f"Final Feature Matrix (first 5 rows):\n{X.head()}")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT AND MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN/TEST SPLIT AND MODEL TRAINING")
print("="*80)

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain Set Size: {X_train.shape[0]} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test Set Size: {X_test.shape[0]} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTraining Set Target Distribution:\n{y_train.value_counts().sort_index()}")
print(f"\nTest Set Target Distribution:\n{y_test.value_counts().sort_index()}")

# Train RandomForestClassifier for multiclass classification
print("\n--- Training RandomForestClassifier ---")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# STEP 5: EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION")
print("="*80)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')

print("\n--- Performance Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (weighted average)")
print(f"Recall:    {recall:.4f} (weighted average)")
print(f"F1-Score:  {f1:.4f} (weighted average)")

# Full classification report
print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

# First 10 predictions vs actual
print("\n--- First 10 Predicted vs Actual Labels ---")
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_pred[:10]
})
print(comparison_df.to_string(index=True))

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETED")
print("="*80)
