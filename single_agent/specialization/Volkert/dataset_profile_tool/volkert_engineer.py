
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

# ============================================================================
# 1. LOAD AND PROFILE DATA
# ============================================================================

# Load the dataset
file_path = 'volkert.csv'
df = pd.read_csv(file_path)

print("=" * 80)
print("STEP 1: DATA PROFILING")
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values count:\n{df.isnull().sum().sum()}")
print(f"\nTarget column 'class' distribution:\n{df['class'].value_counts().sort_index()}")

# ============================================================================
# 2. CLEAN DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: DATA CLEANING")
print("=" * 80)

# Separate target and features
target_column = 'class'
y = df[target_column].copy()
X = df.drop(columns=[target_column]).copy()

print(f"Target variable shape: {y.shape}")
print(f"Features shape before cleaning: {X.shape}")

# Check for missing values in features
missing_before = X.isnull().sum().sum()
print(f"Missing values in features: {missing_before}")

# Remove constant columns (features with only one unique value)
constant_cols = X.columns[X.nunique() <= 1].tolist()
print(f"Number of constant columns (cardinality <= 1): {len(constant_cols)}")
if constant_cols:
    X = X.drop(columns=constant_cols)
    print(f"Constant columns removed: {constant_cols[:10]}... (showing first 10)")

print(f"Features shape after removing constant columns: {X.shape}")

# Verify target column has no missing values
missing_target = y.isnull().sum()
print(f"Missing values in target: {missing_target}")

if missing_target > 0:
    # Remove rows with missing target
    valid_idx = ~y.isnull()
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"Removed {missing_target} rows with missing target. New shape: {X.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING (Imputation and Scaling)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# All features are numeric, so impute missing values with mean
missing_count_before_imputation = X.isnull().sum().sum()
print(f"Missing values before imputation: {missing_count_before_imputation}")

# Impute with mean
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].mean(), inplace=True)

missing_count_after_imputation = X.isnull().sum().sum()
print(f"Missing values after imputation: {missing_count_after_imputation}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Features scaled using StandardScaler")
print(f"Scaled features shape: {X_scaled.shape}")

# ============================================================================
# 4. TRAIN-TEST SPLIT AND MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT")
print("=" * 80)

# Perform 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts().sort_index()}")
print(f"Test target distribution:\n{y_test.value_counts().sort_index()}")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MODEL TRAINING")
print("=" * 80)

# Train RandomForestClassifier for multiclass classification
print("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: MODEL EVALUATION")
print("=" * 80)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print metrics
print(f"\nPerformance Metrics (Weighted Average):")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Print full classification report
print("\n" + "=" * 80)
print("CLASSIFICATION REPORT (weighted average)")
print("=" * 80)
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================================
# 7. PREDICTION COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("FIRST 10 PREDICTIONS VS ACTUAL LABELS")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Match': (y_test.values[:10] == y_pred[:10])
})

print(comparison_df.to_string(index=False))
print(f"\nCorrectly predicted: {comparison_df['Match'].sum()} out of 10")

print("\n" + "=" * 80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 80)
