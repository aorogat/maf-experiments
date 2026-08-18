
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PROFILE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING AND PROFILING DATA")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Utility.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CLEANING DATA")
print("=" * 80)

# Identify target and features
target_column = 'CSRI'
y = df[target_column].copy()
X = df.drop(columns=[target_column]).copy()

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nTarget column: {target_column}")
print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Check for missing values in target
print(f"\nMissing values in target (CSRI): {y.isnull().sum()}")
print(f"Missing values in features:")
print(X.isnull().sum())

# Remove any rows with missing target values (if any)
valid_indices = ~y.isnull()
X = X[valid_indices].copy()
y = y[valid_indices].copy()

print(f"\nData after removing invalid target values: {X.shape}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING (Imputation, Encoding, Scaling)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Create a copy for processing
X_processed = X.copy()

# 3a. Handle numerical features - impute with mean
print("\n3a. Imputing numerical features with mean:")
for col in numerical_features:
    missing_count = X_processed[col].isnull().sum()
    if missing_count > 0:
        mean_value = X_processed[col].mean()
        X_processed[col].fillna(mean_value, inplace=True)
        print(f"  {col}: filled {missing_count} missing values with mean {mean_value:.2f}")
    else:
        print(f"  {col}: no missing values")

# 3b. Handle categorical features - impute with most frequent value
print("\n3b. Imputing categorical features with most frequent value:")
for col in categorical_features:
    missing_count = X_processed[col].isnull().sum()
    if missing_count > 0:
        mode_value = X_processed[col].mode()[0]
        X_processed[col].fillna(mode_value, inplace=True)
        print(f"  {col}: filled {missing_count} missing values with mode '{mode_value}'")
    else:
        print(f"  {col}: no missing values")

# 3c. Encode categorical features using LabelEncoder
print("\n3c. Encoding categorical features:")
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: encoded {len(le.classes_)} unique values")

print(f"\nFeature matrix shape after encoding: {X_processed.shape}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT AND SCALING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT AND SCALING")
print("=" * 80)

# Perform 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0]/len(X_processed)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X_processed)*100:.1f}%)")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Scale only the numerical features
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"\nScaling applied to {len(numerical_features)} numerical features")

# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING REGRESSION MODEL")
print("=" * 80)

# Train RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining RandomForestRegressor...")
model.fit(X_train_scaled, y_train)
print("Model training completed!")

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MODEL EVALUATION")
print("=" * 80)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate Mean Absolute Error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"\nMean Absolute Error (MAE):")
print(f"  Training MAE: {mae_train:.4f}")
print(f"  Test MAE: {mae_test:.4f}")

# ============================================================================
# STEP 7: DISPLAY PREDICTIONS vs ACTUAL VALUES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: FIRST 10 PREDICTIONS vs ACTUAL VALUES (TEST SET)")
print("=" * 80)

# Create a DataFrame with predictions and actual values
results_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_test_pred[:10],
    'Absolute Error': np.abs(y_test.values[:10] - y_test_pred[:10])
})

print("\n" + results_df.to_string(index=True))

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: TOP 10 FEATURE IMPORTANCES")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + feature_importance.head(10).to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print(f"""
Dataset: Utility.csv
Total Samples: {len(df)}
Training Samples: {len(X_train)}
Test Samples: {len(X_test)}

Features:
  - Numerical: {len(numerical_features)}
  - Categorical: {len(categorical_features)}
  - Total: {len(X_processed.columns)}

Target Column: CSRI
Target Range: [{y.min():.2f}, {y.max():.2f}]
Target Mean: {y.mean():.4f}
Target Std: {y.std():.4f}

Model: RandomForestRegressor
  - Estimators: 100
  - Max Depth: 15
  - Min Samples Split: 5
  - Min Samples Leaf: 2

Performance:
  - Training MAE: {mae_train:.4f}
  - Test MAE: {mae_test:.4f}

Preprocessing:
  - Missing value imputation: Applied (mean for numerical, mode for categorical)
  - Categorical encoding: LabelEncoder
  - Feature scaling: StandardScaler (numerical features only)
  - Train-Test Split: 80/20 (random_state=42)
""")
print("=" * 80)
