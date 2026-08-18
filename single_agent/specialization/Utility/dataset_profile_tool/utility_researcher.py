
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PROFILE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING AND PROFILING DATA")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Utility.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn Data Types:")
print(df.dtypes)

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CLEANING DATA")
print("=" * 80)

# Separate features and target
X = df.drop('CSRI', axis=1)
y = df['CSRI']

print(f"\nTarget Variable (CSRI) Statistics:")
print(f"  - Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.4f}")
print(f"  - Missing values: {y.isnull().sum()}")
print(f"  - Data type: {y.dtype}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Check for missing values in features
print(f"\nMissing values in features:")
print(X.isnull().sum())

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Create a copy for processing
X_processed = X.copy()

# Impute missing numerical values with mean
print("\nImputing missing numerical values with mean...")
for col in numerical_cols:
    if X_processed[col].isnull().sum() > 0:
        mean_val = X_processed[col].mean()
        X_processed[col].fillna(mean_val, inplace=True)
        print(f"  - {col}: filled {X_processed[col].isnull().sum()} missing values with mean={mean_val:.4f}")

# Impute missing categorical values with most frequent value
print("\nImputing missing categorical values with mode...")
for col in categorical_cols:
    if X_processed[col].isnull().sum() > 0:
        mode_val = X_processed[col].mode()[0]
        X_processed[col].fillna(mode_val, inplace=True)
        print(f"  - {col}: filled {X_processed[col].isnull().sum()} missing values with mode={mode_val}")

# Encode categorical features using LabelEncoder
print("\nEncoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le
    print(f"  - {col}: encoded with {len(le.classes_)} unique values")

print(f"\nProcessed features shape: {X_processed.shape}")
print(f"\nProcessed features data types:")
print(X_processed.dtypes)

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Scale numerical features
print("\nScaling numerical features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  - Scaling completed for {len(numerical_cols)} numerical features")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING REGRESSION MODEL")
print("=" * 80)

# Train RandomForestRegressor
print("\nTraining RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train_scaled, y_train)
print("  - Model training completed!")

# ============================================================================
# 6. EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MODEL EVALUATION")
print("=" * 80)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate MAE
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nMean Absolute Error (MAE):")
print(f"  - Training Set MAE: {train_mae:.6f}")
print(f"  - Test Set MAE: {test_mae:.6f}")

# Feature importance
print(f"\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 7. PREDICTIONS VS ACTUAL (FIRST 10)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: FIRST 10 PREDICTIONS VS ACTUAL VALUES (TEST SET)")
print("=" * 80)

# Create results dataframe
results_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_test_pred[:10],
    'Absolute_Error': np.abs(y_test.values[:10] - y_test_pred[:10])
})

print("\n")
print(results_df.to_string(index=True))

print(f"\nMean Absolute Error (First 10): {results_df['Absolute_Error'].mean():.6f}")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print(f"""
Dataset: Utility.csv
  - Total samples: {df.shape[0]}
  - Total features: {df.shape[1] - 1}
  - Target variable: CSRI

Data Processing:
  - Categorical features: {len(categorical_cols)} (encoded)
  - Numerical features: {len(numerical_cols)} (scaled)
  - Missing values handled: Yes (imputation)

Model:
  - Algorithm: RandomForestRegressor
  - Train set size: {X_train.shape[0]} (80%)
  - Test set size: {X_test.shape[0]} (20%)
  
Performance Metrics:
  - Training MAE: {train_mae:.6f}
  - Test MAE: {test_mae:.6f}
  
Status: ✓ Pipeline executed successfully!
""")
