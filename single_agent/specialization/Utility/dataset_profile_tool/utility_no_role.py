
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load and Profile Data
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING AND PROFILING DATA")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Utility.csv')

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nDataset info:")
print(df.info())

# ============================================================================
# STEP 2: Data Cleaning
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA CLEANING")
print("=" * 80)

# Check for any NaN or invalid values in target column
target_col = 'CSRI'
print(f"\nTarget column '{target_col}' stats:")
print(f"  Missing values: {df[target_col].isnull().sum()}")
print(f"  Min value: {df[target_col].min()}")
print(f"  Max value: {df[target_col].max()}")
print(f"  Data type: {df[target_col].dtype}")

# Create a copy for processing
df_clean = df.copy()

# Identify feature and target columns
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# STEP 3: Feature Engineering - Identify Feature Types
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING - IDENTIFY FEATURE TYPES")
print("=" * 80)

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# ============================================================================
# STEP 4: Handle Missing Values and Imputation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HANDLE MISSING VALUES AND IMPUTATION")
print("=" * 80)

# Impute numeric features with mean
print("\nImputing numeric features with mean...")
for col in numeric_features:
    missing_count = X[col].isnull().sum()
    if missing_count > 0:
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        print(f"  {col}: filled {missing_count} missing values with mean ({mean_val:.2f})")
    else:
        print(f"  {col}: no missing values")

# Impute categorical features with most frequent value (mode)
print("\nImputing categorical features with mode...")
for col in categorical_features:
    missing_count = X[col].isnull().sum()
    if missing_count > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        print(f"  {col}: filled {missing_count} missing values with mode ({mode_val})")
    else:
        print(f"  {col}: no missing values")

print("\nAfter imputation - Missing values:")
print(X.isnull().sum())

# ============================================================================
# STEP 5: Encode Categorical Features
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: ENCODE CATEGORICAL FEATURES")
print("=" * 80)

label_encoders = {}
X_encoded = X.copy()

print("\nEncoding categorical features with LabelEncoder...")
for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  {col}: encoded {len(le.classes_)} unique values")

print(f"\nDataset shape after encoding: {X_encoded.shape}")

# ============================================================================
# STEP 6: Scale Numerical Features
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: SCALE NUMERICAL FEATURES")
print("=" * 80)

scaler = StandardScaler()
X_encoded[numeric_features] = scaler.fit_transform(X_encoded[numeric_features])

print(f"\nNumerical features scaled using StandardScaler")
print(f"Scaled numeric features statistics:")
for col in numeric_features:
    print(f"  {col}: mean={X_encoded[col].mean():.4f}, std={X_encoded[col].std():.4f}")

# ============================================================================
# STEP 7: Train-Test Split (80/20)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Train/Test ratio: {X_train.shape[0] / X_test.shape[0]:.2f}")

# ============================================================================
# STEP 8: Train Regression Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: TRAIN REGRESSION MODEL (RandomForestRegressor)")
print("=" * 80)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    min_samples_split=5,
    min_samples_leaf=2
)

print("\nTraining RandomForestRegressor...")
model.fit(X_train, y_train)
print("Model training completed!")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 most important features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 9: Make Predictions
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: MAKE PREDICTIONS")
print("=" * 80)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"Training predictions completed: {y_pred_train.shape[0]} samples")
print(f"Testing predictions completed: {y_pred_test.shape[0]} samples")

# ============================================================================
# STEP 10: Evaluate Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: EVALUATE MODEL - MEAN ABSOLUTE ERROR (MAE)")
print("=" * 80)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nTraining MAE: {mae_train:.4f}")
print(f"Testing MAE: {mae_test:.4f}")
print(f"Difference: {abs(mae_test - mae_train):.4f}")

# ============================================================================
# STEP 11: Display First 10 Predictions vs Actual Values
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: FIRST 10 PREDICTIONS VS ACTUAL VALUES (Test Set)")
print("=" * 80)

results_df = pd.DataFrame({
    'Predicted': y_pred_test[:10],
    'Actual': y_test.iloc[:10].values,
    'Error': np.abs(y_pred_test[:10] - y_test.iloc[:10].values)
})

print("\n")
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total samples processed: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Numeric features used: {len(numeric_features)}")
print(f"Categorical features used: {len(categorical_features)}")
print(f"Total features after encoding: {X_encoded.shape[1]}")
print(f"\nFinal Test MAE: {mae_test:.4f}")
print(f"Model: RandomForestRegressor")
print(f"Target variable: {target_col}")
print("=" * 80)
