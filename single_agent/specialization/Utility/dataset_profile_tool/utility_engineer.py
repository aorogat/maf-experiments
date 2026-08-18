
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
df = pd.read_csv('Utility.csv')
print("="*80)
print("STEP 1: DATA LOADING")
print("="*80)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 2: DATA PROFILING & INSPECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA PROFILING")
print("="*80)
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget column (CSRI) statistics:\n{df['CSRI'].describe()}")

# ============================================================================
# STEP 3: IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE TYPE IDENTIFICATION")
print("="*80)

# Separate features and target
X = df.drop(columns=['CSRI'])
y = df['CSRI']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {categorical_cols}")
print(f"Number of categorical features: {len(categorical_cols)}")
print(f"\nNumerical features: {numerical_cols}")
print(f"Number of numerical features: {len(numerical_cols)}")

# ============================================================================
# STEP 4: DATA CLEANING & IMPUTATION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: DATA CLEANING & IMPUTATION")
print("="*80)

# Create a copy for processing
X_clean = X.copy()

# Impute numerical features with mean
print("\nImputing numerical features with mean...")
for col in numerical_cols:
    missing_count = X_clean[col].isnull().sum()
    if missing_count > 0:
        mean_value = X_clean[col].mean()
        X_clean[col].fillna(mean_value, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mean {mean_value:.2f}")
    else:
        print(f"  {col}: no missing values")

# Impute categorical features with mode (most frequent value)
print("\nImputing categorical features with mode...")
for col in categorical_cols:
    missing_count = X_clean[col].isnull().sum()
    if missing_count > 0:
        mode_value = X_clean[col].mode()[0]
        X_clean[col].fillna(mode_value, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mode '{mode_value}'")
    else:
        print(f"  {col}: no missing values")

print(f"\nAfter imputation - Missing values:\n{X_clean.isnull().sum().sum()}")

# ============================================================================
# STEP 5: FEATURE ENGINEERING - ENCODING & SCALING
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE ENGINEERING (ENCODING & SCALING)")
print("="*80)

# Encode categorical variables using LabelEncoder
print("\nEncoding categorical features...")
le_dict = {}
X_encoded = X_clean.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_clean[col].astype(str))
    le_dict[col] = le
    print(f"  {col}: encoded into numeric values (0-{len(le.classes_)-1})")

# Scale numerical features using StandardScaler
print("\nScaling numerical features...")
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
print(f"  Scaled {len(numerical_cols)} numerical features using StandardScaler")

print(f"\nFinal feature matrix shape: {X_encoded.shape}")
print(f"Final feature matrix:\n{X_encoded.head()}")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT (80/20)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TRAIN-TEST SPLIT (80/20)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_encoded)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_encoded)*100:.1f}%)")
print(f"Training target mean: {y_train.mean():.2f}")
print(f"Test target mean: {y_test.mean():.2f}")

# ============================================================================
# STEP 7: MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("STEP 7: MODEL TRAINING")
print("="*80)

# Train RandomForestRegressor
print("\nTraining RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# STEP 8: MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 8: MODEL EVALUATION")
print("="*80)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate MAE
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nMean Absolute Error (MAE):")
print(f"  Training MAE: {train_mae:.4f}")
print(f"  Test MAE: {test_mae:.4f}")

# Feature importance
print(f"\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 9: DISPLAY FIRST 10 PREDICTIONS VS ACTUAL VALUES
# ============================================================================
print("\n" + "="*80)
print("STEP 9: FIRST 10 TEST SET PREDICTIONS VS ACTUAL VALUES")
print("="*80)

results_df = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_test_pred[:10],
    'Absolute_Error': np.abs(y_test.iloc[:10].values - y_test_pred[:10])
})

print(f"\n{results_df.to_string(index=True)}")

print(f"\nMean Absolute Error for first 10 samples: {results_df['Absolute_Error'].mean():.4f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Dataset: Utility.csv (4574 rows × 13 columns)")
print(f"Target: CSRI (numeric regression)")
print(f"Features processed: {len(numerical_cols)} numerical + {len(categorical_cols)} categorical")
print(f"Model: RandomForestRegressor")
print(f"Train-Test Split: 80-20 ({X_train.shape[0]}-{X_test.shape[0]})")
print(f"Test MAE: {test_mae:.4f}")
print(f"Pipeline: Load → Profile → Clean → Encode → Scale → Train → Evaluate ✓")
print("="*80)
