
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD AND PROFILE DATA
# ==============================================================================

print("=" * 80)
print("STEP 1: LOADING AND PROFILING DATA")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Yelp_Merged.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check missing values
print(f"\nMissing values per column:")
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_df)

# Check target variable
print(f"\nTarget variable 'stars' distribution:")
print(df['stars'].value_counts().sort_index())

# Check data types
print(f"\nData types:")
print(df.dtypes.value_counts())

# ==============================================================================
# 2. CLEAN DATA
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 2: CLEANING DATA")
print("=" * 80)

# Ensure target column is clean (remove NaNs if any)
df = df[df['stars'].notna()].copy()
print(f"After removing target NaNs: {df.shape}")

# Remove rows with all NaN features (keeping target and ID columns if present)
print(f"Checking for completely empty rows...")

# ==============================================================================
# 3. FEATURE ENGINEERING: IDENTIFY AND PROCESS FEATURES
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Separate target from features
target = df['stars'].copy()
X = df.drop('stars', axis=1)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols[:10]}...")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:10]}...")

# Remove high-cardinality ID columns that won't help prediction
high_cardinality_cols = ['business_id', 'user_id']
if 'business_id' in categorical_cols:
    X = X.drop('business_id', axis=1)
    categorical_cols.remove('business_id')
if 'user_id' in categorical_cols:
    X = X.drop('user_id', axis=1)
    categorical_cols.remove('user_id')

# Remove review_date as it's high cardinality temporal data
if 'review_date' in categorical_cols:
    X = X.drop('review_date', axis=1)
    categorical_cols.remove('review_date')

print(f"\nAfter removing high-cardinality ID and date columns:")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# ==============================================================================
# 3a. HANDLE MISSING VALUES
# ==============================================================================

print(f"\nHandling missing values...")

# Impute numerical features with mean
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        print(f"  Imputed {col} with mean: {mean_val:.4f}")

# Impute categorical features with most frequent value
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
        X[col].fillna(mode_val, inplace=True)
        print(f"  Imputed {col} with mode: {mode_val}")

print(f"\nMissing values after imputation:")
print(f"  Total missing: {X.isnull().sum().sum()}")

# ==============================================================================
# 3b. ENCODE CATEGORICAL FEATURES
# ==============================================================================

print(f"\nEncoding categorical features...")

le_dict = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    le_dict[col] = le
    print(f"  Encoded {col}")

# ==============================================================================
# 3c. SCALE NUMERICAL FEATURES
# ==============================================================================

print(f"\nScaling numerical features...")

scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

print(f"Scaled {len(numerical_cols)} numerical features")
print(f"\nFinal feature matrix shape: {X_scaled.shape}")
print(f"Features: {X_scaled.columns.tolist()}")

# ==============================================================================
# 4. TRAIN TEST SPLIT
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT (80-20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, target, test_size=0.2, random_state=42, stratify=target
)

print(f"\nTrain set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(target)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(target)*100:.1f}%)")
print(f"\nTrain set target distribution:")
print(y_train.value_counts().sort_index())
print(f"\nTest set target distribution:")
print(y_test.value_counts().sort_index())

# ==============================================================================
# 5. TRAIN MULTICLASS CLASSIFIER
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 5: TRAINING MULTICLASS CLASSIFIER")
print("=" * 80)

print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print("Model training completed!")

# ==============================================================================
# 6. MAKE PREDICTIONS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 6: MAKING PREDICTIONS")
print("=" * 80)

y_pred = model.predict(X_test)
print(f"Predictions made on {len(y_pred)} test samples")

# ==============================================================================
# 7. EVALUATE MODEL
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 7: MODEL EVALUATION")
print("=" * 80)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "-" * 80)
print("PERFORMANCE METRICS")
print("-" * 80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (weighted)")
print(f"Recall:    {recall:.4f} (weighted)")
print(f"F1-Score:  {f1:.4f} (weighted)")

# ==============================================================================
# 8. CLASSIFICATION REPORT
# ==============================================================================

print("\n" + "-" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("-" * 80)
print(classification_report(y_test, y_pred, zero_division=0))

# ==============================================================================
# 9. FIRST 10 PREDICTIONS VS ACTUAL LABELS
# ==============================================================================

print("\n" + "-" * 80)
print("FIRST 10 PREDICTIONS VS ACTUAL LABELS")
print("-" * 80)

comparison_df = pd.DataFrame({
    'Predicted': y_pred[:10],
    'Actual': y_test.values[:10],
    'Match': (y_pred[:10] == y_test.values[:10])
})

print(comparison_df.to_string(index=True))

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
