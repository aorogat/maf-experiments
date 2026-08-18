
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Load dataset with comma delimiter
df = pd.read_csv('Yelp_Merged.csv', delimiter=',')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA CLEANING")
print("=" * 80)

# Check target column
print(f"\nTarget column (stars) info:")
print(f"  Data type: {df['stars'].dtype}")
print(f"  Missing values: {df['stars'].isna().sum()}")
print(f"  Value counts:\n{df['stars'].value_counts().sort_index()}")

# Remove rows where target is missing (if any)
df = df[df['stars'].notna()].copy()
print(f"\nDataset shape after removing missing targets: {df.shape}")

# Convert target to integer class labels
df['stars'] = df['stars'].astype(int)

# Identify columns with missing values
print(f"\nMissing values per column:")
missing_cols = df.isnull().sum()
missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
print(missing_cols)

# Separate target from features
y = df['stars'].copy()
X = df.drop(columns=['stars']).copy()

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Identify categorical and numerical columns
print("\nIdentifying feature types...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}...")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:10]}...")

# Handle missing values in numerical columns - impute with mean
print("\nImputing missing values in numerical columns with mean...")
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        print(f"  {col}: filled {X[col].isnull().sum()} with mean={mean_val:.2f}")

# Handle missing values in categorical columns - impute with most frequent
print("\nImputing missing values in categorical columns with most frequent value...")
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        most_freq = X[col].mode()[0] if len(X[col].mode()) > 0 else 'UNKNOWN'
        X[col].fillna(most_freq, inplace=True)
        print(f"  {col}: filled with mode={most_freq}")

print(f"\nMissing values after imputation:")
print(f"  Total missing: {X.isnull().sum().sum()}")

# Encode categorical features
print("\nEncoding categorical features...")
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: encoded {len(le.classes_)} unique values")

# Scale numerical features
print("\nScaling numerical features...")
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
print(f"  Scaled {len(numerical_cols)} numerical features")

print(f"\nFinal feature matrix shape: {X_encoded.shape}")
print(f"Feature matrix data types: {X_encoded.dtypes.value_counts().to_dict()}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"\nTraining set class distribution:")
print(y_train.value_counts().sort_index())
print(f"\nTest set class distribution:")
print(y_test.value_counts().sort_index())

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING MULTICLASS CLASSIFIER")
print("=" * 80)

print("\nTraining RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

clf.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MODEL EVALUATION")
print("=" * 80)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "-" * 80)
print("OVERALL PERFORMANCE METRICS")
print("-" * 80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n" + "-" * 80)
print("CLASSIFICATION REPORT (Full)")
print("-" * 80)
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================================
# 7. PREDICTIONS VS ACTUAL (First 10 samples)
# ============================================================================
print("\n" + "-" * 80)
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS")
print("-" * 80)

comparison_df = pd.DataFrame({
    'Predicted': y_pred[:10],
    'Actual': y_test.iloc[:10].values,
    'Match': (y_pred[:10] == y_test.iloc[:10].values)
})

print(comparison_df.to_string(index=True))
print(f"\nAccuracy on first 10 samples: {comparison_df['Match'].sum()}/10")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: FEATURE IMPORTANCE (Top 15)")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

print("\n" + "=" * 80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
