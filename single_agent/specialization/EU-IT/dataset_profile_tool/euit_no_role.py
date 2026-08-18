
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)
df = pd.read_csv('EU-IT_cleaned.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(3))

# ============================================================================
# 2. PROFILE AND CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("DATA PROFILING AND CLEANING")
print("=" * 80)

# Display missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop rows with missing target variable (Position)
print(f"\nDropping {df['Position'].isnull().sum()} rows with missing Position values")
df = df.dropna(subset=['Position'])
print(f"Dataset shape after dropping missing targets: {df.shape}")

# Display column data types
print("\nColumn data types:")
print(df.dtypes)

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Separate target from features
y = df['Position']
X = df.drop('Position', axis=1)

print(f"\nTarget distribution:")
print(y.value_counts().head(10))

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}... (showing first 5)")

# Handle Timestamp column specially - drop it as it's not useful for prediction
if 'Timestamp' in categorical_cols:
    X = X.drop('Timestamp', axis=1)
    categorical_cols.remove('Timestamp')
    print("\nDropped 'Timestamp' column (not useful for prediction)")

# Clean salary columns - handle extreme outliers
salary_col = 'Yearly brutto salary (without bonus and stocks) in EUR'
if salary_col in numeric_cols:
    # Replace extremely high values (likely data entry errors) with median
    median_salary = X[salary_col].median()
    X.loc[X[salary_col] > 1000000, salary_col] = median_salary
    print(f"Cleaned '{salary_col}': replaced outliers > 1,000,000 with median")

# Fit and transform - numeric imputation with mean
X_numeric_filled = X[numeric_cols].fillna(X[numeric_cols].mean())
print(f"\nImputed {len(numeric_cols)} numeric columns with mean values")

# Categorical imputation with mode
X_categorical_filled = X[categorical_cols].copy()
for col in categorical_cols:
    mode_val = X_categorical_filled[col].mode()
    if len(mode_val) > 0:
        X_categorical_filled[col] = X_categorical_filled[col].fillna(mode_val[0])
    else:
        X_categorical_filled[col] = X_categorical_filled[col].fillna('Unknown')
print(f"Imputed {len(categorical_cols)} categorical columns with mode values")

# Combine processed features
X_processed = pd.concat([X_numeric_filled, X_categorical_filled], axis=1)

print(f"\nMissing values after imputation: {X_processed.isnull().sum().sum()}")

# ============================================================================
# 4. TRAIN-TEST SPLIT (80/20)
# ============================================================================
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

class_counts = y.value_counts()
min_class_size = int(class_counts.min())
use_stratify = min_class_size >= 2

if use_stratify:
    print("Using stratified split because every class has at least 2 samples.")
    stratify_labels = y
else:
    print("Skipping stratified split because some target classes have fewer than 2 samples.")
    print(f"Smallest class size: {min_class_size}")
    stratify_labels = None

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels,
)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nTraining set target distribution:")
print(y_train.value_counts().head(5))

# ============================================================================
# 5. ENCODE CATEGORICAL FEATURES AND SCALE NUMERIC FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("ENCODING CATEGORICAL AND SCALING NUMERIC FEATURES")
print("=" * 80)

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=100)
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_cols])
X_test_cat_encoded = encoder.transform(X_test[categorical_cols])

print(f"Encoded categorical features shape: {X_train_cat_encoded.shape}")

# Scale numeric features
scaler = StandardScaler()
X_train_numeric_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_numeric_scaled = scaler.transform(X_test[numeric_cols])

print(f"Scaled numeric features shape: {X_train_numeric_scaled.shape}")

# Combine encoded categorical and scaled numeric features
X_train_final = np.hstack([X_train_numeric_scaled, X_train_cat_encoded])
X_test_final = np.hstack([X_test_numeric_scaled, X_test_cat_encoded])

print(f"Final training features shape: {X_train_final.shape}")
print(f"Final test features shape: {X_test_final.shape}")

# ============================================================================
# 6. TRAIN MULTICLASS CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING MULTICLASS CLASSIFIER")
print("=" * 80)

# Train RandomForestClassifier
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
model.fit(X_train_final, y_train)
print("Model training completed!")

# ============================================================================
# 7. EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Predictions on test set
y_pred = model.predict(X_test_final)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n{'PERFORMANCE METRICS':^80}")
print("-" * 80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Classification report
print(f"\n{'CLASSIFICATION REPORT':^80}")
print("-" * 80)
print(classification_report(y_test, y_pred, zero_division=0, digits=3))

# ============================================================================
# 8. DISPLAY FIRST 10 PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS")
print("=" * 80)

results_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Match': y_test.values[:10] == y_pred[:10]
})

print("\n" + results_df.to_string(index=False))

print("\n" + "=" * 80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 80)
