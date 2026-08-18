
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

df = pd.read_csv('EU-IT_cleaned.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# 2. IDENTIFY TARGET AND SEPARATE FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PROFILING DATA")
print("="*80)

target_col = 'Position'
print(f"\nTarget column: {target_col}")
print(f"Target value counts:\n{df[target_col].value_counts().head(10)}")
print(f"Missing values in target: {df[target_col].isna().sum()}")

print(f"\nMissing values overview:")
missing_info = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))

# ============================================================================
# 3. CLEAN DATA - DROP ROWS WITH MISSING TARGET
# ============================================================================
print("\n" + "="*80)
print("STEP 3: CLEANING DATA")
print("="*80)

initial_rows = len(df)
df = df.dropna(subset=[target_col])
print(f"Removed rows with missing target: {initial_rows - len(df)} rows")
print(f"Remaining rows: {len(df)}")

# Separate target and features
y = df[target_col].copy()
X = df.drop(columns=[target_col]).copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: IDENTIFYING FEATURE TYPES")
print("="*80)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"\nCategorical features ({len(categorical_features)}): {categorical_features[:10]}...")

# ============================================================================
# 5. FEATURE ENGINEERING AND PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE ENGINEERING & PREPROCESSING")
print("="*80)

# Create copies for preprocessing
X_processed = X.copy()

# Drop Timestamp because its near-unique values add noise rather than signal.
if 'Timestamp' in X_processed.columns:
    X_processed = X_processed.drop(columns=['Timestamp'])
    categorical_features = [col for col in categorical_features if col != 'Timestamp']
    print("\nDropped 'Timestamp' column (high cardinality, low predictive value)")

# Handle numeric features - impute with mean
print("\nHandling numeric features:")
for col in numeric_features:
    missing_count = X_processed[col].isna().sum()
    if missing_count > 0:
        mean_val = X_processed[col].mean()
        X_processed[col].fillna(mean_val, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mean {mean_val:.2f}")
    else:
        print(f"  {col}: no missing values")

# Handle categorical features - impute with most frequent value
print("\nHandling categorical features:")
for col in categorical_features:
    missing_count = X_processed[col].isna().sum()
    if missing_count > 0:
        mode_val = X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'Unknown'
        X_processed[col].fillna(mode_val, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mode '{mode_val}'")
    else:
        print(f"  {col}: no missing values")

# Remove rows with any remaining NaN values
X_processed = X_processed.dropna()
y = y[X_processed.index]

print(f"\nAfter imputation and cleaning: {X_processed.shape}")

# Group singleton categories before encoding so rare labels do not dominate.
for col in categorical_features:
    value_counts = X_processed[col].value_counts()
    rare_categories = value_counts[value_counts < 2].index
    if len(rare_categories) > 0:
        X_processed[col] = X_processed[col].replace(rare_categories, 'Other')

# Identify low-cardinality categorical features for one-hot encoding
print("\nCategorical feature cardinality:")
low_cardinality_cats = []
high_cardinality_cats = []

for col in categorical_features:
    cardinality = X_processed[col].nunique()
    if cardinality <= 20:
        low_cardinality_cats.append(col)
        print(f"  {col}: {cardinality} unique values (ONE-HOT)")
    else:
        high_cardinality_cats.append(col)
        print(f"  {col}: {cardinality} unique values (LABEL)")

# ============================================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 6: ENCODING CATEGORICAL FEATURES")
print("="*80)

# One-hot encode low-cardinality categorical features
if low_cardinality_cats:
    print(f"\nOne-hot encoding: {low_cardinality_cats}")
    X_onehot = pd.get_dummies(X_processed[low_cardinality_cats], drop_first=True)
    X_processed = X_processed.drop(columns=low_cardinality_cats)
    X_processed = pd.concat([X_processed, X_onehot], axis=1)
    print(f"After one-hot encoding: {X_processed.shape[1]} features")

# Label encode high-cardinality categorical features
if high_cardinality_cats:
    print(f"\nLabel encoding: {high_cardinality_cats}")
    label_encoders = {}
    for col in high_cardinality_cats:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le
    print(f"After label encoding: {X_processed.shape[1]} features")

print(f"\nFinal feature matrix shape: {X_processed.shape}")
print(f"Features: {X_processed.columns.tolist()[:10]}...")

# ============================================================================
# 7. SCALE NUMERICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SCALING NUMERICAL FEATURES")
print("="*80)

scaler = StandardScaler()
numeric_cols_to_scale = numeric_features.copy()
X_processed[numeric_cols_to_scale] = scaler.fit_transform(X_processed[numeric_cols_to_scale])
print(f"Scaled {len(numeric_cols_to_scale)} numeric features")
print(f"Final feature matrix shape: {X_processed.shape}")

# ============================================================================
# 8. ENCODE TARGET VARIABLE
# ============================================================================
print("\n" + "="*80)
print("STEP 8: ENCODING TARGET VARIABLE")
print("="*80)

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"Target classes: {len(target_encoder.classes_)}")
print(f"Target encoding completed")

# ============================================================================
# 9. TRAIN-TEST SPLIT (80/20)
# ============================================================================
print("\n" + "="*80)
print("STEP 9: TRAIN-TEST SPLIT (80/20)")
print("="*80)

class_counts = pd.Series(y_encoded).value_counts()
min_class_size = int(class_counts.min())
use_stratify = min_class_size >= 2

if use_stratify:
    print("Using stratified split because every class has at least 2 samples.")
    stratify_labels = y_encoded
else:
    print("Skipping stratified split because some target classes have fewer than 2 samples.")
    print(f"Smallest class size: {min_class_size}")
    stratify_labels = None

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels,
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training set class distribution:\n{pd.Series(y_train).value_counts().head()}")
print(f"\nTest set class distribution:\n{pd.Series(y_test).value_counts().head()}")

# ============================================================================
# 10. TRAIN MULTICLASS CLASSIFIER
# ============================================================================
print("\n" + "="*80)
print("STEP 10: TRAINING MULTICLASS CLASSIFIER")
print("="*80)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

print("Training RandomForestClassifier...")
model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# 11. MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 11: MAKING PREDICTIONS")
print("="*80)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"Predictions generated for training set: {len(y_pred_train)}")
print(f"Predictions generated for test set: {len(y_pred_test)}")

# ============================================================================
# 12. EVALUATE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 12: MODEL EVALUATION")
print("="*80)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

test_precision = precision_score(y_test, y_pred_test, zero_division=0, average='weighted')
test_recall = recall_score(y_test, y_pred_test, zero_division=0, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, zero_division=0, average='weighted')

print("\n" + "-"*80)
print("OVERALL PERFORMANCE METRICS (Test Set)")
print("-"*80)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

print("\n" + "-"*80)
print("TRAINING vs TEST ACCURACY")
print("-"*80)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:     {test_accuracy:.4f}")

# ============================================================================
# 13. DETAILED CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("="*80)

class_names = target_encoder.classes_
report = classification_report(
    y_test,
    y_pred_test,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0,
    digits=3
)
print(report)

# ============================================================================
# 14. DISPLAY FIRST 10 PREDICTIONS vs ACTUAL
# ============================================================================
print("\n" + "="*80)
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS (Test Set)")
print("="*80)

results_df = pd.DataFrame({
    'Predicted': target_encoder.inverse_transform(y_pred_test[:10]),
    'Actual': target_encoder.inverse_transform(y_test[:10]),
    'Match': y_pred_test[:10] == y_test[:10]
})

print(results_df.to_string(index=True))

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("="*80)
