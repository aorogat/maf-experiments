
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATASET")
print("=" * 80)

df = pd.read_csv('EU-IT_cleaned.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target column 'Position' - Missing values: {df['Position'].isna().sum()}")
print()

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("=" * 80)
print("STEP 2: DATA CLEANING")
print("=" * 80)

# Drop rows with missing target values
df_clean = df.dropna(subset=['Position']).copy()
print(f"Rows after removing missing target: {len(df_clean)}")

# Identify numeric and categorical columns (excluding target and non-predictive columns)
non_predictive = ['Timestamp']  # High cardinality, not useful for prediction
df_clean = df_clean.drop(columns=non_predictive, errors='ignore')

# Separate features and target
X = df_clean.drop('Position', axis=1)
y = df_clean['Position']

print(f"Features shape: {X.shape}")
print(f"Target unique classes: {y.nunique()}")
print(f"Target distribution (top 10):\n{y.value_counts().head(10)}")
print()

# ============================================================================
# 3. FEATURE ENGINEERING - IMPUTATION, ENCODING, SCALING
# ============================================================================
print("=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}... (and {len(categorical_cols)-5} more)")
print()

# Handle outliers in salary column (cap extremely high values)
if 'Yearly brutto salary (without bonus and stocks) in EUR' in numeric_cols:
    salary_col = 'Yearly brutto salary (without bonus and stocks) in EUR'
    q95 = X[salary_col].quantile(0.95)
    X[salary_col] = X[salary_col].clip(upper=q95 * 2)  # Cap at 2x the 95th percentile
    print(f"Salary outliers capped at: {q95 * 2:.2f}")

# Fill missing values
print("Imputing missing values...")
for col in numeric_cols:
    mean_val = X[col].mean()
    X[col].fillna(mean_val, inplace=True)

for col in categorical_cols:
    mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
    X[col].fillna(mode_val, inplace=True)

print(f"Missing values after imputation: {X.isna().sum().sum()}")
print()

# ============================================================================
# 4. PREPROCESSING PIPELINE - ENCODING & SCALING
# ============================================================================
print("=" * 80)
print("STEP 4: ENCODING & SCALING")
print("=" * 80)

# Select categorical columns that can be one-hot encoded (limit cardinality)
categorical_to_encode = []
for col in categorical_cols:
    if X[col].nunique() <= 20:  # Only encode low-cardinality features
        categorical_to_encode.append(col)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20), categorical_to_encode)
    ],
    remainder='drop'  # Drop high-cardinality categorical features
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
print(f"Features after preprocessing: {X_processed.shape}")
print(f"Processed feature matrix dtype: {X_processed.dtype}")
print()

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Target classes: {len(le.classes_)}")
print(f"Sample classes: {list(le.classes_[:5])}")
print()

# ============================================================================
# 5. TRAIN-TEST SPLIT (80/20)
# ============================================================================
print("=" * 80)
print("STEP 5: TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

class_counts = pd.Series(y_encoded).value_counts()
min_class_size = int(class_counts.min())
use_stratify = min_class_size >= 2

if use_stratify:
    print("Using stratified split because every class has at least 2 samples.")
    stratify_labels = y_encoded
else:
    print(
        "Skipping stratified split because some target classes have fewer than 2 samples."
    )
    print(f"Smallest class size: {min_class_size}")
    stratify_labels = None

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels,
)

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X_processed)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X_processed)*100:.1f}%)")
print(f"Training features: {X_train.shape[1]}")
print()

# ============================================================================
# 6. TRAIN MULTICLASS CLASSIFIER
# ============================================================================
print("=" * 80)
print("STEP 6: TRAINING RANDOMFORESTCLASSIFIER")
print("=" * 80)

# Train RandomForestClassifier for multiclass prediction
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

print("Training model...")
clf.fit(X_train, y_train)
print("Model training completed!")
print()

# ============================================================================
# 7. EVALUATE MODEL
# ============================================================================
print("=" * 80)
print("STEP 7: MODEL EVALUATION")
print("=" * 80)

# Make predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

# Print metrics
print("TRAINING SET METRICS:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print()

print("TEST SET METRICS:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print()

# ============================================================================
# 8. DETAILED CLASSIFICATION REPORT
# ============================================================================
print("=" * 80)
print("CLASSIFICATION REPORT (Test Set)")
print("=" * 80)
all_labels = np.arange(len(le.classes_))
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=all_labels,
        target_names=le.classes_,
        zero_division=0,
        digits=3,
    )
)
print()

# ============================================================================
# 9. FIRST 10 PREDICTIONS vs ACTUAL
# ============================================================================
print("=" * 80)
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS (Test Set)")
print("=" * 80)

results_df = pd.DataFrame({
    'Actual': le.inverse_transform(y_test[:10]),
    'Predicted': le.inverse_transform(y_test_pred[:10]),
    'Match': y_test[:10] == y_test_pred[:10]
})

print(results_df.to_string(index=False))
print()

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("=" * 80)
print("PIPELINE EXECUTION SUMMARY")
print("=" * 80)
print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"✓ Data cleaned: {len(X)} valid samples retained")
print(f"✓ Features engineered: {X_processed.shape[1]} features")
print(f"✓ Model trained: RandomForestClassifier with {len(le.classes_)} classes")
print(f"✓ Test Accuracy: {test_accuracy:.4f}")
print(f"✓ Test F1-Score (weighted): {test_f1:.4f}")
print("=" * 80)
