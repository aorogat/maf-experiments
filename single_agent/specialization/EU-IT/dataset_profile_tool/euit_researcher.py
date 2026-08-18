
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
# 1. LOAD AND PROFILE DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING AND PROFILING DATA")
print("="*80)

# Load the dataset
df = pd.read_csv('EU-IT_cleaned.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nMissing values per column:")
print(df.isnull().sum())

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CLEANING DATA")
print("="*80)

# Remove rows where target (Position) is missing
initial_rows = len(df)
df = df.dropna(subset=['Position'])
print(f"\nRows after removing missing targets: {len(df)} (removed {initial_rows - len(df)} rows)")

# Separate features and target
X = df.drop('Position', axis=1)
y = df['Position']

print(f"Target distribution:")
print(y.value_counts().head(10))

# ============================================================================
# 3. IDENTIFY FEATURE TYPES AND HANDLE HIGH-CARDINALITY COLUMNS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

# Drop Timestamp column (too high cardinality, not useful)
if 'Timestamp' in X.columns:
    X = X.drop('Timestamp', axis=1)
    print("Dropped 'Timestamp' column (high cardinality)")

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Handle outliers in salary column (values > 1,000,000 seem like data errors)
if 'Yearly brutto salary (without bonus and stocks) in EUR' in numeric_cols:
    salary_col = 'Yearly brutto salary (without bonus and stocks) in EUR'
    Q1 = X[salary_col].quantile(0.25)
    Q3 = X[salary_col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR
    print(f"\n{salary_col}:")
    print(f"  Before: min={X[salary_col].min()}, max={X[salary_col].max()}")
    X[salary_col] = X[salary_col].clip(upper=upper_bound)
    print(f"  After clipping: min={X[salary_col].min()}, max={X[salary_col].max()}")

# Handle high-cardinality categorical columns
# Columns with too many unique values won't be one-hot encoded
# Instead, we'll group rare categories or drop them
high_cardinality_cols = []
for col in categorical_cols:
    cardinality = X[col].nunique()
    if cardinality > 50:
        high_cardinality_cols.append(col)
        print(f"\n{col}: {cardinality} unique values (high cardinality)")
        # Group rare categories (appear < 2 times) as "Other"
        value_counts = X[col].value_counts()
        rare_categories = value_counts[value_counts < 2].index
        X[col] = X[col].replace(rare_categories, 'Other')
        print(f"  Grouped {len(rare_categories)} rare categories as 'Other'")
        print(f"  New unique values: {X[col].nunique()}")

# ============================================================================
# 4. IMPUTATION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: HANDLING MISSING VALUES")
print("="*80)

# Impute numeric columns with mean
print(f"\nImputing numeric columns with mean:")
for col in numeric_cols:
    missing_count = X[col].isnull().sum()
    if missing_count > 0:
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mean={mean_val:.2f}")

# Impute categorical columns with most frequent value
print(f"\nImputing categorical columns with mode:")
for col in categorical_cols:
    missing_count = X[col].isnull().sum()
    if missing_count > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        print(f"  {col}: imputed {missing_count} missing values with mode='{mode_val}'")

# Verify no missing values remain
print(f"\nMissing values after imputation:")
print(X.isnull().sum().sum())

# ============================================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 5: ENCODING CATEGORICAL FEATURES")
print("="*80)

# Separate into numeric and categorical features after imputation
X_numeric = X[numeric_cols].copy()
X_categorical = X[categorical_cols].copy()

print(f"\nUsing OneHotEncoder for categorical features (max 50 categories per column):")
print(f"  Categorical features: {X_categorical.columns.tolist()}")

# Use OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=50)
X_categorical_encoded = onehot_encoder.fit_transform(X_categorical)
X_categorical_encoded = pd.DataFrame(
    X_categorical_encoded,
    columns=onehot_encoder.get_feature_names_out(X_categorical.columns),
    index=X_categorical.index,
)

# Combine encoded categorical with numeric
X_processed = pd.concat([X_numeric, X_categorical_encoded], axis=1)
print(f"\nFeatures after encoding: {X_processed.shape}")
print(f"Total features: {X_processed.shape[1]}")

# ============================================================================
# 6. SCALE NUMERICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SCALING NUMERICAL FEATURES")
print("="*80)

scaler = StandardScaler()
X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
print(f"Scaled {len(numeric_cols)} numeric features using StandardScaler")
print(f"\nFinal feature matrix shape: {X_processed.shape}")

# ============================================================================
# 7. ENCODE TARGET VARIABLE
# ============================================================================
print("\n" + "="*80)
print("STEP 7: ENCODING TARGET VARIABLE")
print("="*80)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Classes: {label_encoder.classes_[:10]}...")  # Show first 10

# ============================================================================
# 8. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 8: TRAIN-TEST SPLIT")
print("="*80)

class_counts = pd.Series(y_encoded).value_counts()
min_class_size = int(class_counts.min())
use_stratify = min_class_size >= 2

if use_stratify:
    print("\nUsing stratified split because every class has at least 2 samples.")
    stratify_labels = y_encoded
else:
    print("\nSkipping stratified split because some target classes have fewer than 2 samples.")
    print(f"Smallest class size: {min_class_size}")
    stratify_labels = None

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels,
)
print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train-test split: 80/20")

# ============================================================================
# 9. TRAIN CLASSIFIER
# ============================================================================
print("\n" + "="*80)
print("STEP 9: TRAINING CLASSIFIER")
print("="*80)

print("\nTraining RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
clf.fit(X_train, y_train)
print("RandomForestClassifier training complete!")

# ============================================================================
# 10. MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: MAKING PREDICTIONS")
print("="*80)

y_pred = clf.predict(X_test)
print(f"\nPredictions generated for {len(y_pred)} test samples")

# ============================================================================
# 11. EVALUATE PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("STEP 11: MODEL EVALUATION")
print("="*80)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n{'Metric':<20} {'Score':<10}")
print("-" * 30)
print(f"{'Accuracy':<20} {accuracy:<10.4f}")
print(f"{'Precision (weighted)':<20} {precision:<10.4f}")
print(f"{'Recall (weighted)':<20} {recall:<10.4f}")
print(f"{'F1-Score (weighted)':<20} {f1:<10.4f}")

# ============================================================================
# 12. CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)

# Decode predictions and actual labels for readability
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\n" + classification_report(
    y_test_labels, 
    y_pred_labels, 
    zero_division=0,
    digits=3
))

# ============================================================================
# 13. DISPLAY FIRST 10 PREDICTIONS vs ACTUAL
# ============================================================================
print("\n" + "="*80)
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS")
print("="*80)

comparison_df = pd.DataFrame({
    'Actual': y_test_labels[:10],
    'Predicted': y_pred_labels[:10],
    'Match': (y_test_labels[:10] == y_pred_labels[:10])
})

print("\n")
print(comparison_df.to_string(index=True))
print(f"\nMatches in first 10: {comparison_df['Match'].sum()}/10")

# ============================================================================
# 14. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("TOP 15 FEATURE IMPORTANCE")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n")
print(feature_importance.head(15).to_string(index=False))

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETE")
print("="*80)
