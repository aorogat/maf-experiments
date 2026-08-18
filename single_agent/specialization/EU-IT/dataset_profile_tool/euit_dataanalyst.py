
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PROFILE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING AND PROFILING DATA")
print("=" * 80)

# Load the dataset
df = pd.read_csv('EU-IT_cleaned.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Get basic info
print(f"\nDataset Info:")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"\nColumn data types:\n{df.dtypes}")

# Check for missing values
print(f"\nMissing values per column:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])

# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CLEANING DATA")
print("=" * 80)

# Remove rows with missing target values
initial_rows = len(df)
df = df.dropna(subset=['Position'])
cleaned_rows = len(df)
print(f"\nDropped {initial_rows - cleaned_rows} rows with missing target values")
print(f"Remaining rows: {cleaned_rows}")

# Separate target and features
X = df.drop('Position', axis=1)
y = df['Position']

print(f"\nTarget variable (Position) distribution:")
print(y.value_counts().head(10))

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features[:10]}...")
print(f"Categorical features ({len(categorical_features)}): {categorical_features[:10]}...")

# Impute missing numeric values with mean
print("\nImputing numeric features with mean...")
for col in numeric_features:
    if X[col].isnull().sum() > 0:
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        print(f"  - {col}: filled {X[col].isnull().sum()} missing values with mean {mean_val:.2f}")

# Impute missing categorical values with mode (most frequent)
print("\nImputing categorical features with mode...")
for col in categorical_features:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
        X[col].fillna(mode_val, inplace=True)
        print(f"  - {col}: filled {X[col].isnull().sum()} missing values with mode '{mode_val}'")

# Verify no missing values remain
remaining_missing = X.isnull().sum().sum()
print(f"\nTotal remaining missing values: {remaining_missing}")

# Remove columns with extremely high cardinality (e.g., Timestamp) that don't provide predictive value
high_cardinality_threshold = 0.8 * len(X)
columns_to_drop = []
for col in categorical_features:
    unique_count = X[col].nunique()
    if unique_count > high_cardinality_threshold:
        columns_to_drop.append(col)
        print(f"Dropping high-cardinality column: {col} (unique values: {unique_count})")

X = X.drop(columns=columns_to_drop, errors='ignore')

# Update categorical features list
categorical_features = [col for col in categorical_features if col not in columns_to_drop]

print(f"\nCategorical features after filtering: {len(categorical_features)}")
print(f"Numeric features: {len(numeric_features)}")

# ============================================================================
# STEP 4: ENCODING AND SCALING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: ENCODING AND SCALING FEATURES")
print("=" * 80)

# Use ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Fit and transform features
X_preprocessed = preprocessor.fit_transform(X)
print(f"\nPreprocessed feature matrix shape: {X_preprocessed.shape}")
print(f"Numeric features scaled: {len(numeric_features)}")
print(f"Categorical features one-hot encoded")

# ============================================================================
# STEP 5: ENCODE TARGET VARIABLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: ENCODING TARGET VARIABLE")
print("=" * 80)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nTarget classes: {len(label_encoder.classes_)}")
print(f"Sample target classes: {label_encoder.classes_[:10]}")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

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
    X_preprocessed,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels,
)
print(f"\nTrain set size: {X_train.shape[0]} ({100*X_train.shape[0]/len(X_preprocessed):.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({100*X_test.shape[0]/len(X_preprocessed):.1f}%)")
print(f"Feature dimensionality: {X_train.shape[1]}")

# ============================================================================
# STEP 7: TRAIN MULTICLASS CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: TRAINING MULTICLASS CLASSIFIER")
print("=" * 80)

# Train GradientBoostingClassifier (excellent for multiclass problems)
print("\nTraining GradientBoostingClassifier...")
clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=0
)
clf.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# STEP 8: MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: MAKING PREDICTIONS")
print("=" * 80)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(f"\nPredictions generated for train and test sets")

# ============================================================================
# STEP 9: EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: MODEL EVALUATION")
print("=" * 80)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)

train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)

train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

# Print results
print("\n" + "-" * 80)
print("TRAINING SET METRICS")
print("-" * 80)
print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1-Score:  {train_f1:.4f}")

print("\n" + "-" * 80)
print("TEST SET METRICS (80/20 Split - Final Evaluation)")
print("-" * 80)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# ============================================================================
# STEP 10: CLASSIFICATION REPORT
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("=" * 80)

# Map predictions back to original class names for readability
y_pred_labels = label_encoder.inverse_transform(y_pred_test)
y_test_labels = label_encoder.inverse_transform(y_test)

print("\nClassification Report:")
print(classification_report(
    y_test_labels,
    y_pred_labels,
    zero_division=0
))

# ============================================================================
# STEP 11: DISPLAY FIRST 10 PREDICTIONS VS ACTUAL
# ============================================================================
print("\n" + "=" * 80)
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS (Test Set)")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Actual Position': y_test_labels[:10],
    'Predicted Position': y_pred_labels[:10],
    'Match': y_test_labels[:10] == y_pred_labels[:10]
})

print("\n" + comparison_df.to_string(index=False))

# Additional statistics
matches = (y_test_labels[:10] == y_pred_labels[:10]).sum()
print(f"\nMatches in first 10 predictions: {matches}/10 ({100*matches/10:.1f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print(f"""
✓ Dataset loaded: {initial_rows} rows → {cleaned_rows} rows (cleaned)
✓ Features: {X.shape[1]} columns (numeric: {len(numeric_features)}, categorical: {len(categorical_features)})
✓ Target classes: {len(label_encoder.classes_)} unique positions
✓ Train/Test split: 80/20 ({X_train.shape[0]}/{X_test.shape[0]})
✓ Model: GradientBoostingClassifier
✓ Test Accuracy: {test_accuracy:.4f}
✓ Test Precision: {test_precision:.4f}
✓ Test Recall: {test_recall:.4f}
✓ Test F1-Score: {test_f1:.4f}
""")

print("=" * 80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 80)
