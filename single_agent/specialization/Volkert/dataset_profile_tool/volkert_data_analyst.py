
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Load the dataset with comma delimiter
df = pd.read_csv('volkert.csv', delimiter=',')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")

# ============================================================================
# 2. PROFILE DATA
# ============================================================================
print("\n" + "=" * 80)
print("DATA PROFILING")
print("=" * 80)

print(f"\nDataset Info:")
print(f"  - Total rows: {df.shape[0]}")
print(f"  - Total columns: {df.shape[1]}")
print(f"  - Target column: 'class'")
print(f"  - Target data type: {df['class'].dtype}")

print(f"\nMissing values summary:")
missing_counts = df.isnull().sum()
if missing_counts.sum() == 0:
    print("  - No missing values detected")
else:
    print(f"  - Total missing values: {missing_counts.sum()}")
    print(missing_counts[missing_counts > 0])

print(f"\nTarget class distribution:")
class_dist = df['class'].value_counts().sort_index()
for cls, count in class_dist.items():
    print(f"  - Class {cls}: {count} samples")

print(f"\nFeature types:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('class')  # Remove target from feature list
print(f"  - Numeric features: {len(numeric_cols)}")
print(f"  - Categorical features: 0")

# ============================================================================
# 3. CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

print(f"\nFeatures shape before cleaning: {X.shape}")

# Remove constant/zero-variance features (features with cardinality of 1 or very low variance)
print("\nIdentifying and removing low-variance features...")
variances = X.var()
constant_features = variances[variances == 0].index.tolist()
print(f"  - Features with zero variance: {len(constant_features)}")
if len(constant_features) > 0:
    X = X.drop(columns=constant_features)

print(f"Features shape after removing constant features: {X.shape}")

# Check for NaN values and handle if present
if X.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    X = X.fillna(X.mean())
    print("  - Numeric features imputed with mean")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

print(f"\nNumeric feature scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print(f"  - Scaling applied to {X_scaled.shape[1]} features")
print(f"  - Features shape after scaling: {X_scaled.shape}")

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training set class distribution:")
for cls in sorted(y_train.unique()):
    print(f"  - Class {cls}: {(y_train == cls).sum()} samples")

# ============================================================================
# 6. MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# 7. PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("PREDICTIONS")
print("=" * 80)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Predictions generated for both training and test sets")

# ============================================================================
# 8. EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION - TEST SET")
print("=" * 80)

# Compute metrics for test set
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

print(f"\nPerformance Metrics (Test Set):")
print(f"  - Accuracy:  {accuracy:.6f}")
print(f"  - Precision: {precision:.6f}")
print(f"  - Recall:    {recall:.6f}")
print(f"  - F1-Score:  {f1:.6f}")

# Full classification report
print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, zero_division=0))

# ============================================================================
# 9. DETAILED RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("FIRST 10 PREDICTIONS VS ACTUAL LABELS")
print("=" * 80)

results_df = pd.DataFrame({
    'Predicted': y_pred_test[:10],
    'Actual': y_test.values[:10],
    'Match': (y_pred_test[:10] == y_test.values[:10]).astype(int)
})

print("\n" + results_df.to_string(index=False))

# ============================================================================
# 10. TRAINING SET EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION - TRAINING SET")
print("=" * 80)

accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
recall_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

print(f"\nPerformance Metrics (Training Set):")
print(f"  - Accuracy:  {accuracy_train:.6f}")
print(f"  - Precision: {precision_train:.6f}")
print(f"  - Recall:    {recall_train:.6f}")
print(f"  - F1-Score:  {f1_train:.6f}")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)

print(f"""
✓ Data Loading: Loaded volkert.csv successfully
✓ Data Profiling: 58,310 rows × 181 columns, target='class' (10 classes)
✓ Data Cleaning: Removed {len(constant_features)} constant-variance features
✓ Feature Engineering: Scaled {X_scaled.shape[1]} numeric features
✓ Train-Test Split: 80/20 stratified split
✓ Model: RandomForestClassifier (n_estimators=100, max_depth=20)
✓ Evaluation: Comprehensive metrics computed on test set

Final Test Set Performance:
  - Accuracy:  {accuracy:.6f}
  - Precision: {precision:.6f}
  - Recall:    {recall:.6f}
  - F1-Score:  {f1:.6f}
""")

print("=" * 80)
