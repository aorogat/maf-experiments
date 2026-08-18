
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PROFILE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD AND PROFILE DATA")
print("=" * 80)

# Load the dataset
df = pd.read_csv('volkert.csv')
print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")
print(f"\nTarget column 'class' distribution:")
print(df['class'].value_counts().sort_index())

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CLEAN DATA")
print("=" * 80)

# Check for missing values in target column
print(f"\nMissing values in 'class': {df['class'].isnull().sum()}")

# Remove rows with missing target values if any
df = df[df['class'].notna()].copy()
print(f"Shape after removing missing targets: {df.shape}")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target classes: {sorted(y.unique())}")
print(f"Number of classes: {y.nunique()}")

# ============================================================================
# 3. ENGINEER FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: ENGINEER FEATURES")
print("=" * 80)

# Identify and remove constant features (zero variance)
constant_features = []
for col in X.columns:
    if X[col].nunique() == 1:
        constant_features.append(col)

print(f"\nConstant features (cardinality=1) found: {len(constant_features)}")
print(f"Examples: {constant_features[:10]}")

# Remove constant features
X = X.drop(columns=constant_features)
print(f"Feature matrix shape after removing constant features: {X.shape}")

# Handle missing values in numeric features with mean imputation
print(f"\nMissing values in features before imputation:")
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(missing_counts[missing_counts > 0])
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].mean(), inplace=True)
    print("Missing values imputed with mean")
else:
    print("No missing values found in features")

print(f"\nMissing values in features after imputation:")
print(f"Total missing: {X.isnull().sum().sum()}")

# Scale numerical features
print(f"\nScaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Features scaled successfully!")
print(f"Feature matrix shape: {X.shape}")
print(f"Feature statistics after scaling:")
print(f"  Mean (first 5 features): {X.iloc[:, :5].mean().values}")
print(f"  Std (first 5 features): {X.iloc[:, :5].std().values}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTrain target distribution:")
print(y_train.value_counts().sort_index())
print(f"\nTest target distribution:")
print(y_test.value_counts().sort_index())

# ============================================================================
# 5. TRAIN MULTICLASS CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAIN MULTICLASS CLASSIFIER")
print("=" * 80)

print(f"\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train, y_train)
print(f"Model trained successfully!")

# ============================================================================
# 6. MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MAKE PREDICTIONS")
print("=" * 80)

y_pred = model.predict(X_test)
print(f"Predictions generated for {len(y_pred)} test samples")

# ============================================================================
# 7. EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: EVALUATE MODEL PERFORMANCE")
print("=" * 80)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')

print(f"\n{'Metric':<20} {'Score':<15}")
print("-" * 35)
print(f"{'Accuracy':<20} {accuracy:<15.6f}")
print(f"{'Precision (weighted)':<20} {precision:<15.6f}")
print(f"{'Recall (weighted)':<20} {recall:<15.6f}")
print(f"{'F1-Score (weighted)':<20} {f1:<15.6f}")

# ============================================================================
# 8. DETAILED CLASSIFICATION REPORT
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)

report = classification_report(
    y_test, y_pred, 
    zero_division=0,
    target_names=[str(i) for i in sorted(y.unique())]
)
print("\n" + report)

# ============================================================================
# 9. FIRST 10 PREDICTIONS VS ACTUAL
# ============================================================================
print("\n" + "=" * 80)
print("FIRST 10 PREDICTIONS VS ACTUAL LABELS")
print("=" * 80)

results_df = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_pred[:10],
    'Match': (y_test.iloc[:10].values == y_pred[:10])
})

print("\n" + results_df.to_string(index=False))

print(f"\n{'Correct predictions in first 10:':<35} {results_df['Match'].sum()} out of 10")

# ============================================================================
# 10. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + feature_importance.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 80)
