
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD AND PROFILE DATA
# ==============================================================================
print("="*80)
print("STEP 1: LOAD AND PROFILE DATA")
print("="*80)

# Load dataset
df = pd.read_csv('volkert.csv', sep=',')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns[:10])}... (first 10 shown)")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing values")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nTarget column 'class' info:")
print(df['class'].value_counts().sort_index())

# ==============================================================================
# 2. CLEAN DATA
# ==============================================================================
print("\n" + "="*80)
print("STEP 2: CLEAN DATA")
print("="*80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target data type: {y.dtype}")

# Check for missing values
missing_features = X.isnull().sum()
missing_target = y.isnull().sum()

print(f"\nMissing values in features: {missing_features.sum()}")
print(f"Missing values in target: {missing_target}")

# Remove rows with missing target values (if any)
if missing_target > 0:
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"Removed {missing_target} rows with missing target values")

# Remove completely constant features (variance = 0)
constant_features = X.columns[X.nunique() <= 1]
print(f"\nIdentified {len(constant_features)} constant features to remove:")
print(constant_features.tolist())
X = X.drop(constant_features, axis=1)

print(f"Features shape after removing constant features: {X.shape}")

# ==============================================================================
# 3. ENGINEER FEATURES
# ==============================================================================
print("\n" + "="*80)
print("STEP 3: ENGINEER FEATURES (IMPUTATION, ENCODING, SCALING)")
print("="*80)

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Impute missing numerical values with mean
for col in numerical_features:
    if X[col].isnull().sum() > 0:
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        print(f"Imputed {col} with mean: {mean_val:.6f}")

# Impute missing categorical values with most frequent value
for col in categorical_features:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        print(f"Imputed {col} with mode: {mode_val}")

# Encode categorical features (if any exist)
if len(categorical_features) > 0:
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    print(f"\nAfter encoding categorical features, shape: {X.shape}")

print(f"\nFinal features shape before scaling: {X.shape}")
print(f"Data types:\n{X.dtypes.value_counts()}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"\nFeatures scaled using StandardScaler")
print(f"Mean (should be ~0): {X_scaled.mean().mean():.10f}")
print(f"Std (should be ~1): {X_scaled.std().mean():.10f}")

# ==============================================================================
# 4. SPLIT DATA
# ==============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN/TEST SPLIT (80/20)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Train set features: {X_train.shape[1]}")
print(f"\nTrain target distribution:\n{y_train.value_counts().sort_index()}")
print(f"\nTest target distribution:\n{y_test.value_counts().sort_index()}")

# ==============================================================================
# 5. TRAIN MULTICLASS CLASSIFIER
# ==============================================================================
print("\n" + "="*80)
print("STEP 5: TRAIN MULTICLASS CLASSIFIER")
print("="*80)

# Train RandomForestClassifier
print("\nTraining RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)
clf.fit(X_train, y_train)
print("RandomForestClassifier trained successfully!")

# ==============================================================================
# 6. MAKE PREDICTIONS
# ==============================================================================
print("\n" + "="*80)
print("STEP 6: MAKE PREDICTIONS")
print("="*80)

y_pred = clf.predict(X_test)

print(f"Predictions shape: {y_pred.shape}")
print(f"Unique predicted classes: {sorted(np.unique(y_pred))}")
print(f"Actual classes in test set: {sorted(np.unique(y_test))}")

# ==============================================================================
# 7. EVALUATE MODEL
# ==============================================================================
print("\n" + "="*80)
print("STEP 7: EVALUATE MODEL")
print("="*80)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')

print(f"\n{'Metric':<15} {'Score':<10}")
print("-" * 25)
print(f"{'Accuracy':<15} {accuracy:<10.6f}")
print(f"{'Precision':<15} {precision:<10.6f}")
print(f"{'Recall':<15} {recall:<10.6f}")
print(f"{'F1-Score':<15} {f1:<10.6f}")

# ==============================================================================
# 8. DETAILED CLASSIFICATION REPORT
# ==============================================================================
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)

print("\n" + classification_report(y_test, y_pred, zero_division=0))

# ==============================================================================
# 9. FIRST 10 PREDICTIONS VS ACTUAL
# ==============================================================================
print("\n" + "="*80)
print("FIRST 10 PREDICTIONS VS ACTUAL LABELS")
print("="*80)

comparison_df = pd.DataFrame({
    'Index': range(10),
    'Predicted': y_pred[:10],
    'Actual': y_test.values[:10],
    'Correct': (y_pred[:10] == y_test.values[:10]).astype(int)
})

print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETED")
print("="*80)
