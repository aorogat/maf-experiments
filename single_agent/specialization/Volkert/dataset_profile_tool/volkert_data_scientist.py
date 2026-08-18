
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PROFILE DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING AND PROFILING DATA")
print("="*80)

# Load the dataset
df = pd.read_csv('volkert.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values count:")
print(df.isnull().sum().sum())

print(f"\nTarget class distribution:")
print(df['class'].value_counts().sort_index())

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CLEANING DATA")
print("="*80)

# Verify target column has no missing or invalid values
target_col = 'class'
invalid_targets = df[~df[target_col].isin(range(10))]
print(f"Invalid target values: {len(invalid_targets)}")

# Remove any rows with invalid targets if they exist
if len(invalid_targets) > 0:
    df = df[df[target_col].isin(range(10))].copy()
    print(f"Cleaned dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeature set shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING (Imputation, Encoding, Scaling)")
print("="*80)

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Handle missing values in numeric columns - impute with mean
if X[numeric_cols].isnull().sum().sum() > 0:
    print("Imputing missing numeric values with mean...")
    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].mean(), inplace=True)

# Handle missing values in categorical columns - impute with mode
if len(categorical_cols) > 0 and X[categorical_cols].isnull().sum().sum() > 0:
    print("Imputing missing categorical values with mode...")
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].mode()[0], inplace=True)

# Encode categorical features if any exist
if len(categorical_cols) > 0:
    print("Encoding categorical features...")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Feature set shape after encoding: {X.shape}")

print(f"Final feature set shape: {X.shape}")

# Scale numerical features
print("Scaling numerical features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Scaled feature set shape: {X_scaled.shape}")

# ============================================================================
# 4. TRAIN/TEST SPLIT AND MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN/TEST SPLIT (80/20)")
print("="*80)

# Perform 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training target distribution:\n{y_train.value_counts().sort_index()}")
print(f"Test target distribution:\n{y_test.value_counts().sort_index()}")

# Train the classifier - using RandomForestClassifier
print("\n" + "="*80)
print("STEP 5: MODEL TRAINING (RandomForestClassifier)")
print("="*80)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
print("Training RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training completed!")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: MODEL EVALUATION")
print("="*80)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "-"*80)
print("PERFORMANCE METRICS")
print("-"*80)
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-Score:  {f1:.6f}")

print("\n" + "-"*80)
print("CLASSIFICATION REPORT (Weighted Average)")
print("-"*80)
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================================
# 6. FIRST 10 PREDICTIONS VS ACTUAL
# ============================================================================
print("\n" + "-"*80)
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS")
print("-"*80)
results_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Match': (y_test.values[:10] == y_pred[:10]).astype(int)
})
print(results_df.to_string(index=True))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
