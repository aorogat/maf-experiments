
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PROFILE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING AND PROFILING DATA")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Yelp_Merged.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CLEANING DATA")
print("=" * 80)

# Check target column
print(f"\nTarget column (stars) distribution:\n{df['stars'].value_counts().sort_index()}")
print(f"Missing values in target: {df['stars'].isnull().sum()}")

# Create a copy for processing
df_clean = df.copy()

# Identify numerical and categorical columns
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

# Remove target and ID columns from feature lists
target_col = 'stars'
id_cols = ['business_id', 'user_id']

if target_col in numerical_cols:
    numerical_cols.remove(target_col)

feature_cols = [col for col in numerical_cols + categorical_cols 
                if col not in id_cols and col != target_col]

print(f"\nNumerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Total feature columns: {len(feature_cols)}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING (Imputation, Encoding, Scaling)")
print("=" * 80)

# Separate features and target
X = df_clean[feature_cols].copy()
y = df_clean[target_col].copy()

# Convert target to integer class labels
y = y.astype(int)

# 3.1: Impute missing numerical values with mean
numerical_features = [col for col in feature_cols if col in numerical_cols]
for col in numerical_features:
    if X[col].isnull().sum() > 0:
        mean_val = X[col].mean()
        X[col].fillna(mean_val, inplace=True)
        print(f"Imputed {col} with mean: {mean_val:.4f}")

# 3.2: Impute missing categorical values with mode
categorical_features = [col for col in feature_cols if col in categorical_cols]
for col in categorical_features:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        print(f"Imputed {col} with mode: {mode_val}")

# 3.3: Encode categorical features using LabelEncoder
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded categorical feature: {col}")

# 3.4: Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
print(f"\nScaled {len(numerical_features)} numerical features")

# Verify no missing values remain
print(f"\nMissing values after cleaning: {X.isnull().sum().sum()}")

# ============================================================================
# 4. SPLIT DATA (80/20 train/test)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN/TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"\nTraining set class distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"\nTest set class distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

# ============================================================================
# 5. TRAIN MULTICLASS CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 80)

# Train RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining model...")
clf.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# 6. EVALUATE MODEL
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

print(f"\n{'PERFORMANCE METRICS':^80}")
print("=" * 80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (weighted)")
print(f"Recall:    {recall:.4f} (weighted)")
print(f"F1-Score:  {f1:.4f} (weighted)")

# Classification Report
print(f"\n{'DETAILED CLASSIFICATION REPORT':^80}")
print("=" * 80)
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================================
# 7. DISPLAY FIRST 10 PREDICTIONS VS ACTUAL
# ============================================================================
print(f"\n{'FIRST 10 PREDICTIONS VS ACTUAL LABELS':^80}")
print("=" * 80)
results_df = pd.DataFrame({
    'Predicted': y_pred[:10],
    'Actual': y_test.values[:10],
    'Match': (y_pred[:10] == y_test.values[:10])
})
print(results_df.to_string(index=True))

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
