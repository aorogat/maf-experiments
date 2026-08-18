
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PROFILE DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING AND PROFILING DATA")
print("="*80)

# Load the dataset
df = pd.read_csv('Yelp_Merged.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Display basic info
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nFirst few rows:\n{df.head()}")

# Check target column
print(f"\nTarget column 'stars' info:")
print(f"Data type: {df['stars'].dtype}")
print(f"Missing values: {df['stars'].isna().sum()}")
print(f"Unique values: {sorted(df['stars'].unique())}")
print(f"Value counts:\n{df['stars'].value_counts().sort_index()}")

# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CLEANING DATA")
print("="*80)

# Remove rows with missing target values (if any)
initial_rows = len(df)
df = df[df['stars'].notna()].copy()
print(f"\nRows after removing missing targets: {len(df)} (removed {initial_rows - len(df)})")

# Identify missing values per column
missing_info = df.isna().sum()
missing_columns = missing_info[missing_info > 0]
print(f"\nColumns with missing values:")
print(missing_columns)

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

# Separate features and target
X = df.drop('stars', axis=1)
y = df['stars']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Categorical columns: {categorical_cols}")

# ============================================================================
# IMPUTATION
# ============================================================================
print("\nPerforming imputation...")

# Impute numerical features with mean
for col in numerical_cols:
    if X[col].isna().sum() > 0:
        mean_value = X[col].mean()
        X[col].fillna(mean_value, inplace=True)
        print(f"  Imputed '{col}' with mean: {mean_value:.4f}")

# Impute categorical features with most frequent value
for col in categorical_cols:
    if X[col].isna().sum() > 0:
        mode_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'UNKNOWN'
        X[col].fillna(mode_value, inplace=True)
        print(f"  Imputed '{col}' with mode: {mode_value}")

print(f"\nMissing values after imputation: {X.isna().sum().sum()}")

# ============================================================================
# ENCODING CATEGORICAL FEATURES
# ============================================================================
print("\nEncoding categorical features...")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le
    print(f"  Encoded '{col}' with {len(le.classes_)} unique values")

# ============================================================================
# SCALING NUMERICAL FEATURES
# ============================================================================
print("\nScaling numerical features...")

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print(f"  Scaled {len(numerical_cols)} numerical columns")

X = X_scaled

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: MODEL TRAINING")
print("="*80)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Training target distribution:\n{y_train.value_counts().sort_index()}")
print(f"Test target distribution:\n{y_test.value_counts().sort_index()}")

# Train RandomForestClassifier
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# STEP 5: EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION")
print("="*80)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')

print(f"\n{'='*50}")
print("PERFORMANCE METRICS")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Classification Report
print(f"\n{'='*50}")
print("CLASSIFICATION REPORT")
print(f"{'='*50}")
print(classification_report(
    y_test, y_pred, 
    zero_division=0,
    digits=4
))

# ============================================================================
# FIRST 10 PREDICTIONS vs ACTUAL
# ============================================================================
print(f"\n{'='*50}")
print("FIRST 10 PREDICTIONS vs ACTUAL LABELS")
print(f"{'='*50}")

comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_pred[:10],
    'Match': y_test.iloc[:10].values == y_pred[:10]
})

print(f"\n{comparison_df.to_string(index=True)}")
print(f"\nCorrectly predicted: {comparison_df['Match'].sum()}/10")

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("="*80)
