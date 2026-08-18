
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the dataset
print("=" * 80)
print("STEP 1: LOADING DATASET")
print("=" * 80)
df = pd.read_csv('Utility.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn data types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Step 2: Data Profiling and Cleaning
print("\n" + "=" * 80)
print("STEP 2: DATA PROFILING AND CLEANING")
print("=" * 80)
print(f"\nTarget column 'CSRI' statistics:")
print(df['CSRI'].describe())

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('CSRI')  # Remove target from features

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")
print(f"Target column: CSRI")

# Step 3: Feature Engineering
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Make a copy for feature engineering
df_processed = df.copy()

# Handle missing values in numerical features (impute with mean)
for col in numerical_features:
    if df_processed[col].isnull().sum() > 0:
        mean_value = df_processed[col].mean()
        df_processed[col].fillna(mean_value, inplace=True)
        print(f"Imputed {col} with mean: {mean_value}")

# Handle missing values in categorical features (impute with mode)
for col in categorical_features:
    if df_processed[col].isnull().sum() > 0:
        mode_value = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_value, inplace=True)
        print(f"Imputed {col} with mode: {mode_value}")

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}")

# Prepare features and target
X = df_processed[numerical_features + categorical_features]
y = df_processed['CSRI']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Step 4: Train-Test Split
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT (80/20)")
print("=" * 80)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]} ({100*0.8:.0f}%)")
print(f"Test set size: {X_test.shape[0]} ({100*0.2:.0f}%)")

# Step 5: Scale Numerical Features
print("\n" + "=" * 80)
print("STEP 5: FEATURE SCALING")
print("=" * 80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Applied StandardScaler to both training and test sets")
print(f"Scaled training set shape: {X_train_scaled.shape}")
print(f"Scaled test set shape: {X_test_scaled.shape}")

# Step 6: Model Training
print("\n" + "=" * 80)
print("STEP 6: MODEL TRAINING")
print("=" * 80)

# Train GradientBoostingRegressor (best for this type of data)
print("\nTraining GradientBoostingRegressor...")
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=0
)
model.fit(X_train_scaled, y_train)
print("GradientBoostingRegressor training completed")

# Step 7: Model Evaluation
print("\n" + "=" * 80)
print("STEP 7: MODEL EVALUATION")
print("=" * 80)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate MAE
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nMean Absolute Error (MAE):")
print(f"  Training MAE: {train_mae:.6f}")
print(f"  Test MAE: {test_mae:.6f}")

# Step 8: Display First 10 Predictions vs Actual Values
print("\n" + "=" * 80)
print("FIRST 10 PREDICTIONS VS ACTUAL VALUES (TEST SET)")
print("=" * 80)

results_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_test_pred[:10],
    'Absolute Error': np.abs(y_test.values[:10] - y_test_pred[:10])
})
results_df.index = range(1, 11)
results_df.index.name = 'Sample'

print(f"\n{results_df.to_string()}")

print("\n" + "=" * 80)
print("MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nSummary:")
print(f"  - Dataset: Utility.csv (4574 rows, 13 columns)")
print(f"  - Target: CSRI (Continuous)")
print(f"  - Model: GradientBoostingRegressor")
print(f"  - Train/Test Split: 80/20 ({X_train.shape[0]}/{X_test.shape[0]})")
print(f"  - Test MAE: {test_mae:.6f}")
print(f"  - Features used: {len(numerical_features)} numerical + {len(categorical_features)} categorical")
