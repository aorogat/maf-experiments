
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 2: Load the dataset
data = pd.read_csv('Utility.csv')

# Step 3: Examine the data
print(data.head())
print(data.info())
print(data.describe())

# Step 4: Identify the target column and feature columns
target_column = 'CSRI'
feature_columns = data.columns[data.columns != target_column]

# Step 5: Split the dataset into training and testing sets
X = data[feature_columns]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Encode categorical variables
# Identify categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

# One-hot encoding using pandas
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Align the training and test datasets to ensure they have the same columns after encoding
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# Step 7: Train the regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_encoded, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_encoded)

# Step 9: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Step 10: Display results
results = pd.DataFrame({'Actual CSRI': y_test, 'Predicted CSRI': y_pred})
print(results.head(10))
