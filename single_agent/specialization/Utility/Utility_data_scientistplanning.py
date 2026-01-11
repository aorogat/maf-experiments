
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 2: Load the dataset
df = pd.read_csv('Utility.csv')

# Step 3: Verify the contents of the DataFrame
print(df.head())

# Step 4: Identify and separate the target column 'CSRI' from feature columns
X = df.drop(columns=['CSRI'])
y = df['CSRI']

# Step 5: Check the data types of each column
print(X.dtypes)

# Identify categorical columns that require encoding
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessing and modeling pipeline
# Here we use OneHotEncoder for categorical columns and RandomForestRegressor for the model
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep the rest of the columns
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Display the first 10 predicted CSRI values alongside the actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head(10))
