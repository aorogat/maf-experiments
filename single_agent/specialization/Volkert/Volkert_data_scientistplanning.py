
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset 'volkert.csv' from the current working directory
data = pd.read_csv('volkert.csv')

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Handle missing data: Dropping rows with missing values
data = data.dropna()

# Identify non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
print("\nNon-numeric columns to be dropped:", non_numeric_columns)

# Dropping non-numeric columns
data = data.drop(columns=non_numeric_columns)

# Define features and target variable
X = data.drop(columns='class')
y = data['class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline for preprocessing and model training
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# Create a pipeline that first preprocesses the data and then applies a classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the classification model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("\nEvaluation of the model:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the first 10 predicted and actual values
print("\nFirst 10 predicted values:", y_pred[:10])
print("First 10 actual values:", y_test.values[:10])
