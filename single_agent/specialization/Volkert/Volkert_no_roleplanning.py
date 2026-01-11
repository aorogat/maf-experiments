
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('volkert.csv')

# Print the first few rows of the DataFrame
print(data.head())

# Handle missing data and drop non-numeric columns
data = data.dropna()  # Drop rows with missing values
data = data.select_dtypes(include=[np.number])  # Keep only numeric columns

# Define features and target
X = data.drop('class', axis=1)
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for scaling and training the model
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Apply scaling
    ('classifier', RandomForestClassifier(random_state=42))  # Train a RandomForest classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model using Accuracy, Precision, Recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the classification report
print(classification_report(y_test, y_pred))

# Print accuracy, precision, recall, and F1-score
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

# Print the first 10 predicted and actual class values
print("Predicted classes:", y_pred[:10])
print("Actual classes:", y_test.values[:10])
