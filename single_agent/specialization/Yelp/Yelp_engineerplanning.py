
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('Yelp_Merged.csv')

# Display the first few rows of the dataframe to confirm it's loaded correctly
print(df.head())

# Handle missing data and drop non-numeric columns
df = df.drop(columns=['business_id', 'user_id', 'review_date'])
df = df.dropna()

# Define target and features
X = df.drop(columns=['stars'])
y = df['stars']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a preprocessing pipeline
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Define the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline that first transforms the data and then fits a model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the classification model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model using accuracy, precision, recall, and F1-score
print(classification_report(y_test, y_pred))

# Print the first 10 predicted and actual stars values
print("Predicted stars values:", y_pred[:10])
print("Actual stars values:", y_test[:10].values)
