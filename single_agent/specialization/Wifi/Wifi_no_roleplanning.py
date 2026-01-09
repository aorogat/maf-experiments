
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 1. Set the current working directory (assuming already set correctly)

# 2. Load the dataset
data = pd.read_csv('Wifi.csv')

# 3. Display the first few rows of the data
print(data.head())

# 4. Separate features and target
X = data.drop('TechCenter', axis=1)
y = data['TechCenter']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 5. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 6. Create a full pipeline with classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 7. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the model
pipeline.fit(X_train, y_train)

# 9. Make predictions
y_pred = pipeline.predict(X_test)

# 10. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# 11. Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 12. Print the first 10 predicted and actual TechCenter values
print("\nPredicted TechCenter Values:", y_pred[:10])
print("Actual TechCenter Values:", y_test.values[:10])
