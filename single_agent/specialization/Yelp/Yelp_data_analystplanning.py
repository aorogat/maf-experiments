
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('Yelp_Merged.csv')

# Display the first few rows of the dataset
print(data.head())

# Handle missing data by dropping rows with any missing values
data = data.dropna()

# Drop non-numeric columns
data = data.drop(columns=['business_id', 'user_id', 'review_date'])

# Split features and target variable
X = data.drop(columns=['stars'])
y = data['stars']

# Encode categorical features using LabelEncoder
X = X.apply(LabelEncoder().fit_transform)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Print the first 10 predicted and actual stars values
print("First 10 predicted vs actual stars values:")
print(pd.DataFrame({'Predicted': y_pred[:10], 'Actual': y_test[:10].values}))
