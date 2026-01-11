
# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('volkert.csv')

# Handle missing data and drop non-numeric columns
df.dropna(inplace=True)  # Dropping rows with missing values
df = df.select_dtypes(include=['int64', 'float64'])  # Keep only numeric columns

# Define features and target
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)  # Apply scaling to numeric columns
    ])

# Create the classification pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(report)

# Print predicted and actual values
predicted_actual = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(predicted_actual.head(10))
