
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv('Utility.csv')

# Inspect the dataset to identify categorical columns and assess data types
print(data.info())
print(data.describe())
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
print(f'Categorical columns: {categorical_cols}')

# Preprocess the categorical columns using One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keep all other columns
)

# Split the dataset into features (X) and target (y)
X = data.drop('CSRI', axis=1)
y = data['CSRI']

# Further split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the model using the training set
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model's performance by calculating the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Extract and display the first 10 predicted values of CSRI alongside the corresponding actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head(10))
