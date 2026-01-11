
# Step 1: Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load the dataset 'Wifi.csv' into a DataFrame
df = pd.read_csv('Wifi.csv')

# Step 3: Verify the contents of the dataset
print(df.head())

# Splitting features and target variable
X = df.drop('TechCenter', axis=1)
y = df['TechCenter']

# Step 4: Preprocessing
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Create preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 5: Create a pipeline that first transforms the data and then fits a model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
pipeline.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = pipeline.predict(X_test)

# Step 9: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print first 10 predicted and actual TechCenter values
print("First 10 predicted values:", y_pred[:10])
print("First 10 actual values:", y_test.values[:10])
