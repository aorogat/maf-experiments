
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('volkert.csv')

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(df.head())

# Handle missing data
df = df.dropna()  # Dropping rows with missing values

# Drop non-numeric columns and store the 'class' column as the target
X = df.drop(columns=['class']).select_dtypes(include=['number'])
y = df['class']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")

# Print the first 10 predicted and actual values
print("\nFirst 10 predicted and actual values:")
for i in range(10):
    print(f"Predicted: {label_encoder.inverse_transform([y_pred[i]])[0]}, Actual: {label_encoder.inverse_transform([y_test[i]])[0]}")
