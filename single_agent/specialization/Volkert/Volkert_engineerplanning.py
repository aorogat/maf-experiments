
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
df = pd.read_csv('volkert.csv')
print(df.head())

# Step 2: Handle missing data and drop non-numeric columns
df = df.dropna()  # Drop rows with missing values
df = df.select_dtypes(include=['number'])  # Keep only numeric columns

# Step 3: Prepare target and features
X = df.drop(columns=['class'])  # Features
y = df['class']  # Target

# Step 4: Encode categorical targets if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encode target variable

# Step 5: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 6: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Step 10: Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the first 10 predicted and actual values
print("\nFirst 10 Predicted vs Actual values:")
for i in range(10):
    print(f"Predicted: {label_encoder.inverse_transform([y_pred[i]])[0]}, Actual: {label_encoder.inverse_transform([y_test[i]])[0]}")
