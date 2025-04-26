import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('DATASET-balanced.csv')

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Separate features and labels
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler for future use
joblib.dump(model, 'audio_deepfake_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to predict new audio samples
def predict_audio_deepfake(audio_features):
    """
    Predict whether an audio sample is fake or real.
    
    Parameters:
    audio_features (array-like): Audio features in the same format as the training data
    
    Returns:
    str: Prediction ('FAKE' or 'REAL')
    float: Probability of being fake
    """
    # Load the model and scaler
    model = joblib.load('audio_deepfake_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Preprocess the features
    features_scaled = scaler.transform([audio_features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][0]  # Probability of being fake
    
    return prediction, proba

# Example usage (with dummy data - you would replace with actual extracted features)
example_features = X.iloc[0].values  # Using first row as example
prediction, probability = predict_audio_deepfake(example_features)
print(f"\nExample Prediction: {prediction} with {probability:.2f} probability")