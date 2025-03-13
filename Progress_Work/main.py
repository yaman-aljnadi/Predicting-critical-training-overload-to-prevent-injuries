import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("Y:/EnergyEye_Github/Predicting-critical-training-overload-to-prevent-injuries/Progress_Work/day_approach_maskedID_timeseries.csv")

print(df.head())

features = [
    "nr. sessions", "total km", "km Z3-4", "km Z5-T1-T2", "km sprinting",
    "strength training", "hours alternative", "perceived exertion",
    "perceived trainingSuccess", "perceived recovery"
]
label = "injury"  # Target variable (1 = injury, 0 = no injury)

# Handling multiple weeks of data: Take the most recent week's data
df = df.groupby("Athlete ID").last().reset_index()

# Select features and target variable
X = df[features]
y = df[label]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Function to predict injury risk based on user input
def predict_risk():
    print("\nEnter the following details:")
    user_data = []
    
    for feature in features:
        value = float(input(f"{feature}: "))
        user_data.append(value)
    
    input_scaled = scaler.transform([user_data])
    prediction = model.predict(input_scaled)
    print("\nPrediction:", "At Risk" if prediction[0] == 1 else "Not At Risk")

# Run the prediction function
predict_risk()