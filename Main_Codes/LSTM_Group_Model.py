import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os
import datetime
import json
import joblib

os.makedirs("results/LSTM_Group", exist_ok=True)
os.makedirs("logs/fit/LSTM_Group", exist_ok=True)

log_dir = "logs/fit/LSTM_Group/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

df = pd.read_csv("day_approach_maskedID_timeseries.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by=["Athlete ID", "Date"])

n_timesteps = 7  
feature_cols = []

for col in df.columns:
    if col not in ["Athlete ID", "Date", "injury"]:
        feature_cols.append(col)

X, y = [], []

for _, group in df.groupby("Athlete ID"):
    group_features = group[feature_cols].values
    group_labels = group["injury"].values

    for i in range(len(group) - n_timesteps):
        X.append(group_features[i : i + n_timesteps])
        y.append(group_labels[i + n_timesteps])  

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X.shape[-1])).reshape(X_test.shape)

class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

model = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(n_timesteps, X.shape[-1])),
    layers.Dropout(0.3),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[tensorboard_callback],
    verbose=1
)

model.save("results/LSTM_Group/lstm_injury_prediction_model.h5")
joblib.dump(scaler, "results/LSTM/scaler.pkl")

with open("results/LSTM_Group/training_history.json", "w") as f:
    json.dump(history.history, f)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))
print("Classification Report:\n", classification_report(y_test, y_pred))
