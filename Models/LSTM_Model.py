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

os.makedirs("results", exist_ok=True)
os.makedirs("logs/fit/LSTM", exist_ok=True)

log_dir = "logs/fit/LSTM/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

df = pd.read_csv("day_approach_maskedID_timeseries.csv")
df = df.drop(columns=["Athlete ID", "Date"])
df.fillna(df.median(), inplace=True)

n_timesteps = 7  
n_features = int(df.shape[1] / n_timesteps)  

X_raw = df.drop(columns=["injury"]).values
y = df["injury"].values

X = X_raw.reshape(-1, n_timesteps, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

model = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)),  
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

model.save("results/LSTM/lstm_injury_prediction_model.h5")
joblib.dump(scaler, "results/LSTM/scaler.pkl")

with open("results/LSTM/training_history.json", "w") as f:
    json.dump(history.history, f)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))
print("Classification Report:\n", classification_report(y_test, y_pred))