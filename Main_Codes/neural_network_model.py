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
os.makedirs("logs/fit", exist_ok=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

df = pd.read_csv("day_approach_maskedID_timeseries.csv")  
df = df.drop(columns=["Athlete ID", "Date"])
df.fillna(df.median(), inplace=True)

X = df.drop(columns=["injury"])
y = df["injury"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)), 
    layers.Dense(128, activation="relu"),  
    layers.Dropout(0.3),  
    layers.Dense(64, activation="relu"), 
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

model.save("results/injury_prediction_model.h5")

with open("results/training_history.json", "w") as f:
    json.dump(history.history, f)

joblib.dump(scaler, "results/scaler.pkl")

with open("results/class_weights.json", "w") as f:
    json.dump(class_weight_dict, f)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))
print("Classification Report:\n", classification_report(y_test, y_pred))

class CustomTensorBoardCallback(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["roc_auc"] = roc_auc_score(y_test, self.model.predict(X_test))
        super().on_epoch_end(epoch, logs)

custom_tensorboard_callback = CustomTensorBoardCallback(log_dir=log_dir, histogram_freq=1)
