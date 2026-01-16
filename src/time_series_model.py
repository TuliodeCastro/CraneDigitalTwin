import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class WindForecasterLSTM:
    def __init__(self, history_window=3):
        self.window = history_window
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_path = "models/lstm_wind_forecaster.keras"
        self.scaler_path = "models/scaler_wind.pkl"

    def create_sequences(self, data):
        Xs, ys = [], []
        for i in range(len(data) - self.window):
            Xs.append(data[i:(i + self.window)])
            ys.append(data[i + self.window]) 
        return np.array(Xs), np.array(ys)

    def train(self, df, target_col='Virtual_Wind'):
        print("⏳ [TIEMPO] Entrenando LSTM...")
        data = df[[target_col]].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = self.create_sequences(scaled_data)
        
        # Modelo LSTM
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.window, 1)),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='loss', patience=3)
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])
        
        # Guardar
        os.makedirs("models", exist_ok=True)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("✅ [TIEMPO] LSTM entrenado y guardado.")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict_horizon(self, recent_history, steps=2):
        """Predicción recursiva a futuro"""
        if self.model is None: self.load()
        
        preds = []
        curr_hist = list(recent_history)
        
        for _ in range(steps):
            # Preparar input
            input_arr = np.array(curr_hist[-self.window:]).reshape(-1, 1)
            scaled_input = self.scaler.transform(input_arr).reshape(1, self.window, 1)
            
            # Predecir
            scaled_pred = self.model.predict(scaled_input, verbose=0)
            pred = self.scaler.inverse_transform(scaled_pred)[0][0]
            
            # Lógica física: El viento no puede ser negativo
            pred = max(0.0, pred)
            
            preds.append(pred)
            curr_hist.append(pred)
            
        return preds