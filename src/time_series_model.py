import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class WindForecasterLSTM:
    def __init__(self, history_window=3):
        """
        Initializes the LSTM Forecaster.
        
        Args:
            history_window (int): Number of past time steps to use for predicting the next step.
        """
        self.window = history_window
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Directory setup
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.figures_dir = os.path.join(self.base_dir, "figures")
        
        self.model_path = os.path.join(self.models_dir, "lstm_wind_forecaster.keras")
        self.scaler_path = os.path.join(self.models_dir, "scaler_wind.pkl")
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def create_sequences(self, data):
        """
        Converts time-series data into input (X) and output (y) sequences for LSTM.
        """
        Xs, ys = [], []
        for i in range(len(data) - self.window):
            Xs.append(data[i:(i + self.window)])
            ys.append(data[i + self.window]) 
        return np.array(Xs), np.array(ys)

    def train(self, df, target_col='Virtual_Wind'):
        """
        Trains the LSTM model on the provided dataframe.
        Generates training history and validation plots.
        """
        print(f"[INFO] Preprocessing data for LSTM training...")
        
        # Data Preprocessing
        data = df[[target_col]].values
        scaled_data = self.scaler.fit_transform(data)
        X, y = self.create_sequences(scaled_data)
        
        # Define Architecture
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.window, 1), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print("[INFO] Starting training process (with validation split)...")
        history = self.model.fit(
            X, y, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.2, 
            verbose=1, 
            callbacks=[early_stop]
        )
        
        # Save artifacts
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("[SUCCESS] Model and scaler saved successfully.")
        
        # Generate analysis plots
        self._plot_training_history(history)
        self._plot_validation_results(X, y)

    def _plot_training_history(self, history):
        """
        Plots and saves the training vs validation loss curve.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Training History')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.figures_dir, "lstm_training_history.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Training history plot saved to: {save_path}")

    def _plot_validation_results(self, X, y):
        """
        Plots and saves a comparison between Real vs. Predicted wind speeds.
        """
        # Generate predictions
        y_pred_scaled = self.model.predict(X)
        
        # Inverse transform to get real units (m/s)
        y_test_inv = self.scaler.inverse_transform(y.reshape(-1, 1))
        y_pred_inv = self.scaler.inverse_transform(y_pred_scaled)
        
        # Plot subset for clarity (last 150 points)
        subset = 150
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv[-subset:], label='Real Wind Speed (m/s)', color='blue')
        plt.plot(y_pred_inv[-subset:], label='Predicted Wind Speed (m/s)', color='orange', linestyle='--')
        plt.title(f'LSTM Validation: Real vs Predicted (Last {subset} points)')
        plt.ylabel('Wind Velocity (m/s)')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.figures_dir, "lstm_forecast_validation.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Forecast validation plot saved to: {save_path}")

    def load(self):
        """
        Loads the trained model and scaler from disk.
        Returns:
            bool: True if successful, False otherwise.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict_horizon(self, recent_history, steps=2):
        """
        Performs recursive forecasting for N steps into the future.
        
        Args:
            recent_history (list): The last known data points.
            steps (int): How many steps ahead to predict.
            
        Returns:
            list: The predicted values.
        """
        if self.model is None: 
            self.load()
            
        preds = []
        curr_hist = list(recent_history)
        
        for _ in range(steps):
            # Prepare input
            input_arr = np.array(curr_hist[-self.window:]).reshape(-1, 1)
            scaled_input = self.scaler.transform(input_arr).reshape(1, self.window, 1)
            
            # Predict
            scaled_pred = self.model.predict(scaled_input, verbose=0)
            pred = self.scaler.inverse_transform(scaled_pred)[0][0]
            
            # Physical constraint: Wind speed cannot be negative
            pred = max(0.0, pred)
            
            preds.append(pred)
            curr_hist.append(pred)
            
        return preds

if __name__ == "__main__":
    # Unit Test / Debug execution
    print("[TEST] Running isolated test...")
    dummy_data = pd.DataFrame({'Virtual_Wind': np.sin(np.linspace(0, 20, 200)) + 10})
    lstm = WindForecasterLSTM(history_window=3)
    lstm.train(dummy_data)