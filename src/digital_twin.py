import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Add the current directory to sys.path to ensure local modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Time Series model
try:
    from time_series_model import WindForecasterLSTM
except ImportError:
    # Fallback if running from a different directory depth
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from time_series_model import WindForecasterLSTM

class DigitalTwinBackend:
    def __init__(self):
        """
        Initializes the Digital Twin Backend.
        Sets up paths for models and initializes the LSTM architecture.
        """
        # 1. Path Detection
        self.base_dir = Path(__file__).resolve().parent
        self.models_dir = self.base_dir.parent / "models"
        
        # 2. Risk Model Path (Random Forest)
        self.risk_model_path = self.models_dir / "surrogate_risk_model.pkl"
        
        # 3. LSTM Model Paths (Time Series)
        self.lstm_model_path = self.models_dir / "lstm_wind_forecaster.keras"
        self.scaler_path = self.models_dir / "scaler_wind.pkl"

        self.risk_model = None
        
        # Initialize LSTM class (structure only, weights loaded later)
        self.lstm = WindForecasterLSTM(history_window=3)
        # Override internal paths of the LSTM instance to ensure consistency
        self.lstm.model_path = str(self.lstm_model_path)
        self.lstm.scaler_path = str(self.scaler_path)

    def load_models(self):
        """
        Loads the trained machine learning models from disk into memory.
        """
        # A. Load Physics Model (Random Forest)
        if self.risk_model_path.exists():
            self.risk_model = joblib.load(self.risk_model_path)
            print("Backend: Physics Risk Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Risk model not found at: {self.risk_model_path}")
        
        # B. Load Time-Series Model (LSTM)
        if not self.lstm.load():
            print("Warning: LSTM model could not be loaded. Predictions will be unavailable.")

    def get_digital_twin_status(self, current_wind, current_angle, recent_history):
        """
        Calculates the real-time status of the Digital Twin.
        
        Args:
            current_wind (float): Current wind speed (m/s).
            current_angle (float): Current wind angle (degrees).
            recent_history (array): Last 3 wind speed measurements for LSTM input.
            
        Returns:
            dict: Dictionary containing current risk, predicted wind, predicted risk, and status.
        """
        # 1. Calculate CURRENT Risk (Physics)
        input_now = pd.DataFrame({'angle': [current_angle], 'velocity': [current_wind]})
        risk_now = self.risk_model.predict(input_now)[0]

        # 2. Predict FUTURE Wind (Time Series - 10 mins ahead)
        # We predict 2 steps ahead (assuming 5-min intervals = 10 mins)
        if len(recent_history) >= 3:
            future_winds = self.lstm.predict_horizon(recent_history, steps=2)
            wind_t10 = future_winds[-1] 
        else:
            # Fallback if insufficient history
            wind_t10 = current_wind 

        # 3. Calculate FUTURE Risk (Physics)
        input_future = pd.DataFrame({'angle': [current_angle], 'velocity': [wind_t10]})
        risk_future = self.risk_model.predict(input_future)[0]

        # 4. Determine Traffic Light Status
        # Thresholds: <0.5 (Safe), 0.5-0.8 (Warning), >0.8 (Critical)
        status = "SAFE"
        if risk_now > 0.8: 
            status = "CRITICAL"
        elif risk_future > 0.8: 
            status = "WARNING_PREDICTED"
        elif risk_now > 0.5: 
            status = "WARNING"

        return {
            "current_wind": current_wind,
            "current_risk": risk_now,
            "future_wind_10m": wind_t10,
            "future_risk_10m": risk_future,
            "status": status
        }

    def get_forecast_data(self, recent_history, steps=12):
        """
        Generates a recursive forecast table for reporting.
        
        Args:
            recent_history (array): Input for LSTM.
            steps (int): Number of steps to predict (default 12 steps = 1 hour).
            
        Returns:
            pd.DataFrame: Table with predicted wind and associated risk.
        """
        # 1. Get wind predictions
        future_winds = self.lstm.predict_horizon(recent_history, steps=steps)
        
        rows = []
        for wind in future_winds:
            # 2. Map every future wind point to physical risk
            # We assume a worst-case angle (0 degrees) for general forecasting
            input_df = pd.DataFrame({'angle': [0], 'velocity': [wind]})
            predicted_risk = self.risk_model.predict(input_df)[0]
            
            status_label = "SAFE"
            if predicted_risk > 0.8: status_label = "CRITICAL"
            elif predicted_risk > 0.5: status_label = "WARNING"
            
            rows.append({
                "Predicted Wind (m/s)": wind,
                "Predicted Risk": predicted_risk,
                "Status": status_label
            })
            
        return pd.DataFrame(rows)

    def run_stress_test(self, wind_range, angle_fixed=0):
        """
        Simulates the 'Curve of Death' for the Stress Test Lab.
        
        Args:
            wind_range (array): Array of wind speeds to test.
            angle_fixed (float): The angle to fix for the simulation.
            
        Returns:
            list: Risk scores corresponding to the input winds.
        """
        sim_data = []
        # Batch prediction for performance
        input_df = pd.DataFrame({
            'angle': [angle_fixed] * len(wind_range),
            'velocity': wind_range
        })
        
        risks = self.risk_model.predict(input_df)
        return risks