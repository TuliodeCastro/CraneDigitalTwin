import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Add the current directory to sys.path to ensure local modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from time_series_model import WindForecasterLSTM

class DigitalTwinBackend:
    def __init__(self):
        """
        Initializes the Digital Twin Backend.
        Sets up paths for models and initializes the LSTM architecture.
        """
        # 1. Path Detection
        self.base_dir = Path(__file__).parent.resolve()
        
        # Locate the Risk Model (Random Forest - Physics)
        # Checks current directory and parent directory for robustness
        path_local = self.base_dir / "models" / "surrogate_risk_model.pkl"
        path_parent = self.base_dir.parent / "models" / "surrogate_risk_model.pkl"
        
        if path_local.exists():
            self.risk_model_path = path_local
        elif path_parent.exists():
            self.risk_model_path = path_parent
        else:
            # Fallback path
            self.risk_model_path = Path("models/surrogate_risk_model.pkl")

        self.risk_model = None
        
        # 2. Initialize LSTM (Time Series)
        self.lstm = WindForecasterLSTM(history_window=3)
        
        # Force LSTM paths relative to the found risk model path to ensure consistency
        self.lstm.model_path = str(self.risk_model_path.parent / "lstm_wind_forecaster.keras")
        self.lstm.scaler_path = str(self.risk_model_path.parent / "scaler_wind.pkl")

    def load_models(self):
        """
        Loads the trained machine learning models from disk into memory.
        """
        # A. Load Physics Model (Random Forest)
        if os.path.exists(self.risk_model_path):
            self.risk_model = joblib.load(self.risk_model_path)
            print(f"Backend: Physics Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Risk model not found at: {self.risk_model_path}")
        
        # B. Load Time-Series Model (LSTM)
        if not self.lstm.load():
            print("Warning: LSTM model not found. Predictions will default to simple heuristics.")

    def get_digital_twin_status(self, current_wind, current_angle, recent_history):
        """
        Calculates the real-time status of the Digital Twin for the Dashboard.
        
        Args:
            current_wind (float): Current wind speed in m/s.
            current_angle (float): Current wind angle in degrees.
            recent_history (list/array): List of the last 3 wind speed measurements.
            
        Returns:
            dict: A dictionary containing current and predicted metrics and status.
        """
        # 1. Calculate CURRENT Risk (Physics)
        input_now = pd.DataFrame({'angle': [current_angle], 'velocity': [current_wind]})
        risk_now = self.risk_model.predict(input_now)[0]

        # 2. Predict FUTURE Wind (Time Series - 10 mins ahead)
        # We predict 2 steps ahead (assuming 5-min intervals)
        future_winds = self.lstm.predict_horizon(recent_history, steps=2)
        wind_t10 = future_winds[-1] 

        # 3. Calculate FUTURE Risk (Physics)
        input_future = pd.DataFrame({'angle': [current_angle], 'velocity': [wind_t10]})
        risk_future = self.risk_model.predict(input_future)[0]

        # 4. Determine System Status
        status = "SAFE"
        if risk_now > 0.8:
            status = "CRITICAL"
        elif risk_future > 0.8:
            status = "WARNING_PREDICTED"
        elif risk_now > 0.5:
            status = "WARNING"

        return {
            "current_wind": round(current_wind, 2),
            "current_risk": round(risk_now, 4),
            "future_wind_10m": round(wind_t10, 2),
            "future_risk_10m": round(risk_future, 4),
            "status": status,
            "forecast_series": future_winds
        }

    def get_forecast_data(self, recent_history, steps=12):
        """
        Generates a detailed forecast table for the next hour (default 12 steps).
        
        Args:
            recent_history (list/array): Input data for the LSTM.
            steps (int): Number of future steps to predict.
            
        Returns:
            pd.DataFrame: Dataframe with predicted wind speeds and associated risks.
        """
        # 1. Obtain future wind predictions via LSTM
        future_winds = self.lstm.predict_horizon(recent_history, steps=steps)
        
        rows = []
        for wind in future_winds:
            # 2. Query the Physics Model for risk
            # We assume a 0-degree angle (worst-case frontal wind) for general forecasting
            input_df = pd.DataFrame({'angle': [0], 'velocity': [wind]})
            predicted_risk = self.risk_model.predict(input_df)[0]
            
            # 3. Determine label
            status_label = "SAFE"
            if predicted_risk > 0.8:
                status_label = "CRITICAL"
            elif predicted_risk > 0.5:
                status_label = "WARNING"
            
            rows.append({
                "Predicted Wind (m/s)": wind,
                "Predicted Risk": predicted_risk,
                "Status": status_label
            })
            
        return pd.DataFrame(rows)

    def run_stress_test(self, wind_range, angle_fixed=0):
        """
        Generates data for the structural vulnerability curve (Stress Test).
        
        Args:
            wind_range (list/array): List of wind speeds to simulate.
            angle_fixed (float): The fixed angle of incidence for the simulation.
            
        Returns:
            list: Calculated risk scores for each wind speed in the range.
        """
        sim_data = []
        
        # Batch prediction for performance optimization
        input_df = pd.DataFrame({
            'angle': [angle_fixed] * len(wind_range),
            'velocity': wind_range
        })
        
        sim_data = self.risk_model.predict(input_df)
        
        return list(sim_data)