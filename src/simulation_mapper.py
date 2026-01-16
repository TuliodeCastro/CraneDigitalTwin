import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# --- TRUCO DE RUTAS ---
# Esto ayuda a que Python encuentre el archivo time_series_model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from time_series_model import WindForecasterLSTM

class DigitalTwinBackend:
    def __init__(self):
        # 1. Detectar rutas automáticamente
        self.base_dir = Path(__file__).parent.resolve()
        
        # Buscar modelo de riesgo (Random Forest - Física)
        path_local = self.base_dir / "models" / "surrogate_risk_model.pkl"
        path_parent = self.base_dir.parent / "models" / "surrogate_risk_model.pkl"
        
        if path_local.exists():
            self.risk_model_path = path_local
        elif path_parent.exists():
            self.risk_model_path = path_parent
        else:
            self.risk_model_path = "models/surrogate_risk_model.pkl" # Fallback

        self.risk_model = None
        
        # 2. Inicializar LSTM (Tiempo)
        self.lstm = WindForecasterLSTM(history_window=3)
        # Forzar rutas del LSTM relativas al modelo de riesgo
        self.lstm.model_path = str(self.risk_model_path.parent / "lstm_wind_forecaster.keras")
        self.lstm.scaler_path = str(self.risk_model_path.parent / "scaler_wind.pkl")

    def load_models(self):
        """Carga los modelos entrenados en memoria."""
        # A. Cargar Random Forest (Física)
        if os.path.exists(self.risk_model_path):
            self.risk_model = joblib.load(self.risk_model_path)
            print(f"✅ Backend: Modelo Físico cargado.")
        else:
            raise Exception(f"❌ Modelo de riesgo no encontrado en: {self.risk_model_path}")
        
        # B. Cargar LSTM (Tiempo)
        if not self.lstm.load():
            print("⚠️ Backend: LSTM no encontrado. Se usará predicción simple.")

    def get_digital_twin_status(self, current_wind, current_angle, recent_history):
        """
        Para el Dashboard en vivo (Live Operations).
        """
        # 1. Riesgo ACTUAL
        input_now = pd.DataFrame({'angle': [current_angle], 'velocity': [current_wind]})
        risk_now = self.risk_model.predict(input_now)[0]

        # 2. Predicción del FUTURO (t+10 min)
        future_winds = self.lstm.predict_horizon(recent_history, steps=2)
        wind_t10 = future_winds[-1] 

        # 3. Riesgo FUTURO
        input_future = pd.DataFrame({'angle': [current_angle], 'velocity': [wind_t10]})
        risk_future = self.risk_model.predict(input_future)[0]

        # 4. Determinar Estado
        status = "SAFE"
        if risk_now > 0.8: status = "CRITICAL"
        elif risk_future > 0.8: status = "WARNING_PREDICTED"
        elif risk_now > 0.5: status = "WARNING"

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
        [ESTA ES LA FUNCIÓN QUE FALTABA]
        Genera una tabla de predicción detallada.
        """
        # 1. Obtener vientos futuros con el LSTM
        future_winds = self.lstm.predict_horizon(recent_history, steps=steps)
        
        rows = []
        for wind in future_winds:
            # 2. Para cada viento futuro, preguntarle al modelo físico el riesgo
            # Asumimos ángulo 0 (peor caso frontal) para el pronóstico general
            input_df = pd.DataFrame({'angle': [0], 'velocity': [wind]})
            predicted_risk = self.risk_model.predict(input_df)[0]
            
            # 3. Determinar etiqueta
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
        Genera datos para la curva de la muerte (Stress Test).
        """
        sim_data = []
        for w in wind_range:
            input_df = pd.DataFrame({'angle': [angle_fixed], 'velocity': [w]})
            risk = self.risk_model.predict(input_df)[0]
            sim_data.append(risk)
        return sim_data