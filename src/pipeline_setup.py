import os
import glob
import re
import numpy as np
import pandas as pd
import joblib  # Para guardar el modelo entrenado
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

class CFDPipeline:
    def __init__(self, raw_profiles_path: str, output_lookup_path: str, rupture_pressure: float = 355.0):
        self.profiles_path = raw_profiles_path
        self.output_csv = output_lookup_path
        self.rupture_pressure = rupture_pressure
        self.model = None

    def _read_prof_pressure(self, filepath):
        """
        Parsea el archivo .prof de ANSYS (Lógica de tu compañero).
        """
        pressures = []
        reading = False
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    # Detectar inicio de bloque de presión
                    if line.startswith("(pressure"):
                        reading = True
                        continue
                    
                    if reading:
                        # Detectar fin de bloque
                        if line.startswith(")"):
                            break
                        # Intentar leer número
                        try:
                            val = float(line)
                            pressures.append(val)
                        except ValueError:
                            continue # Saltar líneas que no sean números
            return np.array(pressures)
        except Exception as e:
            print(f"⚠️ Error leyendo {filepath}: {e}")
            return np.array([])

    def _compute_risk(self, pressure_array):
        """Calcula riesgo basado en presión máxima vs ruptura."""
        if len(pressure_array) == 0:
            return 0.0
        p_max = np.max(np.abs(pressure_array))
        # Clip para asegurar que el riesgo esté entre 0 y 1 (o más si excede ruptura)
        risk = p_max / self.rupture_pressure
        return risk, p_max

    def generate_lookup_table(self):
        """
        1. Busca archivos .prof
        2. Extrae ángulo y velocidad del nombre
        3. Calcula riesgo
        4. Guarda simulation_lookup.csv
        """
        print("⚙️  Iniciando procesamiento de archivos CFD (.prof)...")
        
        # Búsqueda recursiva en todas las carpetas dentro de profiles
        # Asume nombres como: "..../ang_45_vel_10.prof" o similar
        search_path = os.path.join(self.profiles_path, "**", "*.prof")
        files = glob.glob(search_path, recursive=True)
        
        if not files:
            raise FileNotFoundError(f"❌ No se encontraron archivos .prof en {self.profiles_path}")

        rows = []
        print(f"   Encontrados {len(files)} archivos de simulación.")

        for filepath in files:
            # Regex robusto para encontrar angulo y velocidad en el nombre del archivo
            # Busca patrones como "ang_45" y "vel_10" sin importar el orden
            match_ang = re.search(r"ang[a-z_]*(\d+)", filepath, re.IGNORECASE)
            match_vel = re.search(r"vel[a-z_]*(\d+)", filepath, re.IGNORECASE)

            if not match_ang or not match_vel:
                print(f"   ⚠️ Saltando archivo (nombre incorrecto): {os.path.basename(filepath)}")
                continue

            angle = float(match_ang.group(1))
            velocity = float(match_vel.group(1))

            # Procesar física
            pressures = self._read_prof_pressure(filepath)
            risk, p_max = self._compute_risk(pressures)

            rows.append({
                "angle": angle,
                "velocity": velocity,
                "max_pressure_pa": p_max,
                "risk_score": min(risk, 1.0), # Cap en 1.0 para el CSV
                "safety_factor": self.rupture_pressure / p_max if p_max > 0 else 999
            })

        if not rows:
            raise ValueError("❌ No se pudieron procesar datos válidos de los archivos .prof")

        # Crear DataFrame y guardar
        df_risk = pd.DataFrame(rows)
        df_risk.to_csv(self.output_csv, index=False)
        print(f"✅ Tabla de búsqueda generada: {self.output_csv} ({len(df_risk)} escenarios)")
        return df_risk

    def train_surrogate_model(self, df_risk):
        """
        Entrena un Random Forest para predecir el riesgo en ángulos/velocidades
        que NO fueron simulados (Interpolación inteligente).
        """
        print("🧠 Entrenando Modelo Sustituto (Random Forest)...")
        
        X = df_risk[["angle", "velocity"]]
        y = df_risk["risk_score"]

        # Configuración robusta del modelo
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X, y)
        
        # Validación rápida
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        
        print(f"   Entrenamiento completado. Métricas internas:")
        print(f"   R²: {r2:.4f} | RMSE: {rmse:.4f}")
        
        # Guardar modelo para uso posterior (opcional)
        joblib.dump(self.model, "models/surrogate_risk_model.pkl")
        print("💾 Modelo guardado en models/surrogate_risk_model.pkl")
        
        return self.model

    def predict_risk_batch(self, df_iot, wind_col='Wind Speed (m/sec)_z1', dir_col='Wind Direction (°)_z1'):
        """
        Toma el dataset de sensores IoT y predice el riesgo usando el modelo entrenado.
        """
        if self.model is None:
            raise Exception("❌ El modelo no ha sido entrenado. Ejecuta generate_lookup_table primero.")

        print("🔮 Prediciendo riesgo para dataset IoT...")
        
        # Preparar inputs (Mapeo de nombres)
        X_live = pd.DataFrame()
        X_live['angle'] = df_iot[dir_col]
        X_live['velocity'] = df_iot[wind_col]
        
        # Predecir
        predicted_risk = self.model.predict(X_live)
        
        # Clip para seguridad (0 a 1)
        return np.clip(predicted_risk, 0, 1)