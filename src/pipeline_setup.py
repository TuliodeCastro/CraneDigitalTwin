import os
import glob
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class CFDPipeline:
    def __init__(self, raw_profiles_path, output_lookup_path, rupture_pressure=355.0):
        self.profiles_path = raw_profiles_path
        self.output_csv = output_lookup_path
        self.rupture_pressure = rupture_pressure
        self.model = None

    def _read_prof_pressure(self, filepath):
        pressures = []
        reading = False
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("(pressure"):
                        reading = True
                        continue
                    if reading:
                        if line.startswith(")"): break
                        try:
                            val = float(line)
                            pressures.append(val)
                        except ValueError: continue
            return np.array(pressures)
        except Exception:
            return np.array([])

    def generate_lookup_table(self):
        print("⚙️  [FÍSICA] Procesando perfiles CFD de ANSYS...")
        search_path = os.path.join(self.profiles_path, "**", "*.prof")
        files = glob.glob(search_path, recursive=True)
        
        rows = []
        for filepath in files:
            # Busca patrones como 'ang_45' y 'vel_10'
            match_ang = re.search(r"ang[a-z_]*(\d+)", filepath, re.IGNORECASE)
            match_vel = re.search(r"vel[a-z_]*(\d+)", filepath, re.IGNORECASE)

            if match_ang and match_vel:
                angle = float(match_ang.group(1))
                velocity = float(match_vel.group(1))
                pressures = self._read_prof_pressure(filepath)
                
                if len(pressures) > 0:
                    p_max = np.max(np.abs(pressures))
                    risk = min(p_max / self.rupture_pressure, 1.0)
                    rows.append({"angle": angle, "velocity": velocity, "risk_score": risk})

        df = pd.DataFrame(rows)
        df.to_csv(self.output_csv, index=False)
        print(f"✅ [FÍSICA] Lookup Table generada: {len(df)} escenarios.")
        return df

    def train_surrogate_model(self, df_risk):
        print("🧠 [FÍSICA] Entrenando Modelo Sustituto (Random Forest)...")
        X = df_risk[["angle", "velocity"]]
        y = df_risk["risk_score"]

        self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(X, y)
        
        # Guardar
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/surrogate_risk_model.pkl")
        print(f"💾 [FÍSICA] Modelo guardado (R2: {r2_score(y, self.model.predict(X)):.4f})")