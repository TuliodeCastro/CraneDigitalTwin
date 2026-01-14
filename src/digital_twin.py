# src/twin_physics.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class DigitalTwinPhysics:
    def __init__(self, lookup_file_path: str):
        """
        Inicializa el motor de física cargando la tabla de resultados de ANSYS.
        """
        self.sim_data = pd.read_csv(lookup_file_path)
        
        # Crear interpoladores: Convierten puntos discretos (10m/s, 12m/s) en funciones continuas
        self.stress_func = interp1d(
            self.sim_data['wind_speed_ref'], 
            self.sim_data['max_stress_mpa'], 
            kind='linear', 
            fill_value="extrapolate"
        )
        self.safety_func = interp1d(
            self.sim_data['wind_speed_ref'], 
            self.sim_data['safety_factor'], 
            kind='linear', 
            fill_value="extrapolate"
        )

    def calculate_virtual_wind(self, row):
        """
        Calcula el viento en la grúa usando las 3 estaciones (Promedio simple o ponderado).
        Usa las columnas reales de tu dataset.
        """
        w1 = row.get('Wind Speed (m/sec)_z1', 0)
        w2 = row.get('Wind Speed (m/sec)_z2', 0)
        w3 = row.get('Wind Speed (m/sec)_z3', 0)
        
        # Promedio simple (se puede mejorar con IDW si tienes coordenadas)
        return (w1 + w2 + w3) / 3.0

    def assess_structural_health(self, wind_speed: float):
        """
        Entrada: Velocidad de viento real (Virtual Sensor).
        Salida: Estrés estimado (MPa) y Factor de Seguridad (basado en ANSYS).
        """
        # Evitar valores negativos
        wind_speed = max(0.0, wind_speed)
        
        est_stress = float(self.stress_func(wind_speed))
        est_safety = float(self.safety_func(wind_speed))
        
        # Determinar estado
        if est_safety < 1.0:
            status = "CRITICAL" # Falla estructural
        elif est_safety < 1.5:
            status = "WARNING"  # Margen bajo
        else:
            status = "SAFE"     # Operación normal
            
        return est_stress, est_safety, status

    def run_simulation_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa todo el dataset histórico y añade columnas de física simulada.
        """
        print("⚙️  Calculando Viento Virtual (Centro de la obra)...")
        df['Virtual_Wind_Speed'] = df.apply(self.calculate_virtual_wind, axis=1)
        
        print("⚙️  Interpolando estrés estructural (Physics-Based)...")
        
        # Aplicamos la física fila por fila
        simulation_results = df['Virtual_Wind_Speed'].apply(
            lambda x: self.assess_structural_health(x)
        )
        
        # Desempaquetar resultados en nuevas columnas
        # Zip (*...) es un truco eficiente de Python para separar tuplas
        df['Sim_VonMises_Stress'], df['Sim_Safety_Factor'], df['Sim_Status'] = zip(*simulation_results)
        
        return df