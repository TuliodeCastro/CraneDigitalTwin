import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

class DataLoader:
    def __init__(self, raw_data_path: str):
        self.base_path = Path(raw_data_path)

    def load_sensor_data(self, filename: str) -> pd.DataFrame:
        """
        Carga y limpia el dataset de sensores IoT.
        """
        file_path = self.base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        print(f"📊 Cargando datos de sensores desde {filename}...")
        df = pd.read_csv(file_path)

        # 1. Convertir fecha a datetime para manejo de series temporales
        # Ajustamos el nombre de la columna según tu CSV
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.sort_values('Date').reset_index(drop=True)
        
        # 2. Manejo básico de nulos (Robustez)
        # Rellenar valores faltantes numéricos con interpolación lineal (común en series de tiempo)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear').fillna(method='bfill')

        return df

    def parse_ansys_profile(self, filename: str) -> pd.DataFrame:
        """
        Parsea un archivo .prof de ANSYS para extraer la tabla de Altura vs Velocidad.
        Asume formato estándar de Fluent Profile ((z 1 2 3) (v 5 6 7)).
        """
        file_path = self.base_path / 'profiles' / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el perfil: {file_path}")

        with open(file_path, 'r') as f:
            content = f.read()

        # Lógica Robusta: Usar Regex para encontrar listas de datos entre paréntesis
        # Buscamos patrones como (z 0.0 10.0 20.0)
        try:
            # Extraer coordenadas Z (altura)
            z_match = re.search(r'\(z\s+([\d\.\s\w\+\-]+)\)', content)
            # Extraer velocidad (usualmente u, v, o magnitude)
            v_match = re.search(r'\((?:velocity|u|magnitude)\s+([\d\.\s\w\+\-]+)\)', content)

            if not z_match or not v_match:
                raise ValueError("Formato .prof no reconocido o faltan datos.")

            # Convertir string a lista de floats
            z_values = [float(x) for x in z_match.group(1).split()]
            v_values = [float(x) for x in v_match.group(1).split()]

            return pd.DataFrame({'height_m': z_values, 'wind_speed_ms': v_values})

        except Exception as e:
            print(f"⚠️ Error parseando {filename}: {e}")
            return pd.DataFrame() # Retorna vacío en caso de error para no romper el programa