import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

class DataLoader:
    def __init__(self, raw_data_path: str):
        """
        Initializes the DataLoader with the base directory for raw data.
        """
        self.base_path = Path(raw_data_path)

    def load_iot_dataset(self, filename: str) -> pd.DataFrame:
        """
        Loads and cleans the IoT sensor dataset.
        
        Args:
            filename (str): The name of the CSV file to load.
            
        Returns:
            pd.DataFrame: A cleaned DataFrame with datetime indexing and interpolated values.
        """
        file_path = self.base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Loading sensor data from {filename}...")
        df = pd.read_csv(file_path)

        # 1. Convert 'Date' column to datetime objects for time-series handling
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.sort_values('Date').reset_index(drop=True)
        
        # 2. Handle missing values (Robustness)
        # Fill missing numeric values using linear interpolation followed by backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Note: Updated to avoid FutureWarning from .fillna(method='bfill')
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill()

        return df

    def parse_ansys_profile(self, filename: str) -> pd.DataFrame:
        """
        Parses an ANSYS .prof file to extract Height vs. Velocity tables.
        Assumes standard Fluent Profile format: ((z 1 2 3) (v 5 6 7)).
        
        Args:
            filename (str): The name of the profile file.
            
        Returns:
            pd.DataFrame: DataFrame containing 'height_m' and 'wind_speed_ms'.
                          Returns empty DataFrame on failure.
        """
        file_path = self.base_path / 'profiles' / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Profile not found: {file_path}")

        with open(file_path, 'r') as f:
            content = f.read()

        # Robust Logic: Use Regex to find data lists enclosed in parentheses
        # Looks for patterns like (z 0.0 10.0 20.0)
        try:
            # Extract Z coordinates (height)
            z_match = re.search(r'\(z\s+([\d\.\s\w\+\-]+)\)', content)
            # Extract velocity (usually labeled as u, v, or magnitude)
            v_match = re.search(r'\((?:velocity|u|magnitude)\s+([\d\.\s\w\+\-]+)\)', content)

            if not z_match or not v_match:
                raise ValueError("ANSYS .prof format not recognized or missing data.")

            # Convert space-separated strings to lists of floats
            z_values = [float(x) for x in z_match.group(1).split()]
            v_values = [float(x) for x in v_match.group(1).split()]

            return pd.DataFrame({'height_m': z_values, 'wind_speed_ms': v_values})

        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return pd.DataFrame()