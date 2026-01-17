import os
import glob
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

class CFDPipeline:
    def __init__(self, raw_profiles_path: str, output_lookup_path: str, rupture_pressure: float = 355.0):
        """
        Initializes the physics pipeline.
        :param raw_profiles_path: Folder containing ANSYS .prof files.
        :param output_lookup_path: Path to save the summary CSV.
        :param rupture_pressure: Pressure (Pa) where the structure fails (ISO 4302).
        """
        self.profiles_path = Path(raw_profiles_path)
        self.output_csv = Path(output_lookup_path)
        self.rupture_pressure = rupture_pressure
        self.model = None
        
        # Path where trained models will be saved
        # Assumes a 'models' folder at the same level as 'src' or within the project root
        self.models_dir = self.output_csv.parent.parent / "models"
        
        # Ensure output directories exist
        os.makedirs(self.output_csv.parent, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def _read_prof_pressure(self, filepath):
        """
        Parses an ANSYS Fluent .prof file to extract the pressure list.
        Looks for the block: (pressure ... data ...)
        """
        pressures = []
        reading = False
        try:
            with open(filepath, "r") as f:
                content = f.read()
                
            # Robust method using block partitioning by parentheses
            # We look for something starting with (pressure and ending with )
            # Note: .prof files can be complex, this is a simple sequential read
            lines = content.split('\n')
            for line in lines:
                clean_line = line.strip()
                
                # Detect start of pressure block
                if clean_line.startswith("(pressure"):
                    reading = True
                    continue
                
                if reading:
                    # Detect end of block (a closing parenthesis alone or at the end)
                    if clean_line.startswith(")"):
                        reading = False
                        break
                    
                    # Extract numbers from the line
                    # Lines may contain multiple numbers: "1.23 4.56 7.89"
                    tokens = clean_line.split()
                    for token in tokens:
                        try:
                            # Ignore parentheses if they remain attached
                            val = float(token.replace(')', ''))
                            pressures.append(val)
                        except ValueError:
                            continue

            return np.array(pressures)
            
        except Exception as e:
            print(f"⚠️ Error reading file {filepath}: {e}")
            return np.array([])

    def generate_lookup_table(self):
        """
        Iterates through the profile folder, processes physics, and generates the CSV.
        """
        print(f"⚙️  [PHYSICS] Searching for CFD profiles in: {self.profiles_path}")
        
        # Recursive search for .prof files
        search_pattern = str(self.profiles_path / "**" / "*.prof")
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            raise FileNotFoundError(f"❌ No .prof files found in {self.profiles_path}")

        rows = []
        print(f"   Found {len(files)} simulation files. Processing...")

        for filepath in files:
            filename = os.path.basename(filepath)
            
            # Robust regex to extract Angle and Velocity from filename
            # Examples: "ang_45_vel_10.prof", "v10_a0.prof", etc.
            # Looks for digits after 'ang' or 'a', and after 'vel' or 'v'
            match_ang = re.search(r"(?:ang|a)[a-z_]*(\d+)", filename, re.IGNORECASE)
            match_vel = re.search(r"(?:vel|v)[a-z_]*(\d+)", filename, re.IGNORECASE)

            if match_ang and match_vel:
                angle = float(match_ang.group(1))
                velocity = float(match_vel.group(1))
                
                # Read pressures from file
                pressures = self._read_prof_pressure(filepath)
                
                if len(pressures) > 0:
                    # PURE PHYSICS:
                    # 1. Get absolute max pressure (can be negative suction or positive pressure)
                    p_max = np.max(np.abs(pressures))
                    
                    # 2. Calculate Risk (0.0 to 1.0, or >1.0 if failed)
                    risk_score = p_max / self.rupture_pressure
                    
                    # 3. Safety factor (Inverse of risk)
                    safety_factor = self.rupture_pressure / p_max if p_max > 0 else 999.9

                    rows.append({
                        "angle": angle,
                        "velocity": velocity,
                        "max_pressure_pa": round(p_max, 2),
                        "risk_score": round(risk_score, 4), # Not clipped here so the model learns real trends
                        "safety_factor": round(safety_factor, 2)
                    })
            else:
                # Optional: Warn if there are files with unrecognized naming conventions
                # print(f"   ⚠️ Skipping {filename}: unrecognized naming format.")
                pass

        if not rows:
            raise ValueError("❌ Could not extract valid data. Check your .prof filenames.")

        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save CSV
        df.to_csv(self.output_csv, index=False)
        print(f"✅ [PHYSICS] Lookup Table generated: {self.output_csv} ({len(df)} scenarios)")
        return df

    def train_surrogate_model(self, df_risk):
        """
        Trains the Random Forest to act as a surrogate for ANSYS.
        """
        print("🧠 [AI PHYSICS] Training Surrogate Model (Random Forest)...")
        
        # Features (X) and Target (y)
        X = df_risk[["angle", "velocity"]]
        y = df_risk["risk_score"]

        # Model Configuration
        # n_estimators=200: Enough trees to smooth the prediction
        self.model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        self.model.fit(X, y)
        
        # Quick Evaluation
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        print(f"   Training completed.")
        print(f"   Accuracy (R²): {r2:.4f} (Ideal close to 1.0)")
        print(f"   Mean Absolute Error (MAE): {mae:.4f}")

        # Save Model
        model_path = self.models_dir / "surrogate_risk_model.pkl"
        joblib.dump(self.model, model_path)
        print(f"💾 [MODEL SAVED] {model_path}")
        
        return self.model

# --- DIRECT EXECUTION BLOCK (OPTIONAL FOR TESTING) ---
if __name__ == "__main__":
    # Test configuration if running this file standalone
    BASE_DIR = Path(os.getcwd())
    if "src" in str(BASE_DIR):
        BASE_DIR = BASE_DIR.parent
        
    RAW_PROFILES = BASE_DIR / "data" / "profiles"
    OUTPUT_CSV = BASE_DIR / "data" / "simulation_lookup.csv"
    
    pipeline = CFDPipeline(str(RAW_PROFILES), str(OUTPUT_CSV))
    try:
        df = pipeline.generate_lookup_table()
        pipeline.train_surrogate_model(df)
    except Exception as e:
        print(f"Error: {e}")