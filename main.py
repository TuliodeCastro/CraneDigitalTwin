import os
import sys
import pandas as pd
import numpy as np
import subprocess  # Para ejecutar comandos de terminal

# --- CONFIGURACIÓN DE RUTAS ---
# Definimos las rutas base relativas a ESTE script (main.py)
# Asumimos que main.py está en la raíz del proyecto (ICRERA/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Añadimos 'src' al path de Python para poder importar módulos propios
sys.path.append(SRC_DIR)

# Importaciones de nuestros módulos (ahora Python los encontrará bien)
try:
    from src.data_loader import DataLoader
    from src.pipeline_setup import CFDPipeline
    from src.time_series_model import WindForecasterLSTM
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("Asegúrate de que 'src' tenga un archivo __init__.py o que las rutas sean correctas.")
    sys.exit(1)

def run_pipeline():
    """
    Ejecuta el pipeline completo del Gemelo Digital:
    1. Limpieza de Datos IoT
    2. Entrenamiento del Modelo Físico (CFD Surrogate)
    3. Entrenamiento del Pronóstico Temporal (LSTM)
    4. Lanzamiento Automático del Dashboard
    """
    print("🚀 Starting Digital Twin End-to-End Pipeline...")
    print("-" * 60)
    
    # ---------------------------------------------------------
    # 1. DEFINICIÓN DE DIRECTORIOS
    # ---------------------------------------------------------
    data_dir = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    figures_dir = os.path.join(BASE_DIR, "figures")
    processed_dir = os.path.join(data_dir, "processed")

    # Archivos específicos
    raw_iot_file = "processed/crane_digital_twin_ml_dataset.csv"
    processed_iot_file = os.path.join(processed_dir, "iot_processed.csv")
    dashboard_script = os.path.join(SRC_DIR, "dashboard.py")
    
    # Crear carpetas si no existen
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. LIMPIEZA DE DATOS (DATA FOUNDATION)
    # ---------------------------------------------------------
    print("\n[STEP 1/5] Data Foundation: Cleaning IoT Data...")
    loader = DataLoader(data_dir)
    
    try:
        raw_path = os.path.join(data_dir, raw_iot_file)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data not found at: {raw_path}")

        df_raw = pd.read_csv(raw_path)
        
        # Estandarizar Fechas
        if 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], utc=True)
            df_raw = df_raw.sort_values('Date').reset_index(drop=True)
        
        # Rellenar nulos
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        df_raw[numeric_cols] = df_raw[numeric_cols].interpolate(method='linear').bfill()
        
        # Calcular Viento Virtual
        wind_cols = [c for c in df_raw.columns if 'Wind Speed' in c]
        if wind_cols:
            df_raw['Virtual_Wind'] = df_raw[wind_cols].mean(axis=1)
            print(f"   [INFO] Virtual Wind calculated using {len(wind_cols)} sensors.")
        else:
            df_raw['Virtual_Wind'] = 0.0 # Fallback
            print("   [WARNING] No wind columns found. Virtual_Wind set to 0.")

        # Guardar dataset procesado
        df_raw.to_csv(processed_iot_file, index=False)
        print(f"   [SUCCESS] Processed dataset saved to: {processed_iot_file}")
        
    except Exception as e:
        print(f"   [ERROR] Data preparation failed: {e}")
        return

    # ---------------------------------------------------------
    # 3. MODELADO FÍSICO (CFD -> SURROGATE)
    # ---------------------------------------------------------
    print("\n[STEP 2/5] Physics Modeling: Training Structural Risk Surrogate...")
    try:
        profiles_path = os.path.join(data_dir, "profiles")
        lookup_path = os.path.join(data_dir, "simulation_lookup.csv")

        # Solo ejecutamos si existe la carpeta profiles
        if os.path.exists(profiles_path) and os.listdir(profiles_path):
            cfd_pipeline = CFDPipeline(
                raw_profiles_path=profiles_path,
                output_lookup_path=lookup_path
            )
            df_sim = cfd_pipeline.generate_lookup_table()
            
            if not df_sim.empty:
                cfd_pipeline.train_surrogate_model(df_sim)
                print("   [SUCCESS] Physics Surrogate Model (Random Forest) trained.")
            else:
                print("   [WARNING] CFD data empty. Skipping training.")
        else:
            print("   [INFO] No 'profiles' folder found. Skipping Physics Training (using pre-trained if avail).")
            
    except Exception as e:
        print(f"   [ERROR] Physics modeling failed: {e}")

    # ---------------------------------------------------------
    # 4. MODELADO TEMPORAL (LSTM FORECASTING)
    # ---------------------------------------------------------
    print("\n[STEP 3/5] Time-Series Modeling: Training Wind Forecaster (LSTM)...")
    try:
        df_iot = pd.read_csv(processed_iot_file)
        
        lstm = WindForecasterLSTM(history_window=3)
        # Forzar guardado en la carpeta correcta
        lstm.model_path = os.path.join(models_dir, "lstm_wind_forecaster.keras")
        lstm.scaler_path = os.path.join(models_dir, "scaler_wind.pkl")
        
        lstm.train(df_iot, target_col='Virtual_Wind')
        
        print("   [SUCCESS] LSTM Model trained and saved.")
        
    except Exception as e:
        print(f"   [ERROR] Time-series training failed: {e}")

    # ---------------------------------------------------------
    # 5. LANZAMIENTO DEL DASHBOARD
    # ---------------------------------------------------------
    print("\n[STEP 5/5] Launching Digital Twin Interface...")
    print("-" * 60)
    print(f"Target Script: {dashboard_script}")
    print("Press Ctrl+C to stop the server.")
    print("-" * 60)

    try:
        # --- CORRECCIÓN CLAVE AQUÍ ---
        # Usamos sys.executable para llamar al mismo Python que está corriendo este script
        # y usamos "-m streamlit" para asegurarnos de que encuentre el módulo.
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_script], check=True)
        
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch Streamlit: {e}")

if __name__ == "__main__":
    run_pipeline()