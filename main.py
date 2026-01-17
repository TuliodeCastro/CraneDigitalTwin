import os
import sys
import pandas as pd
import numpy as np
import subprocess  # Required to execute terminal commands

# Ensure local modules can be found by adding the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from pipeline_setup import CFDPipeline
from time_series_model import WindForecasterLSTM

def run_pipeline():
    """
    Executes the complete Digital Twin configuration pipeline:
    1. Data Ingestion & Cleaning
    2. Physics Model Training (CFD Surrogate)
    3. Time-Series Forecasting Training (LSTM)
    4. Automatic Launch of the Live Dashboard
    """
    print("Starting Digital Twin End-to-End Pipeline...")
    print("-" * 60)
    
    # ---------------------------------------------------------
    # 1. ENVIRONMENT SETUP & PATH CONFIGURATION
    # ---------------------------------------------------------
    # Define base paths relative to this script
    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    figures_dir = os.path.join(base_dir, "figures")
    processed_dir = os.path.join(data_dir, "processed")

    # File names
    raw_iot_file = "crane_digital_twin_ml_dataset.csv"
    processed_iot_file = os.path.join(processed_dir, "iot_processed.csv")
    dashboard_script = os.path.join(src_dir, "dashboard.py")
    
    # Create necessary directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. DATA FOUNDATION: CLEANING & PREPARATION
    # ---------------------------------------------------------
    print("\n[STEP 1/5] Data Foundation: Cleaning IoT Data...")
    loader = DataLoader(data_dir)
    
    try:
        # Load raw dataset
        raw_path = os.path.join(data_dir, raw_iot_file)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data not found at: {raw_path}")

        df_raw = pd.read_csv(raw_path)
        
        # Standardize timestamps
        if 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], utc=True)
            df_raw = df_raw.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values using interpolation followed by backward fill
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        df_raw[numeric_cols] = df_raw[numeric_cols].interpolate(method='linear').bfill()
        
        # Feature Engineering: Calculate 'Virtual_Wind'
        wind_cols = [c for c in df_raw.columns if 'Wind Speed' in c]
        if wind_cols:
            df_raw['Virtual_Wind'] = df_raw[wind_cols].mean(axis=1)
            print(f"   [INFO] Virtual Wind calculated using columns: {len(wind_cols)} sensors.")
        else:
            raise ValueError("No wind speed columns found in dataset.")

        # Save the processed "Golden Dataset"
        df_raw.to_csv(processed_iot_file, index=False)
        print(f"   [SUCCESS] Processed dataset saved to: {processed_iot_file}")
        
    except Exception as e:
        print(f"   [ERROR] Data preparation failed: {e}")
        return

    # ---------------------------------------------------------
    # 3. PHYSICS MODELING (CFD -> SURROGATE)
    # ---------------------------------------------------------
    print("\n[STEP 2/5] Physics Modeling: Training Structural Risk Surrogate...")
    try:
        profiles_path = os.path.join(data_dir, "profiles")
        lookup_path = os.path.join(data_dir, "simulation_lookup.csv")

        cfd_pipeline = CFDPipeline(
            raw_profiles_path=profiles_path,
            output_lookup_path=lookup_path
        )
        
        # Parse ANSYS .prof files to generate the lookup table
        df_sim = cfd_pipeline.generate_lookup_table()
        
        if not df_sim.empty:
            # Train the Random Forest Regressor
            cfd_pipeline.train_surrogate_model(df_sim)
            print("   [SUCCESS] Physics Surrogate Model (Random Forest) trained and saved.")
        else:
            print("   [WARNING] No valid CFD profiles found. Physics training skipped.")
            
    except Exception as e:
        print(f"   [ERROR] Physics modeling failed: {e}")

    # ---------------------------------------------------------
    # 4. TIME-SERIES MODELING (LSTM FORECASTING)
    # ---------------------------------------------------------
    print("\n[STEP 3/5] Time-Series Modeling: Training Wind Forecaster (LSTM)...")
    try:
        # Load the clean data generated in Step 1
        df_iot = pd.read_csv(processed_iot_file)
        
        # Initialize and train LSTM
        lstm = WindForecasterLSTM(history_window=3)
        lstm.train(df_iot, target_col='Virtual_Wind')
        
        print("   [SUCCESS] LSTM Model trained and saved.")
        print(f"   [INFO] Validation plots saved to: {figures_dir}")
        
    except Exception as e:
        print(f"   [ERROR] Time-series training failed: {e}")

    # ---------------------------------------------------------
    # 5. PIPELINE FINALIZATION
    # ---------------------------------------------------------
    print("\n[STEP 4/5] Pipeline Finalized.")
    print("   1. Data processed.")
    print("   2. Physics Model ready.")
    print("   3. AI Forecast Model ready.")

    # ---------------------------------------------------------
    # 6. LAUNCH DIGITAL TWIN DASHBOARD
    # ---------------------------------------------------------
    print("\n[STEP 5/5] Launching Digital Twin Interface...")
    print("-" * 60)
    print(f"Executing: streamlit run {dashboard_script}")
    print("Press Ctrl+C to stop the server.")
    print("-" * 60)

    try:
        # This command launches Streamlit and blocks the terminal until the user quits
        subprocess.run(["streamlit", "run", dashboard_script], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch Streamlit: {e}")

if __name__ == "__main__":
    run_pipeline()