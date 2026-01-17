import os
import pandas as pd
from pipeline_setup import CFDPipeline
from time_series_model import WindForecasterLSTM
from data_loader import DataLoader

def main():
    """
    Main entry point for configuring and training the Digital Twin models.
    """
    print("Starting Digital Twin Configuration...")
    
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    
    # ---------------------------------------------------------
    # 1. TRAIN PHYSICS-BASED MODEL (Random Forest)
    # ---------------------------------------------------------
    print("\n--- Step 1: Physics Model Training (CFD Surrogate) ---")
    
    pipeline = CFDPipeline(
        raw_profiles_path=os.path.join(data_dir, "profiles"),
        output_lookup_path=os.path.join(data_dir, "simulation_lookup.csv")
    )
    
    try:
        # Generate lookup table from raw ANSYS profiles
        df_sim = pipeline.generate_lookup_table()
        # Train the surrogate model
        pipeline.train_surrogate_model(df_sim)
    except Exception as e:
        print(f"[WARNING] Skipping physics model training (Check .prof files): {e}")

    # ---------------------------------------------------------
    # 2. TRAIN TIME-SERIES MODEL (LSTM)
    # ---------------------------------------------------------
    print("\n--- Step 2: Time-Series Model Training (LSTM) ---")
    
    loader = DataLoader(data_dir)
    
    try:
        # Load the processed dataset
        # Ensure the file path matches your directory structure
        dataset_path = os.path.join("processed", "crane_digital_twin_ml_dataset.csv")
        df_iot = loader.load_iot_dataset(dataset_path)
        
        # Feature Engineering: Calculate Virtual Wind if it does not exist
        if 'Virtual_Wind' not in df_iot.columns:
            print("Calculating 'Virtual_Wind' average from sensor zones...")
            wind_cols = [c for c in df_iot.columns if 'Wind Speed' in c]
            df_iot['Virtual_Wind'] = df_iot[wind_cols].mean(axis=1)

        # Initialize and train the LSTM model
        lstm = WindForecasterLSTM(history_window=3)
        lstm.train(df_iot, target_col='Virtual_Wind')
        
    except Exception as e:
        print(f"[ERROR] Critical failure loading IoT data: {e}")

    print("\nConfiguration Complete. System ready.")
    print("To launch the interface, run: streamlit run dashboard.py")

if __name__ == "__main__":
    main()