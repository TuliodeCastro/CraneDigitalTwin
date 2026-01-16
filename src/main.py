import os
import pandas as pd
from pipeline_setup import CFDPipeline
from time_series_model import WindForecasterLSTM
from data_loader import DataLoader # Tu clase original

def main():
    print("🚀 INICIANDO CONFIGURACIÓN DEL GEMELO DIGITAL")
    
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # 1. ENTRENAR MODELO FÍSICO (Random Forest)
    # -----------------------------------------
    pipeline = CFDPipeline(
        raw_profiles_path=os.path.join(DATA_DIR, "profiles"),
        output_lookup_path=os.path.join(DATA_DIR, "simulation_lookup.csv")
    )
    try:
        df_sim = pipeline.generate_lookup_table()
        pipeline.train_surrogate_model(df_sim)
    except Exception as e:
        print(f"⚠️ Salto entrenamiento físico (revisa archivos .prof): {e}")

    # 2. ENTRENAR MODELO TEMPORAL (LSTM)
    # ----------------------------------
    loader = DataLoader(DATA_DIR)
    try:
        df_iot = loader.load_iot_dataset("crane_digital_twin_ml_dataset.csv")
        
        # Calcular Viento Virtual si no existe
        if 'Virtual_Wind' not in df_iot.columns:
            # Lógica simple de promedio
            cols = [c for c in df_iot.columns if 'Wind Speed' in c]
            df_iot['Virtual_Wind'] = df_iot[cols].mean(axis=1)

        lstm = WindForecasterLSTM(history_window=3)
        lstm.train(df_iot, target_col='Virtual_Wind')
        
    except Exception as e:
        print(f"❌ Error crítico cargando datos IoT: {e}")

    print("\n✅ TODO LISTO. Ejecuta ahora: streamlit run dashboard.py")

if __name__ == "__main__":
    main()