# main.py
import pandas as pd
import os
from src.loader import DataLoader
from src.pipeline_setup import CFDPipeline # Importamos la clase nueva

def main():
    # --- RUTAS Y CONFIGURACIÓN ---
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Inputs
    IOT_FILE = "crane_digital_twin_ml_dataset.csv"
    PROFILES_DIR = os.path.join(DATA_DIR, "profiles") # Carpeta raíz de los .prof
    
    # Outputs (Generados automáticamente)
    LOOKUP_CSV = os.path.join(DATA_DIR, "simulation_lookup.csv")
    RUPTURE_PRESSURE = 355.0  # Dato físico de la grúa

    # ---------------------------------------------------------
    # FASE 1: PIPELINE DE INGENIERÍA (CFD -> CSV -> AI Model)
    # ---------------------------------------------------------
    print("="*60)
    print(" INICIANDO PIPELINE DE FÍSICA E INGENIERÍA")
    print("="*60)
    
    # Inicializar el pipeline
    cfd_engine = CFDPipeline(PROFILES_DIR, LOOKUP_CSV, RUPTURE_PRESSURE)
    
    try:
        # 1. Generar Lookup Table desde archivos crudos .prof
        df_sim = cfd_engine.generate_lookup_table()
        
        # 2. Entrenar el modelo sustituto (Random Forest)
        cfd_engine.train_surrogate_model(df_sim)
        
    except Exception as e:
        print(f"\n❌ Error Crítico en Pipeline CFD: {e}")
        print("Asegúrate de que los archivos .prof estén en data/profiles/ y tengan nombres como 'ang_0_vel_10.prof'")
        return

    # ---------------------------------------------------------
    # FASE 2: GEMELO DIGITAL (IoT + Modelo Entrenado)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" EJECUTANDO GEMELO DIGITAL")
    print("="*60)

    loader = DataLoader(DATA_DIR)
    
    # 1. Cargar datos IoT
    try:
        df_iot = loader.load_iot_dataset(IOT_FILE)
    except FileNotFoundError:
        print("❌ No se encontró el dataset de sensores.")
        return

    # 2. Aplicar el modelo físico al dataset IoT
    # Usamos el modelo entrenado en Fase 1 para predecir sobre datos nuevos
    # Asumimos que queremos calcular el riesgo basado en el viento promedio (Sensor Virtual)
    
    # Calcular Viento Promedio (Sensor Virtual)
    df_iot['Virtual_Wind'] = (df_iot['Wind Speed (m/sec)_z1'] + 
                              df_iot['Wind Speed (m/sec)_z2'] + 
                              df_iot['Wind Speed (m/sec)_z3']) / 3
                              
    # Calcular Dirección Promedio (Simplificado)
    df_iot['Virtual_Angle'] = df_iot['Wind Direction (°)_z1'] 

    # Predicción de Riesgo Físico
    df_iot['Predicted_Structure_Risk'] = cfd_engine.predict_risk_batch(
        df_iot, 
        wind_col='Virtual_Wind', 
        dir_col='Virtual_Angle'
    )

    # 3. Mostrar Resultados
    print("\n📊 Muestra de Resultados del Gemelo Digital:")
    print(df_iot[['Date', 'Virtual_Wind', 'Virtual_Angle', 'Predicted_Structure_Risk']].head(10))
    
    high_risk = df_iot[df_iot['Predicted_Structure_Risk'] > 0.8]
    print(f"\n🚨 Alertas de Integridad Estructural (>80% Riesgo): {len(high_risk)} eventos")

    # Guardar resultado final
    df_iot.to_csv(os.path.join(DATA_DIR, "final_digital_twin_results.csv"), index=False)
    print("💾 Resultados guardados exitosamente.")

if __name__ == "__main__":
    main()