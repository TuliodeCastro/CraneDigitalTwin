import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

def run_stress_test():
    # 1. Configuración
    BASE_DIR = os.getcwd()
    # Asegúrate de que esta ruta apunte a donde realmente se guardó el modelo
    # En tu ejecución anterior dice: "models/surrogate_risk_model.pkl"
    MODEL_PATH = os.path.join(BASE_DIR, "models", "surrogate_risk_model.pkl")
    
    # 2. Cargar el modelo entrenado
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No se encuentra el modelo en {MODEL_PATH}")
        return
    
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Modelo cargado para pruebas de estrés.")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return

    # 3. Generar Datos Sintéticos (Escenario de Huracán)
    winds = np.linspace(0, 30, 100)
    
    # Escenario A: Viento frontal (0 grados)
    df_scenario_a = pd.DataFrame({
        'angle': [0] * 100,
        'velocity': winds
    })
    
    # Escenario B: Viento lateral crítico (90 grados)
    df_scenario_b = pd.DataFrame({
        'angle': [90] * 100,
        'velocity': winds
    })

    # CORRECCIÓN IMPORTANTE: Forzar el orden exacto de columnas que usó el entrenamiento
    expected_cols = ["angle", "velocity"]
    df_scenario_a = df_scenario_a[expected_cols]
    df_scenario_b = df_scenario_b[expected_cols]

    # 4. Predecir Riesgo
    try:
        risk_a = model.predict(df_scenario_a)
        risk_b = model.predict(df_scenario_b)
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return

    # 5. Visualizar "La Curva de la Muerte"
    plt.figure(figsize=(10, 6))
    
    plt.plot(winds, risk_a, label='Viento Frontal (0°)', color='blue', linewidth=2)
    plt.plot(winds, risk_b, label='Viento Lateral (90°)', color='orange', linewidth=2, linestyle='--')
    
    # Zonas de Riesgo
    plt.axhline(y=0.8, color='red', linestyle=':', label='Umbral Crítico (80%)')
    plt.axhspan(0.8, 1.1, color='red', alpha=0.1)
    plt.text(1, 0.85, "ZONA DE PELIGRO", color='red', fontweight='bold')

    plt.title("Prueba de Estrés del Gemelo Digital: Respuesta ante Vientos Extremos")
    plt.xlabel("Velocidad del Viento (m/s)")
    plt.ylabel("Nivel de Riesgo Estructural (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    output_path = os.path.join(BASE_DIR, "data", "stress_test_result.png")
    # Asegurarse de que el directorio data exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"📊 Gráfico de prueba de estrés guardado en: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_stress_test()