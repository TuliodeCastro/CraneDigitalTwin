import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------
# 1) Cargar datos (Usa tus rutas locales)
# ----------------------------
# Asegúrate de que las rutas a tus archivos CSV sean correctas
path_z1 = "/Users/santiagosantafe/Downloads/ambient-weather-20250322-20250925-3.csv"
path_z2 = "/Users/santiagosantafe/Downloads/ICRERA/Z2_GIRALDA_ambient-weather-20250322-20250925.csv"
path_z3 = "/Users/santiagosantafe/Downloads/ICRERA/Z3_OIKOS_ambient-weather-20250322-20250925.csv"

try:
    z1_raw = pd.read_csv(path_z1, sep=",")
    z2_raw = pd.read_csv(path_z2, sep=";")
    z3_raw = pd.read_csv(path_z3, sep=";")
except FileNotFoundError:
    print("Error: No se encontraron los archivos CSV. Por favor, verifica las rutas.")
    exit()

for df in (z1_raw, z2_raw, z3_raw):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=['Date'], inplace=True) # Eliminar filas donde la fecha no se pudo parsear

# ----------------------------
# 2) Función para seleccionar y renombrar columnas
# ----------------------------
def pick(df, zone):
    cols_to_keep = [
        "Date", "Outdoor Temperature (°C)", "Solar Radiation (W/m^2)"
    ]
    rename_map = {
        "Outdoor Temperature (°C)": f"Temp_{zone}",
        "Solar Radiation (W/m^2)":  f"Solar_{zone}",
    }
    # Filtra solo las columnas que realmente existen en el DataFrame
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df_picked = df[existing_cols]
    return df_picked.rename(columns=rename_map)

z1 = pick(z1_raw, "Z1")
z2 = pick(z2_raw, "Z2")
z3 = pick(z3_raw, "Z3")

# ----------------------------
# 3) Usar 'concat' (join='outer')
# ----------------------------
# Esto asegura que los datos de Z1 y Z3 se muestren 
# incluso cuando Z2 no tiene datos.
df_list = [z1.set_index('Date'), z2.set_index('Date'), z3.set_index('Date')]
df_full = pd.concat(df_list, axis=1, join='outer').sort_index()

# ----------------------------
# 4) Remuestreo a media horaria (1h)
# ----------------------------
hourly = df_full.resample("1h").mean()

# ----------------------------
# 5) Configuración de Gráficas (Estilo para Presentación)
# ----------------------------
c1, c2, c3 = "#1f77b4", "#ff7f0e", "#2ca02c" 
title_fontsize = 18
label_fontsize = 14
legend_fontsize = 12
line_width = 1.5 

print("Generando gráfica de Temperatura...")

# ----------------------------
# 6) GRÁFICA 1: TEMPERATURA
# ----------------------------
plt.figure(figsize=(14, 7))
ax1 = plt.gca()

ax1.plot(hourly.index, hourly["Temp_Z1"], c1, lw=line_width, label="Zone 1 – Cajicá (Urban)")
ax1.plot(hourly.index, hourly["Temp_Z2"], c2, lw=line_width, label="Zone 2 – La Giralda (River Basin)")
ax1.plot(hourly.index, hourly["Temp_Z3"], c3, lw=line_width, label="Zone 3 – Oikos (Transitional)")

ax1.set_title("a) Outdoor Temperature — Evidence of Microclimate Variability", fontsize=title_fontsize, weight='bold')
ax1.set_ylabel("Outdoor Temperature (°C)", fontsize=label_fontsize)
ax1.set_xlabel("Date", fontsize=label_fontsize)
ax1.grid(True, ls="-", alpha=0.3)
ax1.legend(loc="upper left", fontsize=legend_fontsize)
ax1.set_xlim(hourly.index.min(), hourly.index.max())

plt.tight_layout()
plt.savefig('temperature_plot_microclimate.png', dpi=150)
plt.close() # Cierra la figura para liberar memoria

print("Gráfica de Temperatura guardada como 'temperature_plot_microclimate.png'")

# ----------------------------
# 7) GRÁFICA 2: RADIACIÓN SOLAR
# ----------------------------
print("Generando gráfica de Radiación Solar...")
plt.figure(figsize=(14, 7))
ax2 = plt.gca()

if "Solar_Z1" in hourly.columns:
    ax2.plot(hourly.index, hourly["Solar_Z1"], c1, lw=line_width, label="Zone 1 – Cajicá (Urban)")
if "Solar_Z2" in hourly.columns:
    ax2.plot(hourly.index, hourly["Solar_Z2"], c2, lw=line_width, label="Zone 2 – La Giralda (River Basin)")
if "Solar_Z3" in hourly.columns:
    ax2.plot(hourly.index, hourly["Solar_Z3"], c3, lw=line_width, label="Zone 3 – Oikos (Transitional)")

ax2.set_title("c) Solar Radiation — Evidence of Microclimate Variability", fontsize=title_fontsize, weight='bold')
ax2.set_ylabel("Solar Radiation (W/m^2)", fontsize=label_fontsize)
ax2.set_xlabel("Date", fontsize=label_fontsize)
ax2.grid(True, ls="-", alpha=0.3)
ax2.legend(loc="upper left", fontsize=legend_fontsize)
ax2.set_xlim(hourly.index.min(), hourly.index.max())

plt.tight_layout()
plt.savefig('solar_radiation_plot_microclimate.png', dpi=150)
plt.close() # Cierra la figura para liberar memoria

print("Gráfica de Radiación Solar guardada como 'solar_radiation_plot_microclimate.png'")
print("\n¡Proceso completado!")