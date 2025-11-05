import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1) Load data
# ----------------------------
# Asegúrate de que las rutas a tus archivos CSV sean correctas
z1 = pd.read_csv("Z1_CAJICA_ambient-weather-20250604-20251104.csv", sep=",")
z2 = pd.read_csv("Z2_GIRALDA_ambient-weather-20250604-20251104.csv", sep=";")
z3 = pd.read_csv("Z3_OIKOS_ambient-weather-20250604-20251104.csv", sep=";")

for df in (z1, z2, z3):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=['Date'], inplace=True) # Buena práctica: eliminar filas sin fecha

# ----------------------------
# 2) Keep only columns we need and rename per zone
# ----------------------------
def pick(df, zone):
    # Lista de columnas de interés
    cols_to_keep = [
        "Date",
        "Outdoor Temperature (°C)",
        "Humidity (%)",
        "Solar Radiation (W/m^2)",
        "Wind Speed (m/sec)",
        "Daily Rain (mm)",
        "Relative Pressure (mmHg)"
    ]
    
    # Renombrado de columnas
    rename_map = {
        "Outdoor Temperature (°C)": f"Temp_{zone}",
        "Humidity (%)":             f"Hum_{zone}",
        "Solar Radiation (W/m^2)":  f"Solar_{zone}",
        "Wind Speed (m/sec)":       f"Wind_{zone}",
        "Daily Rain (mm)":          f"Rain_{zone}",
        "Relative Pressure (mmHg)": f"Pres_{zone}",
    }
    
    # Filtra las columnas que existen en el DF
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df_picked = df[existing_cols]
    
    # Renombra
    return df_picked.rename(columns=rename_map)


z1_picked = pick(z1, "Z1")
z2_picked = pick(z2, "Z2")
z3_picked = pick(z3, "Z3")

# --------------------------------------------------------------------
# 3) !! CORRECCIÓN CRÍTICA: USAR 'OUTER JOIN' CON PD.CONCAT !!
# --------------------------------------------------------------------
# Poner 'Date' como índice en cada DataFrame ANTES de unirlos
df_list = [
    z1_picked.set_index('Date'), 
    z2_picked.set_index('Date'), 
    z3_picked.set_index('Date')
]

# Usar pd.concat con axis=1 (unir por columnas) y join='outer' (mantener todos los datos)
# Esto reemplaza tu 'inner merge'
df = pd.concat(df_list, axis=1, join='outer')
df = df.sort_index() # Ordenar por fecha
# --------------------------------------------------------------------

# ----------------------------
# 4) Hourly mean aggregation
# ----------------------------
# Usar .mean(numeric_only=True) es una buena práctica por si alguna columna no es numérica
hourly = df.resample("1h").mean(numeric_only=True)

# ----------------------------
# 5) Plot: 3 x 2 grid (a–f)
# ----------------------------
plt.figure(figsize=(14, 8))

# Colores por zona (matplotlib defaults para consistencia)
c1, c2, c3 = "#1f77b4", "#ff7f0e", "#2ca02c"
lw = 1.0 # Un poco más de grosor para visibilidad

# a) Temperature
ax1 = plt.subplot(3, 2, 1)
ax1.plot(hourly.index, hourly["Temp_Z1"], c1, lw=lw, label="Zone 1 – Cajicá")
ax1.plot(hourly.index, hourly["Temp_Z2"], c2, lw=lw, label="Zone 2 – La Giralda")
ax1.plot(hourly.index, hourly["Temp_Z3"], c3, lw=lw, label="Zone 3 – Oikos")
ax1.set_title("a) Outdoor Temperature (°C) — Comparative Across Zones (Hourly Mean)")
ax1.set_ylabel("Outdoor Temperature (°C)")
ax1.grid(True, ls="--", alpha=0.5)
ax1.legend(loc="upper left", fontsize=8)

# b) Humidity
ax2 = plt.subplot(3, 2, 3, sharex=ax1)
ax2.plot(hourly.index, hourly["Hum_Z1"], c1, lw=lw, label="Zone 1 – Cajicá")
ax2.plot(hourly.index, hourly["Hum_Z2"], c2, lw=lw, label="Zone 2 – La Giralda")
ax2.plot(hourly.index, hourly["Hum_Z3"], c3, lw=lw, label="Zone 3 – Oikos")
ax2.set_title("b) Humidity (%) — Comparative Across Zones (Hourly Mean)")
ax2.set_ylabel("Humidity (%)")
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend(loc="lower center", fontsize=8)

# c) Solar Radiation
ax3 = plt.subplot(3, 2, 5, sharex=ax1)
ax3.plot(hourly.index, hourly["Solar_Z1"], c1, lw=lw, label="Zone 1 – Cajicá")
ax3.plot(hourly.index, hourly["Solar_Z2"], c2, lw=lw, label="Zone 2 – La Giralda")
ax3.plot(hourly.index, hourly["Solar_Z3"], c3, lw=lw, label="Zone 3 – Oikos")
ax3.set_title("c) Solar Radiation (W/m^2) — Comparative Across Zones (Hourly Mean)")
ax3.set_xlabel("Date")
ax3.set_ylabel("Solar Radiation (W/m^2)")
ax3.grid(True, ls="--", alpha=0.5)
ax3.legend(loc="upper left", fontsize=8)

# d) Wind Speed
ax4 = plt.subplot(3, 2, 2, sharex=ax1)
ax4.plot(hourly.index, hourly["Wind_Z1"], c1, lw=lw, label="Zone 1 – Cajicá")
ax4.plot(hourly.index, hourly["Wind_Z2"], c2, lw=lw, label="Zone 2 – La Giralda")
ax4.plot(hourly.index, hourly["Wind_Z3"], c3, lw=lw, label="Zone 3 – Oikos")
ax4.set_title("d) Wind Speed (m/s) — Comparative Across Zones (Hourly Mean)")
ax4.set_ylabel("Wind Speed (m/s)")
ax4.grid(True, ls="--", alpha=0.5)
ax4.legend(loc="upper left", fontsize=8)

# e) Daily Rain
ax5 = plt.subplot(3, 2, 4, sharex=ax1)
ax5.plot(hourly.index, hourly["Rain_Z1"], c1, lw=lw, label="Zone 1 – Cajicá")
ax5.plot(hourly.index, hourly["Rain_Z2"], c2, lw=lw, label="Zone 2 – La Giralda")
ax5.plot(hourly.index, hourly["Rain_Z3"], c3, lw=lw, label="Zone 3 – Oikos")
ax5.set_title("e) Daily Rain (mm) — Comparative Across Zones (Hourly Mean)")
ax5.set_ylabel("Daily Rain (mm)")
ax5.grid(True, ls="--", alpha=0.5)
ax5.legend(loc="upper left", fontsize=8)

# f) Relative Pressure
ax6 = plt.subplot(3, 2, 6, sharex=ax1)
ax6.plot(hourly.index, hourly["Pres_Z1"], c1, lw=lw, label="Zone 1 – Cajicá")
ax6.plot(hourly.index, hourly["Pres_Z2"], c2, lw=lw, label="Zone 2 – La Giralda")
ax6.plot(hourly.index, hourly["Pres_Z3"], c3, lw=lw, label="Zone 3 – Oikos")
ax6.set_title("f) Relative Pressure (mmHg) — Comparative Across Zones (Hourly Mean)")
ax6.set_xlabel("Date")
ax6.set_ylabel("Relative Pressure (mmHg)")
ax6.grid(True, ls="--", alpha=0.5)
ax6.legend(loc="upper left", fontsize=8)

# Ocultar etiquetas de X en los ejes superiores para limpiar la gráfica
for ax in [ax1, ax2, ax4, ax5]:
    plt.setp(ax.get_xticklabels(), visible=False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.15) # Ajustar espaciado
plt.show()