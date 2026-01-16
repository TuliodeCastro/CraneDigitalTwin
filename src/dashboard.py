import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from simulation_mapper import DigitalTwinBackend

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crane Digital Twin | Enterprise", 
    layout="wide", 
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL CSS (INDUSTRIAL LOOK) ---
st.markdown("""
<style>
    /* General Layout */
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1.5rem; }
    
    /* Metrics Styling */
    div.stMetric { 
        background-color: white; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Status Banner Styles */
    .status-safe { background-color: #28a745; color: white; padding: 15px; text-align: center; border-radius: 5px; font-weight: bold; font-size: 20px; }
    .status-warning { background-color: #ffc107; color: black; padding: 15px; text-align: center; border-radius: 5px; font-weight: bold; font-size: 20px; }
    .status-critical { background-color: #dc3545; color: white; padding: 15px; text-align: center; border-radius: 5px; font-weight: bold; font-size: 20px; }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: white; border-radius: 5px 5px 0px 0px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    
    /* Section Headers */
    .sub-header { font-size: 18px; font-weight: 600; color: #495057; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

# --- 3. ROBUST FILE LOADING & PROCESSING ---
@st.cache_resource
def load_resources():
    # Intelligent Path Detection
    current_script = Path(__file__).resolve()
    project_root = current_script.parent.parent 
    
    # 1. Locate Data
    csv_candidates = [
        project_root / "data" / "processed" / "crane_digital_twin_ml_dataset.csv",
        project_root / "data" / "crane_digital_twin_ml_dataset.csv"
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    
    # 2. Locate 3D Model
    obj_candidates = [
        Path("/Users/santiagosantafe/Desktop/ICRERA/data/Kran.obj"), # Absolute path priority
        project_root / "data" / "profiles" / "Kran.obj",
        project_root / "data" / "Kran.obj"
    ]
    obj_path = next((p for p in obj_candidates if p.exists()), None)
    
    # 3. Load Backend
    backend = DigitalTwinBackend()
    df = pd.DataFrame()
    
    try:
        if not csv_path: return None, None, None, "CSV Data not found."
        
        backend.load_models()
        df = pd.read_csv(csv_path)
        
        # Ensure 'Date' is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
        # Ensure Virtual Wind exists
        if 'Virtual_Wind' not in df.columns:
            wind_cols = [c for c in df.columns if 'Wind Speed' in c]
            df['Virtual_Wind'] = df[wind_cols].mean(axis=1) if wind_cols else 0.0
            
    except Exception as e:
        return None, None, None, str(e)
        
    return backend, df, obj_path, None

@st.cache_resource
def process_3d_geometry(filepath):
    """
    Lee un archivo .obj, extrae geometría y aplica la rotación específica.
    """
    if not filepath or not filepath.exists(): return None
    
    vertices = []
    faces = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '): 
                    vertices.append([float(x) for x in line.strip().split()[1:]])
                elif line.startswith('f '): 
                    face = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                    if len(face) == 3: faces.append(face)
                    elif len(face) > 3:
                        for i in range(1, len(face) - 1): faces.append([face[0], face[i], face[i+1]])

        vertices = np.array(vertices)
        if len(vertices) == 0: return None
        
        # --- SECCIÓN DE ROTACIÓN CORREGIDA ---
        # 1. Rotar -90 grados alrededor del eje X (Ajuste inicial)
        theta_x = np.radians(-90) 
        c_x, s_x = np.cos(theta_x), np.sin(theta_x)
        rotation_x = np.array([
            [1,    0,     0],
            [0,  c_x, -s_x],
            [0,  s_x,  c_x]
        ])
        
        # 2. Rotar 90 grados alrededor del eje Z (Orientación)
        theta_z = np.radians(90)
        c_z, s_z = np.cos(theta_z), np.sin(theta_z)
        rotation_z = np.array([
            [c_z, -s_z, 0],
            [s_z,  c_z, 0],
            [0,     0,  1]
        ])

        # 3. Rotar 180 grados alrededor del eje Y (VOLTEAR COMPLETAMENTE)
        theta_y = np.radians(180)
        c_y, s_y = np.cos(theta_y), np.sin(theta_y)
        rotation_y = np.array([
            [ c_y,  0,  s_y],
            [   0,  1,    0],
            [-s_y,  0,  c_y]
        ])
        
        # Aplicar las tres rotaciones secuencialmente
        vertices = vertices.dot(rotation_x.T).dot(rotation_z.T).dot(rotation_y.T)
        # ---------------------------------------------------
        
        return (vertices[:,0], vertices[:,1], vertices[:,2], 
                [f[0] for f in faces], [f[1] for f in faces], [f[2] for f in faces])
    except: return None

# --- LOAD SYSTEM ---
backend, df, obj_path, error_msg = load_resources()
if error_msg: st.error(f"System Error: {error_msg}"); st.stop()

geo_data = process_3d_geometry(obj_path)

# --- 4. SESSION STATE & SIDEBAR ---
if 'idx' not in st.session_state: st.session_state.idx = 1000
if 'run' not in st.session_state: st.session_state.run = False

with st.sidebar:
    st.header("Control Panel")
    
    # Mode Selection
    mode = st.radio("Operation Mode", ["Live Simulation", "Static Analysis"], index=0)
    
    st.markdown("---")
    if mode == "Live Simulation":
        c1, c2 = st.columns(2)
        if c1.button("START", use_container_width=True, type="primary"): 
            st.session_state.run = True
            st.rerun()
        if c2.button("STOP", use_container_width=True): 
            st.session_state.run = False
            st.rerun()
            
        speed = st.slider("Update Speed (s)", 0.1, 2.0, 0.5)
        st.session_state.idx = st.number_input("Start Index", 0, len(df), st.session_state.idx)
    
    st.markdown("---")
    st.caption("System Version: v2.9.0 (Y-Flip)")
    st.caption("Model: Hybrid LSTM-RF")

# --- 5. MAIN DASHBOARD LAYOUT ---
st.title("Crane Digital Twin Operation Center")

# TABS STRUCTURE (INCLUYENDO STRESS TEST)
tab_live, tab_forecast, tab_analytics, tab_stress = st.tabs(["Live Operations", "Forecast Report", "Historical Analytics", "⚡ Stress Test Lab"])

# === LOGIC FOR LIVE DATA ===
current_row = df.iloc[st.session_state.idx]
hist_window = df.iloc[max(0, st.session_state.idx-3) : st.session_state.idx]['Virtual_Wind'].values

# AI Inference
results = backend.get_digital_twin_status(
    current_wind=current_row['Virtual_Wind'],
    current_angle=current_row.get('Wind Direction (°)_z1', 0),
    recent_history=hist_window
)

# === TAB 1: LIVE OPERATIONS ===
with tab_live:
    # A. Status Banner
    status_class = "status-safe"
    if "WARNING" in results['status']: status_class = "status-warning"
    if "CRITICAL" in results['status']: status_class = "status-critical"
    
    st.markdown(f'<div class="{status_class}">SYSTEM STATUS: {results["status"].replace("_", " ")}</div>', unsafe_allow_html=True)
    
    # B. Main Content
    col_3d, col_metrics = st.columns([1.5, 1])
    
    with col_3d:
        st.markdown('<div class="sub-header">Site Visualization (Digital Twin)</div>', unsafe_allow_html=True)
        if geo_data:
            x, y, z, i, j, k = geo_data
            color_hex = "#dc3545" if "CRITICAL" in results['status'] else ("#ffc107" if "WARNING" in results['status'] else "#28a745")
            
            fig3d = go.Figure(data=[go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, 
                color=color_hex, opacity=1.0, flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.9, roughness=0.1)
            )])
            
            # --- AJUSTE DE CÁMARA ---
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(visible=False), 
                    yaxis=dict(visible=False), 
                    zaxis=dict(visible=False),
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=2.5, y=2.5, z=2.0),
                        center=dict(x=0, y=0, z=0)
                    )
                ),
                height=400, margin=dict(l=0,r=0,t=0,b=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.warning("3D Model not loaded.")

    with col_metrics:
        st.markdown('<div class="sub-header">Live Telemetry</div>', unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        m1.metric("Wind Speed", f"{results['current_wind']:.2f} m/s")
        m2.metric("Direction", f"{current_row.get('Wind Direction (°)_z1', 0):.0f}°")
        
        st.metric("Structural Risk Index", f"{results['current_risk']:.4f}", 
                  delta="Critical" if results['current_risk'] > 0.8 else "Stable", delta_color="inverse")
        
        st.markdown('<div class="sub-header">Short-Term Forecast (AI)</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        f1.metric("Pred. Wind (+10m)", f"{float(results['future_wind_10m']):.2f} m/s")
        f2.metric("Pred. Risk (+10m)", f"{results['future_risk_10m']:.4f}")

    # C. Live Chart
    st.markdown('<div class="sub-header">Wind Velocity Trend</div>', unsafe_allow_html=True)
    past_data = df.iloc[max(0, st.session_state.idx - 60) : st.session_state.idx + 1] # Last 5 hours approx
    
    fig_chart = go.Figure()
    fig_chart.add_trace(go.Scatter(x=past_data['Date'], y=past_data['Virtual_Wind'], 
                                   name="Historical", line=dict(color='#6c757d')))
    # Add Prediction Point
    future_time = pd.to_datetime(current_row['Date']) + pd.Timedelta(minutes=10)
    fig_chart.add_trace(go.Scatter(x=[future_time], y=[results['future_wind_10m']],
                                   mode='markers', marker=dict(color='red', size=10), name="AI Forecast"))
    
    fig_chart.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Wind (m/s)")
    st.plotly_chart(fig_chart, use_container_width=True)


# === TAB 2: FORECAST REPORT ===
with tab_forecast:
    st.markdown("### 📋 Predictive Safety Report")
    
    if st.button("Generate Forecast Table"):
        # Generate prediction for next 60 minutes (12 steps)
        # Reconstruct history for the backend
        curr_hist = []
        for n in range(3):
            r = df.iloc[st.session_state.idx - (2-n)]
            # Simple avg wind
            w = (r.get('Wind Speed (m/sec)_z1',0) + r.get('Wind Speed (m/sec)_z2',0) + r.get('Wind Speed (m/sec)_z3',0))/3
            curr_hist.append(w)
            
        forecast_df = backend.get_forecast_data(curr_hist, steps=12)
        
        # Add Time timestamps based on current simulation time
        start_time = pd.to_datetime(current_row['Date'])
        forecast_df['Timestamp'] = [start_time + pd.Timedelta(minutes=(i+1)*5) for i in range(len(forecast_df))]
        
        # Display formatted table
        st.dataframe(
            forecast_df[['Timestamp', 'Predicted Wind (m/s)', 'Predicted Risk', 'Status']],
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Prediction Time", format="HH:mm"),
                "Predicted Risk": st.column_config.ProgressColumn("Risk Level", min_value=0, max_value=1, format="%.4f"),
            },
            use_container_width=True
        )
    else:
        st.info("Click the button above to run the recursive LSTM forecasting model for the next 60 minutes.")

# === TAB 3: ANALYTICS ===
with tab_analytics:
    st.markdown("### 📊 Historical Data Analysis")
    
    if 'Date' in df.columns:
        min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
        d_range = st.date_input("Filter Date Range", [min_date, max_date])
        
        if len(d_range) == 2:
            mask = (df['Date'].dt.date >= d_range[0]) & (df['Date'].dt.date <= d_range[1])
            df_filtered = df.loc[mask]
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Wind Speed Distribution**")
                fig_hist = px.histogram(df_filtered, x="Virtual_Wind", nbins=50, 
                                      color_discrete_sequence=['#2C3E50'], title="")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with c2:
                st.markdown("**Wind Direction vs. Intensity**")
                # Polar plot or Scatter
                if 'Wind Direction (°)_z1' in df_filtered.columns:
                    fig_pol = px.scatter_polar(df_filtered, r="Virtual_Wind", theta="Wind Direction (°)_z1",
                                             color="Virtual_Wind", color_continuous_scale="Plasma")
                    st.plotly_chart(fig_pol, use_container_width=True)
    else:
        st.warning("Date column not found in dataset for analytics.")

# === TAB 4: STRESS TEST LAB (RECUPERADO) ===
with tab_stress:
    st.markdown("### ⚡ Interactive Stress Test Simulator")
    st.info("Manually override environmental conditions to test structural integrity limits.")
    
    col_sim, col_res = st.columns([1, 2])
    
    with col_sim:
        st.markdown("**Simulation Parameters**")
        sim_wind = st.slider("Simulated Wind Speed (m/s)", 0, 50, 10, key="stress_wind")
        sim_angle = st.slider("Wind Angle (°)", 0, 360, 0, key="stress_angle")
        
        # Calcular riesgo manual
        input_manual = pd.DataFrame({'angle': [sim_angle], 'velocity': [sim_wind]})
        sim_risk = backend.risk_model.predict(input_manual)[0]
        
        st.divider()
        st.metric("Simulated Risk Result", f"{sim_risk:.4f}")
        
        if sim_risk > 0.8:
            st.error("⛔ STRUCTURAL FAILURE")
        elif sim_risk > 0.5:
            st.warning("⚠️ CRITICAL LOAD")
        else:
            st.success("✅ STRUCTURAL INTEGRITY OK")
            
    with col_res:
        st.markdown("**Structural Vulnerability Curve**")
        # Generar Curva dinámica
        winds = np.linspace(0, 45, 50)
        risks = backend.run_stress_test(winds, angle_fixed=sim_angle)
        
        fig_stress = go.Figure()
        
        # Curva de Riesgo
        fig_stress.add_trace(go.Scatter(x=winds, y=risks, mode='lines', name='Risk Curve', line=dict(color='blue', width=3)))
        
        # Línea de Límite
        fig_stress.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Failure Threshold")
        
        # Punto Actual Simulado
        fig_stress.add_trace(go.Scatter(x=[sim_wind], y=[sim_risk], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Current Sim'))
        
        fig_stress.update_layout(title=f"Risk vs Wind Speed at {sim_angle}° Angle", xaxis_title="Wind Speed (m/s)", yaxis_title="Risk Index", height=400)
        st.plotly_chart(fig_stress, use_container_width=True)

# === SIMULATION LOOP ===
if st.session_state.run and mode == "Live Simulation":
    time.sleep(speed)
    st.session_state.idx += 1
    if st.session_state.idx >= len(df)-10: st.session_state.idx = 1000
    st.rerun()