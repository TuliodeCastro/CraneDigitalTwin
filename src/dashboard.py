import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend logic
# Note: Ensure the file containing DigitalTwinBackend is named 'digital_twin.py'
from digital_twin import DigitalTwinBackend

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crane Digital Twin | Enterprise Edition", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL CSS (INDUSTRIAL THEME) ---
st.markdown("""
<style>
    /* General Layout */
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1.5rem; }
    
    /* Metrics Styling */
    div.stMetric { 
        background-color: white; 
        padding: 15px; 
        border-radius: 4px; 
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Status Banner Styles */
    .status-safe { 
        background-color: #28a745; 
        color: white; 
        padding: 15px; 
        text-align: center; 
        border-radius: 4px; 
        font-weight: 600; 
        font-size: 18px; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-warning { 
        background-color: #ffc107; 
        color: #212529; 
        padding: 15px; 
        text-align: center; 
        border-radius: 4px; 
        font-weight: 600; 
        font-size: 18px; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-critical { 
        background-color: #dc3545; 
        color: white; 
        padding: 15px; 
        text-align: center; 
        border-radius: 4px; 
        font-weight: 600; 
        font-size: 18px; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; 
        background-color: white; 
        border-radius: 4px 4px 0px 0px; 
        border: 1px solid #dee2e6;
        border-bottom: none;
    }
    
    /* Section Headers */
    .sub-header { 
        font-size: 16px; 
        font-weight: 700; 
        color: #343a40; 
        margin-top: 20px; 
        margin-bottom: 12px; 
        border-bottom: 2px solid #ced4da; 
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    """
    Loads the backend system, datasets, and 3D assets dynamically.
    """
    # 1. Path Setup
    base_dir = Path(__file__).resolve().parent.parent # Points to project root
    data_dir = base_dir / "data"
    
    # 2. Locate CSV Data
    csv_path = data_dir / "processed" / "crane_digital_twin_ml_dataset.csv"
    if not csv_path.exists():
        # Fallback to raw folder if processed doesn't exist
        csv_path = data_dir / "crane_digital_twin_ml_dataset.csv"

    # 3. Locate 3D Object
    # Checks multiple common locations for robustness
    obj_path = None
    possible_paths = [
        data_dir / "profiles" / "Kran.obj",
        data_dir / "Kran.obj",
        base_dir / "Kran.obj"
    ]
    
    for p in possible_paths:
        if p.exists():
            obj_path = p
            break
    
    # 4. Initialize Backend
    backend = DigitalTwinBackend()
    df = pd.DataFrame()
    
    try:
        if not csv_path.exists():
            return None, None, None, f"CSV Data not found at: {csv_path}"
        
        backend.load_models()
        df = pd.read_csv(csv_path)
        
        # Data Cleaning
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
        # Feature Engineering (Virtual Wind)
        if 'Virtual_Wind' not in df.columns:
            wind_cols = [c for c in df.columns if 'Wind Speed' in c]
            df['Virtual_Wind'] = df[wind_cols].mean(axis=1) if wind_cols else 0.0
            
    except Exception as e:
        return None, None, None, f"Initialization Error: {str(e)}"
        
    return backend, df, obj_path, None

@st.cache_resource
def process_3d_geometry(filepath):
    """
    Parses .obj file and applies coordinate transformations for Plotly visualization.
    """
    if not filepath or not filepath.exists(): 
        return None
    
    vertices = []
    faces = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '): 
                    vertices.append([float(x) for x in line.strip().split()[1:]])
                elif line.startswith('f '): 
                    # Parse faces (indexes are 1-based in OBJ)
                    face = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                    if len(face) == 3: 
                        faces.append(face)
                    elif len(face) > 3:
                        # Triangulate polygons
                        for i in range(1, len(face) - 1): 
                            faces.append([face[0], face[i], face[i+1]])

        vertices = np.array(vertices)
        if len(vertices) == 0: 
            return None
        
        # --- COORDINATE TRANSFORMATION ---
        # 1. Rotate -90 deg on X (Upright correction)
        theta_x = np.radians(-90) 
        c_x, s_x = np.cos(theta_x), np.sin(theta_x)
        rotation_x = np.array([[1, 0, 0], [0, c_x, -s_x], [0, s_x, c_x]])
        
        # 2. Rotate 90 deg on Z (Orientation correction)
        theta_z = np.radians(90)
        c_z, s_z = np.cos(theta_z), np.sin(theta_z)
        rotation_z = np.array([[c_z, -s_z, 0], [s_z, c_z, 0], [0, 0, 1]])

        # 3. Rotate 180 deg on Y (Flip correction)
        theta_y = np.radians(180)
        c_y, s_y = np.cos(theta_y), np.sin(theta_y)
        rotation_y = np.array([[c_y, 0, s_y], [0, 1, 0], [-s_y, 0, c_y]])
        
        # Apply transformations
        vertices = vertices.dot(rotation_x.T).dot(rotation_z.T).dot(rotation_y.T)
        
        return (
            vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            [f[0] for f in faces], [f[1] for f in faces], [f[2] for f in faces]
        )
    except Exception as e:
        print(f"Error processing 3D geometry: {e}")
        return None

# --- LOAD SYSTEM ---
backend, df, obj_path, error_msg = load_resources()

if error_msg:
    st.error(f"System Failure: {error_msg}")
    st.stop()

geo_data = process_3d_geometry(obj_path)

# --- 4. SESSION STATE & SIDEBAR ---
if 'idx' not in st.session_state: 
    st.session_state.idx = 1000
if 'run' not in st.session_state: 
    st.session_state.run = False

with st.sidebar:
    st.header("Control Panel")
    
    # Operation Mode
    mode = st.radio("Simulation Mode", ["Live Stream", "Static Analysis"], index=0)
    
    st.markdown("---")
    
    if mode == "Live Stream":
        c1, c2 = st.columns(2)
        if c1.button("START", use_container_width=True, type="primary"): 
            st.session_state.run = True
            st.rerun()
        if c2.button("STOP", use_container_width=True): 
            st.session_state.run = False
            st.rerun()
            
        speed = st.slider("Refresh Rate (s)", 0.1, 2.0, 0.5)
        st.session_state.idx = st.number_input("Time Step Index", 0, len(df)-1, st.session_state.idx)
    
    st.markdown("---")
    st.markdown("**System Info**")
    st.caption("Version: 2.9.0 Enterprise")
    st.caption("Model: Hybrid LSTM-RF")
    st.caption("Status: Online")

# --- 5. MAIN DASHBOARD ---
st.title("Crane Digital Twin Operations Center")

# TABS
tab_live, tab_forecast, tab_analytics, tab_stress = st.tabs([
    "Live Monitoring", 
    "Forecast Reports", 
    "Analytics", 
    "Stress Testing"
])

# === DATA PROCESSING ===
current_row = df.iloc[st.session_state.idx]
# Extract history window for LSTM (last 3 points)
hist_window = df.iloc[max(0, st.session_state.idx-3) : st.session_state.idx]['Virtual_Wind'].values

# Run Inference
results = backend.get_digital_twin_status(
    current_wind=current_row['Virtual_Wind'],
    current_angle=current_row.get('Wind Direction (°)_z1', 0),
    recent_history=hist_window
)

# === TAB 1: LIVE MONITORING ===
with tab_live:
    # A. Status Banner
    status_label = results["status"].replace("_", " ")
    status_class = "status-safe"
    
    if "WARNING" in status_label: 
        status_class = "status-warning"
    if "CRITICAL" in status_label: 
        status_class = "status-critical"
    
    st.markdown(f'<div class="{status_class}">SYSTEM STATUS: {status_label}</div>', unsafe_allow_html=True)
    
    # B. Content Grid
    col_3d, col_metrics = st.columns([1.5, 1])
    
    with col_3d:
        st.markdown('<div class="sub-header">3D Site Visualization</div>', unsafe_allow_html=True)
        if geo_data:
            x, y, z, i, j, k = geo_data
            
            # Dynamic Color based on risk
            mesh_color = "#28a745" # Green
            if "WARNING" in results['status']: mesh_color = "#ffc107" # Orange
            if "CRITICAL" in results['status']: mesh_color = "#dc3545" # Red
            
            fig3d = go.Figure(data=[go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, 
                color=mesh_color, opacity=1.0, flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.9, roughness=0.1)
            )])
            
            # Camera View
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
            st.info("3D Model visualization disabled (File not found).")

    with col_metrics:
        st.markdown('<div class="sub-header">Telemetry</div>', unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        m1.metric("Wind Speed", f"{results['current_wind']:.2f} m/s")
        m2.metric("Direction", f"{current_row.get('Wind Direction (°)_z1', 0):.0f}°")
        
        st.metric("Structural Risk Index", f"{results['current_risk']:.4f}", 
                  delta="Critical" if results['current_risk'] > 0.8 else "Safe", delta_color="inverse")
        
        st.markdown('<div class="sub-header">AI Forecast (+10 min)</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        f1.metric("Predicted Wind", f"{float(results['future_wind_10m']):.2f} m/s")
        f2.metric("Predicted Risk", f"{results['future_risk_10m']:.4f}")

    # C. Trend Chart
    st.markdown('<div class="sub-header">Velocity Trend Analysis</div>', unsafe_allow_html=True)
    
    # Get recent history
    past_data = df.iloc[max(0, st.session_state.idx - 60) : st.session_state.idx + 1]
    
    fig_chart = go.Figure()
    fig_chart.add_trace(go.Scatter(
        x=past_data['Date'], 
        y=past_data['Virtual_Wind'], 
        name="Historical", 
        line=dict(color='#6c757d', width=2)
    ))
    
    # Add Prediction Marker
    future_time = pd.to_datetime(current_row['Date']) + pd.Timedelta(minutes=10)
    fig_chart.add_trace(go.Scatter(
        x=[future_time], 
        y=[results['future_wind_10m']],
        mode='markers', 
        marker=dict(color='red', size=12, symbol='diamond'), 
        name="AI Forecast"
    ))
    
    fig_chart.update_layout(
        height=300, 
        margin=dict(l=0,r=0,t=20,b=0), 
        yaxis_title="Wind Velocity (m/s)",
        xaxis_title="Time",
        template="plotly_white"
    )
    st.plotly_chart(fig_chart, use_container_width=True)


# === TAB 2: FORECAST REPORT ===
with tab_forecast:
    st.markdown("### Predictive Safety Report")
    st.caption("Recursive LSTM forecasting for the next 60 minutes.")
    
    if st.button("Generate Forecast"):
        with st.spinner("Calculating future scenarios..."):
            # Reconstruct history input
            curr_hist = []
            for n in range(3):
                idx_h = st.session_state.idx - (2-n)
                if idx_h >= 0:
                    r = df.iloc[idx_h]
                    # Average wind calculation
                    w = (r.get('Wind Speed (m/sec)_z1',0) + 
                         r.get('Wind Speed (m/sec)_z2',0) + 
                         r.get('Wind Speed (m/sec)_z3',0)) / 3
                    curr_hist.append(w)
                else:
                    curr_hist.append(0.0)
                
            forecast_df = backend.get_forecast_data(curr_hist, steps=12)
            
            # Timestamp generation
            start_time = pd.to_datetime(current_row['Date'])
            forecast_df['Timestamp'] = [start_time + pd.Timedelta(minutes=(i+1)*5) for i in range(len(forecast_df))]
            
            # Display Table
            st.dataframe(
                forecast_df[['Timestamp', 'Predicted Wind (m/s)', 'Predicted Risk', 'Status']],
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn("Time", format="HH:mm"),
                    "Predicted Risk": st.column_config.ProgressColumn(
                        "Risk Probability", 
                        min_value=0, 
                        max_value=1, 
                        format="%.4f"
                    ),
                },
                use_container_width=True
            )
    else:
        st.info("Select 'Generate Forecast' to run the predictive engine.")

# === TAB 3: ANALYTICS ===
with tab_analytics:
    st.markdown("### Historical Data Analysis")
    
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        col_date1, col_date2 = st.columns(2)
        d_start = col_date1.date_input("Start Date", min_date)
        d_end = col_date2.date_input("End Date", max_date)
        
        if d_start <= d_end:
            mask = (df['Date'].dt.date >= d_start) & (df['Date'].dt.date <= d_end)
            df_filtered = df.loc[mask]
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Wind Velocity Distribution**")
                fig_hist = px.histogram(
                    df_filtered, 
                    x="Virtual_Wind", 
                    nbins=50, 
                    color_discrete_sequence=['#34495E'],
                    labels={"Virtual_Wind": "Wind Speed (m/s)"}
                )
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with c2:
                st.markdown("**Wind Rose (Direction vs Intensity)**")
                if 'Wind Direction (°)_z1' in df_filtered.columns:
                    fig_pol = px.scatter_polar(
                        df_filtered, 
                        r="Virtual_Wind", 
                        theta="Wind Direction (°)_z1",
                        color="Virtual_Wind", 
                        color_continuous_scale="Viridis",
                        labels={"Virtual_Wind": "Speed"}
                    )
                    st.plotly_chart(fig_pol, use_container_width=True)
                else:
                    st.warning("Wind Direction data unavailable.")
        else:
            st.error("Start date must be before end date.")
    else:
        st.warning("Date column missing from dataset.")

# === TAB 4: STRESS TEST LAB ===
with tab_stress:
    st.markdown("### Structural Stress Simulator")
    st.info("Manual overrides to test structural integrity limits under extreme conditions.")
    
    col_sim, col_res = st.columns([1, 2])
    
    with col_sim:
        st.markdown("**Parameters**")
        sim_wind = st.slider("Wind Velocity (m/s)", 0, 50, 10, key="stress_wind")
        sim_angle = st.slider("Incidence Angle (°)", 0, 360, 0, key="stress_angle")
        
        # Calculate Risk
        input_manual = pd.DataFrame({'angle': [sim_angle], 'velocity': [sim_wind]})
        sim_risk = backend.risk_model.predict(input_manual)[0]
        
        st.divider()
        st.metric("Simulated Risk Factor", f"{sim_risk:.4f}")
        
        if sim_risk > 0.8:
            st.error("STRUCTURAL FAILURE PREDICTED")
        elif sim_risk > 0.5:
            st.warning("CRITICAL LOAD DETECTED")
        else:
            st.success("INTEGRITY CONFIRMED")
            
    with col_res:
        st.markdown("**Vulnerability Curve**")
        
        # Generate Curve Data
        winds_range = np.linspace(0, 45, 50)
        risks_curve = backend.run_stress_test(winds_range, angle_fixed=sim_angle)
        
        fig_stress = go.Figure()
        
        # Risk Curve
        fig_stress.add_trace(go.Scatter(
            x=winds_range, 
            y=risks_curve, 
            mode='lines', 
            name='Risk Profile', 
            line=dict(color='#007bff', width=3)
        ))
        
        # Threshold Line
        fig_stress.add_hline(
            y=0.8, 
            line_dash="dash", 
            line_color="#dc3545", 
            annotation_text="Failure Limit"
        )
        
        # Current Point
        fig_stress.add_trace(go.Scatter(
            x=[sim_wind], 
            y=[sim_risk], 
            mode='markers', 
            marker=dict(color='#dc3545', size=15, symbol='x'), 
            name='Current Simulation'
        ))
        
        fig_stress.update_layout(
            title=f"Risk Analysis at {sim_angle}° Incidence", 
            xaxis_title="Wind Velocity (m/s)", 
            yaxis_title="Risk Probability", 
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_stress, use_container_width=True)

# === MAIN LOOP ===
if st.session_state.run and mode == "Live Stream":
    time.sleep(speed)
    st.session_state.idx += 1
    # Loop back if end of data reached
    if st.session_state.idx >= len(df)-5: 
        st.session_state.idx = 1000
    st.rerun()