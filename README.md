# CraneDigitalTwin

**Course Project for Digital Twins**
**University of Leoben | Semester 2025/2**

## 1. Project Overview

This project implements a **Hybrid Digital Twin** for a construction crane. The system is designed to monitor structural risk in real-time and predict future hazardous conditions caused by wind gusts.

The solution integrates two modeling approaches:
1.  **Physics-Based Modeling:** A Surrogate Model (Random Forest) trained on CFD simulations (ANSYS) to map wind velocity and angle to structural stress.
2.  **Data-Driven Forecasting:** A Time-Series Model (LSTM) to predict wind conditions 10-60 minutes into the future based on historical IoT sensor data.

---

## 2. Project Structure

The project has been organized to separate data, artifacts (models/figures), and source code logic.

```text
ICRERA/
│
├── main.py                   # MASTER SCRIPT: Executes the full End-to-End Pipeline
├── requirements.txt          # List of Python dependencies
├── README.md                 # Project documentation
│
├── data/                     # Data storage
│   ├── crane_digital_twin_ml_dataset.csv  # Raw IoT Sensor Data
│   ├── profiles/                          # Raw ANSYS CFD profiles (.prof files)
│   └── processed/                         # Cleaned data (Auto-generated)
│
├── models/                   # Trained Models (Auto-generated)
│   ├── surrogate_risk_model.pkl   # Physics Model
│   ├── lstm_wind_forecaster.keras # AI Forecast Model
│   └── scaler_wind.pkl            # Data Scaler
│
├── figures/                  # Validation Plots (Auto-generated)
│   ├── lstm_training_history.png
│   └── lstm_forecast_validation.png
│
└── src/                      # Source Code Modules (Logic)
    ├── __init__.py           # Package initializer
    ├── dashboard.py          # Streamlit Visualization Interface
    ├── digital_twin.py       # Backend Inference Engine
    ├── pipeline_setup.py     # Physics Model Training Logic
    ├── time_series_model.py  # LSTM Training Logic
    └── data_loader.py        # Data Cleaning Utilities

```

---

## 3. Installation & Setup

To run this project on a local machine, please follow these steps.

### Prerequisites

* **Python 3.9** or higher.
* **Pip** package manager.

### Step 1: Install Dependencies

Open your terminal in the root directory (`ICRERA`) and run:

```bash
pip install -r requirements.txt

```

*(Key libraries required: pandas, numpy, scikit-learn, tensorflow, streamlit, plotly, joblib)*

---

## 4. How to Run the Project

We have streamlined the workflow into a single **Master Pipeline** script that handles data cleaning, model training, and application launching.

**Execute the following command from the root folder:**

```bash
python main.py

```

### What happens when you run this?

1. **Data Ingestion:** The system loads the raw CSV, performs missing value imputation, calculates "Virtual Wind", and saves the clean dataset.
2. **Physics Training:** It parses ANSYS files, generates a lookup table, and trains the **Random Forest** surrogate model ().
3. **AI Training:** It trains the **LSTM Neural Network** on the wind history to predict future trends.
4. **Auto-Launch:** Once the pipeline finishes, it automatically opens the **Digital Twin Dashboard** in your web browser.

*(Note: If the dashboard does not launch automatically, you can run it manually with: `streamlit run src/dashboard.py`)*

---

## 5. Dashboard User Guide

The dashboard is divided into four main sections:

1. **Live Operations:**
* Real-time monitoring of wind speed and structural risk.
* 3D visualization of the crane with dynamic color coding (Green/Orange/Red).
* AI-powered short-term forecast (t+10 min).


2. **Forecast Report:**
* Generates a recursive prediction table for the next 60 minutes.


3. **Historical Analytics:**
* Tools to explore past data distributions and wind direction patterns (Wind Rose).


4. **Stress Test Lab ("The Curve of Death"):**
* A physics simulation environment.
* Allows the user to manually override wind speed and angle to validate the structural failure threshold (Risk > 0.8).



---

## 6. Technical Specifications

* **Rupture Pressure:** 355 Pa (Theoretical limit based on beam material).
* **Risk Calculation:** .
* **Time-Series Window:** The LSTM uses a look-back window of **3 time-steps**.
* **Decision Logic:**
* **SAFE:** Risk < 0.5
* **WARNING:** 0.5 ≤ Risk < 0.8
* **CRITICAL:** Risk ≥ 0.8




## 7. Authors

* Tulio Leandro De Castro
* Maria Eduarda Martins de Oliveira
* Andres Santiago Santafe Silva
