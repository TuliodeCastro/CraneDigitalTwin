

Based on the `EDA.ipynb` file and your requirements, here is the roadmap to build the Machine Learning model and integrate the crane simulation data.

### **Strategy: Weather Forecasting & Risk Assessment**

Since you need to forecast weather at a specific point (the "Middle Zone") where you likely don't have a physical sensor, we will create a **"Virtual Sensor"** using data from stations Z1, Z2, and Z3.

-----

### **Phase 1: Data Engineering (Preparing the Dataset)**

Before training, we need to transform the data you visualized in the EDA into a format the model can understand.

**Task 1.1: Create the "Virtual Station" (The Target)**
Since the crane is in the middle, we need to estimate the *ground truth* weather there to train the model.

  * **Action:** Calculate the weighted average of Wind Speed and Gust from Z1, Z2, and Z3 based on their distance to the crane.
  * **Formula (Inverse Distance Weighting):**
    $$Wind_{middle} = \frac{\sum (Wind_i / Distance_i)}{\sum (1 / Distance_i)}$$
    *(Where $i$ is Z1, Z2, Z3)*.
  * **Result:** A new column `Wind_Speed_Target` in your dataframe.

**Task 1.2: Feature Engineering (The Inputs)**
The model needs to know the *past* to predict the *future*.

  * **Lag Features:** Create columns for past values (e.g., `Wind_Speed_Z1_t-5min`, `Wind_Speed_Z1_t-10min`).
  * **Time Features:** Extract `Hour`, `Month`, and `DayOfWeek` from the `Date` column (the EDA shows `Date` is already datetime format).
  * **Crane Constraints:** Add columns for crane limits (e.g., `Max_Safe_Wind_Speed` from your simulation data) as static features or thresholds.

-----

### **Phase 2: Machine Learning Modeling**

We will build two components: a **Forecaster** (predicts numbers) and a **Classifier** (predicts safety).

**Task 2.1: The Forecasting Model (Regressor)**

  * **Goal:** Predict wind speed at the Middle Point for the next 2-3 hours.
  * **Algorithm:** **XGBoost Regressor** or **Random Forest Regressor** and LSTM. These handle non-linear weather patterns better than simple regression.
  * **Input ($X$):** Current wind, gust, direction, temperature, pressure (from Z1, Z2, Z3) + Time features.
  * **Output ($y$):** `Wind_Speed_Target` (at time $t+10$).

**Task 2.2: The Safety Model (Rule-Based or Classifier)**

  * **Goal:** Output a binary "Safe / Unsafe" signal or a Risk Level (Low/Medium/High).
  * **Logic:**
      * Take the *Predicted Wind Speed* from Task 2.1.
      * Compare it against the *Crane Simulation Limits*.
      * *Example:* If `Predicted_Gust` \> `Crane_Limit_20m_Height`, then `Status = UNSAFE`.

-----

### **Phase 3: Code Structure for Your Notebook**

You can add these steps directly after your EDA. Here is a Python template to guide you:

```python
# 1. Feature Engineering
# Create a target variable (Virtual Middle Point)
# Assuming equal distance for simplicity, or use specific weights
df['Wind_Speed_Middle'] = (df['Wind Speed (m/sec)_z1'] +
                           df['Wind Speed (m/sec)_z2'] +
                           df['Wind Speed (m/sec)_z3']) / 3

# Create Target: Shift data to predict 10 mins into the future
# Assuming 1 row = 1 minute data
df['Target_Wind_10min'] = df['Wind_Speed_Middle'].shift(-10)

# Drop rows with NaN (the last 10 mins)
df_ml = df.dropna()

# 2. Select Features
features = [
    'Wind Speed (m/sec)_z1', 'Wind Direction (°)_z1',
    'Wind Speed (m/sec)_z2', 'Wind Direction (°)_z2',
    'Wind Speed (m/sec)_z3', 'Wind Direction (°)_z3',
    'Outdoor Temperature (°C)_z1', 'Absolute Pressure (mmHg)_z1'
]

X = df_ml[features]
y = df_ml['Target_Wind_10min']

# 3. Train Model
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f} m/s")

# 5. Safety Logic (Integration with Crane Data)
# Example threshold from crane specs
CRANE_LIMIT = 15.0 # m/s

def get_safety_status(pred_speed):
    if pred_speed >= CRANE_LIMIT:
        return "CRITICAL RISK"
    elif pred_speed >= CRANE_LIMIT * 0.8:
        return "WARNING"
    else:
        return "SAFE"

# Test on a few predictions
print(f"Predicted: {predictions[0]:.2f} m/s -> Status: {get_safety_status(predictions[0])}")
```

-----

### **Phase 4: Export for Dashboard (Web)**

To connect this to your future dashboard with the 3D part:

**Task 4.1: Save the Model**

  * Use `joblib` or `pickle` to save the trained model file (e.g., `wind_predictor.pkl`).

**Task 4.2: Create an Inference Script**

  * Write a small Python script that:
    1.  Receives current live data from the API (Z1, Z2, Z3).
    2.  Loads `wind_predictor.pkl`.
    3.  Predicts the wind for the next 10 mins.
    4.  Returns a JSON object: `{'forecast': 12.5, 'status': 'UNSAFE', 'coordinates': [x, y, z]}`.

**Task 4.3: 3D Visualization Data**

  * For the 3D web part, your ML output acts as the "driver." If ML says "High Wind," the 3D crane in the dashboard should visually change color (e.g., to red) or show a warning animation.


Here is the comprehensive step-by-step guide in English, designed to be built from scratch and split between two developers.

### **Project Architecture: The "Virtual Sensor" Strategy**

Since there is no physical sensor in the middle of the crane site, we will create a **Virtual Sensor** using the data from stations Z1, Z2, and Z3.

  * **Input:** Data from Z1, Z2, Z3 (Wind, Gust, Direction).
  * **Virtual Target:** Calculated using **Inverse Distance Weighting (IDW)**. The closer a station is to the crane, the more influence it has.
  * **Model:** **LSTM (Long Short-Term Memory)** to predict the future values of this Virtual Sensor.
  * **Output:** Safety Status (Safe/Warning/Danger) for the web dashboard.

-----

### **Team Roles & Tasks**

#### **👷 Person A: Data Engineer (The Architect)**

**Focus:** Data reliability, math, and preparing the "tensors" for the neural network.

  * **Task A.1 (Coordinate System):** Define the $(x, y)$ coordinates for the 3 stations and the Crane.
  * **Task A.2 (The IDW Algorithm):** Write the function that calculates the weighted average for the "Middle Point" (Virtual Sensor). This is your "Ground Truth" for training.
  * **Task A.3 (Data Cleaning & Smoothing):** Ensure there are no `NaN` values (fill with interpolation). LSTM crashes with empty data.
  * **Task A.4 (Sequence Generation):** Create the 3D arrays required for LSTM: `(Samples, Time_Steps, Features)`.

#### **👩‍💻 Person B: ML Engineer (The Brain)**

**Focus:** Neural network architecture, training, and logic.

  * **Task B.1 (LSTM Architecture):** Design the layers (LSTM, Dropout, Dense).
  * **Task B.2 (Training Loop):** Handle the training process, validation split, and avoid overfitting.
  * **Task B.3 (Safety Logic):** Define the thresholds (e.g., if Wind \> 15m/s $\rightarrow$ Unsafe).
  * **Task B.4 (Saving):** Save the model as `.h5` or `.keras` for the web dashboard.

-----

### **The Master Code: `crane_forecast_model.py`**

Here is the complete code structure. You can create this file and work on it simultaneously (using Git) or split the functions.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib  # To save the scaler

# ==========================================
# CONFIGURATION (Both Agree on this)
# ==========================================
CONFIG = {
    'SEQ_LENGTH': 30,       # Look back 30 minutes
    'PREDICT_HORIZON': 10,  # Predict 10 minutes into the future
    'CRANE_LIMIT_MS': 15.0, # Max safe wind speed (m/s)
    'COORDS': {
        'z1': np.array([0, 0]),    # Example coordinates
        'z2': np.array([100, 0]),
        'z3': np.array([50, 100]),
        'crane': np.array([50, 50]) # Middle point
    }
}

# ==========================================
# 👷 PERSON A: DATA ENGINEERING
# ==========================================

def calculate_idw_target(df, coords):
    """
    Calculates the 'Virtual Sensor' at the crane location using 
    Inverse Distance Weighting (IDW).
    """
    print("👷 (Person A) Calculating Virtual Sensor (IDW)...")
    
    # 1. Calculate Distances
    d1 = np.linalg.norm(coords['crane'] - coords['z1'])
    d2 = np.linalg.norm(coords['crane'] - coords['z2'])
    d3 = np.linalg.norm(coords['crane'] - coords['z3'])
    
    # 2. Calculate Weights (Inverse Distance)
    # Add small epsilon to avoid division by zero
    w1 = 1 / (d1 + 1e-6)
    w2 = 1 / (d2 + 1e-6)
    w3 = 1 / (d3 + 1e-6)
    total_w = w1 + w2 + w3
    
    # 3. Compute Weighted Average for Wind Speed
    # NOTE: Update these column names to match your CSV exactly!
    df['Virtual_Wind_Speed'] = (
        (df['Wind Speed (m/sec)_z1'] * w1) + 
        (df['Wind Speed (m/sec)_z2'] * w2) + 
        (df['Wind Speed (m/sec)_z3'] * w3)
    ) / total_w
    
    return df

def create_sequences(data, seq_length, horizon):
    """
    Converts 2D data into 3D sequences for LSTM.
    X: [t-30, ..., t]
    y: [t + horizon]
    """
    print("👷 (Person A) Creating 3D Sequences for LSTM...")
    X, y = [], []
    for i in range(len(data) - seq_length - horizon):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length + horizon, 0]) # Assuming col 0 is target
    return np.array(X), np.array(y)

def prepare_data(file_path):
    # Load
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').fillna(method='ffill') # Safe fill
    
    # Feature Engineering (Person A)
    df = calculate_idw_target(df, CONFIG['COORDS'])
    
    # Select Features for Model
    # Target (Virtual_Wind) MUST be first for easier indexing
    feature_cols = [
        'Virtual_Wind_Speed', 
        'Wind Speed (m/sec)_z1', 'Wind Direction (°)_z1',
        'Wind Speed (m/sec)_z2', 'Wind Direction (°)_z2',
        'Wind Speed (m/sec)_z3', 'Wind Direction (°)_z3'
    ]
    
    dataset = df[feature_cols].values
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Sequence Generation
    X, y = create_sequences(scaled_data, CONFIG['SEQ_LENGTH'], CONFIG['PREDICT_HORIZON'])
    
    return X, y, scaler

# ==========================================
# 👩‍💻 PERSON B: MACHINE LEARNING MODEL
# ==========================================

def build_lstm_model(input_shape):
    """
    Defines the Neural Network architecture.
    """
    print("👩‍💻 (Person B) Building LSTM Architecture...")
    model = Sequential([
        # Layer 1: LSTM
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), # Prevent overfitting
        
        # Layer 2: LSTM
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Layer 3: Output
        Dense(16, activation='relu'),
        Dense(1) # Output: Predicted Wind Speed
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def determine_safety(predicted_speed):
    """
    Business Logic for the Dashboard.
    """
    if predicted_speed >= CONFIG['CRANE_LIMIT_MS']:
        return "🔴 DANGER: STOP OPERATIONS"
    elif predicted_speed >= CONFIG['CRANE_LIMIT_MS'] * 0.8:
        return "🟡 WARNING: REDUCE LOAD"
    else:
        return "🟢 SAFE TO OPERATE"

# ==========================================
# MAIN EXECUTION (Run this to train)
# ==========================================
if __name__ == "__main__":
    # 1. Prepare Data
    csv_path = 'data/prepared/zones_combined_prepared.csv' # Adjust path
    X, y, scaler = prepare_data(csv_path)
    
    # 2. Split Data (No shuffling for Time Series!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. Build & Train (Person B)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5)]
    )
    
    # 4. Save Everything for Web Dashboard
    model.save('crane_lstm_model.h5')
    joblib.dump(scaler, 'scaler.gz')
    print("\n✅ Model and Scaler saved successfully!")
    
    # 5. Test Simulation (What the dashboard will do)
    print("\n--- DASHBOARD SIMULATION ---")
    last_sequence = X_test[-1].reshape(1, CONFIG['SEQ_LENGTH'], -1)
    predicted_scaled = model.predict(last_sequence)
    
    # Inverse transform (trick: create dummy array to inverse transform only target)
    dummy = np.zeros((1, X_test.shape[2]))
    dummy[0, 0] = predicted_scaled[0, 0]
    predicted_real = scaler.inverse_transform(dummy)[0, 0]
    
    status = determine_safety(predicted_real)
    
    print(f"Predicted Wind (Next 10 min): {predicted_real:.2f} m/s")
    print(f"System Status: {status}")
```

### **Next Steps for the Web Dashboard**

Once this script runs successfully:

1.  **Backend (Python/Flask/FastAPI):** Will load `crane_lstm_model.h5`.
2.  **Input:** It will receive live data from Z1, Z2, Z3.
3.  **Process:** It will run `calculate_idw_target` $\rightarrow$ `scaler.transform` $\rightarrow$ `model.predict`.
4.  **Frontend (3D):** If the result is "🔴 DANGER", the 3D crane turns red.