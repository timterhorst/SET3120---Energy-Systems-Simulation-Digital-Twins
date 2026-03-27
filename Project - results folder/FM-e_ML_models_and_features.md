# Focus Module e: Forecasting Module
## ML Model Selection & Feature Engineering

---

## PART 1: ML MODEL OPTIONS

### A. Load Forecasting (Household Consumption)

Target variable: `P_load[t]` [kW] — total electrical load at time t

---

#### **Option 1: Random Forest Regressor** ⭐ RECOMMENDED

**Description:**
Ensemble of decision trees trained on bootstrapped samples. Each tree votes on prediction, final output is average.

**Advantages:**
- ✅ Handles non-linear relationships well (heat pump ON/OFF switching, driver behavior)
- ✅ Robust to outliers and missing data
- ✅ No feature scaling required
- ✅ Built-in feature importance → interpretable model
- ✅ Low risk of overfitting (averaging reduces variance)
- ✅ Works well with moderate dataset sizes (~1 year hourly data)

**Disadvantages:**
- ❌ Cannot extrapolate beyond training range
- ❌ Larger model size (ensemble of trees)
- ❌ Slower inference than linear models

**Hyperparameters to tune:**
- `n_estimators`: 100-500 (number of trees)
- `max_depth`: 10-30 (tree depth)
- `min_samples_split`: 2-10
- `max_features`: 'sqrt' or 'log2'

**When to use:**
Best for complex consumption patterns with multiple interacting factors (temperature, time, driver behavior, etc.)

---

#### **Option 2: Gradient Boosting (XGBoost / LightGBM)**

**Description:**
Sequential ensemble — each new tree corrects errors of previous trees. More sophisticated than Random Forest.

**Advantages:**
- ✅ Often 2-5% more accurate than Random Forest
- ✅ Excellent for tabular data (industry standard for competitions)
- ✅ Fast training with LightGBM implementation
- ✅ Handles missing values natively
- ✅ Feature importance available

**Disadvantages:**
- ❌ More hyperparameters to tune (learning rate, max depth, boosting rounds)
- ❌ Higher risk of overfitting if not tuned carefully
- ❌ Requires more computational resources
- ❌ Slightly less interpretable than Random Forest

**Hyperparameters to tune:**
- `learning_rate`: 0.01-0.3
- `n_estimators`: 100-1000
- `max_depth`: 3-10
- `subsample`: 0.7-1.0

**When to use:**
When you have time for hyperparameter optimization and want maximum accuracy.

---

#### **Option 3: LSTM (Long Short-Term Memory Neural Network)**

**Description:**
Recurrent neural network that learns temporal dependencies in sequences. State-of-the-art for time series.

**Advantages:**
- ✅ Can learn long-term dependencies (e.g., weekly patterns)
- ✅ No manual feature engineering needed (learns features)
- ✅ Handles variable-length sequences
- ✅ State-of-the-art performance on large datasets

**Disadvantages:**
- ❌ Requires MUCH more training data (>3 years for good performance)
- ❌ Slow to train (GPU recommended)
- ❌ Black box — difficult to interpret
- ❌ Sensitive to hyperparameters (sequence length, hidden units, dropout)
- ❌ Overkill for this project scope

**Architecture example:**
```
Input: [batch_size, sequence_length=24, n_features]
LSTM(64 units) → Dropout(0.2) → LSTM(32 units) → Dense(1)
Output: P_load[t+1]
```

**When to use:**
Only if you have >3 years of high-frequency data and GPU resources. Not recommended for midterm project.

---

### B. PV Generation Forecasting

Target variable: `P_PV[t]` [kW] — photovoltaic power output at time t

---

#### **Option 1: Linear Regression** ⭐ RECOMMENDED

**Description:**
Simple linear model: `P_PV = β₀ + β₁×hour + β₂×day_of_year + β₃×solar_angle + ε`

**Advantages:**
- ✅ PV output is physically deterministic (solar irradiance → power)
- ✅ Fast training and inference
- ✅ Very interpretable (coefficients have physical meaning)
- ✅ Works well with limited data
- ✅ Easy to add clear-sky baseline

**Disadvantages:**
- ❌ Cannot capture complex weather patterns (clouds)
- ❌ Assumes linear relationship (OK for aggregate daily patterns)

**Features:**
```python
- hour_sin = sin(2π × hour / 24)
- hour_cos = cos(2π × hour / 24)
- day_of_year
- solar_elevation_angle = f(latitude, longitude, hour, day)
- T_outside (optional, slight temperature effect on panel efficiency)
```

**When to use:**
As baseline model. Add weather features (cloud cover) if available for improvement.

---

#### **Option 2: Random Forest Regressor**

**Description:**
Same as for load forecasting, applied to PV generation.

**Advantages:**
- ✅ Can capture non-linear weather effects
- ✅ Handles cloud patterns better than linear regression
- ✅ Works if you include cloud cover / irradiance features

**Disadvantages:**
- ❌ More complex than needed for clear-sky predictions
- ❌ Requires weather forecast data for day-ahead predictions

**When to use:**
If you have access to weather forecast API (cloud cover, GHI data) and want higher accuracy.

---

#### **Option 3: Hybrid Physics + ML Model**

**Description:**
Combine physics-based clear-sky model with ML correction factor:
```
P_PV_pred = P_clear_sky × correction_factor(weather_features)
```

**Advantages:**
- ✅ Physically grounded baseline (clear-sky)
- ✅ ML learns only the deviations (clouds, soiling)
- ✅ Generalizes better with limited data
- ✅ Interpretable: separate physics from data-driven adjustment

**Disadvantages:**
- ❌ Requires implementing clear-sky model (e.g., Ineichen-Perez)
- ❌ More complex pipeline

**When to use:**
If you want to showcase physics-informed ML (good for academic report).

---

## PART 2: FEATURE ENGINEERING

### A. Raw Features (Direct from Data)

#### **Temporal Features**
```python
- timestamp: datetime (index)
- hour: 0-23 (extracted from timestamp)
- day_of_week: 0-6 (Monday=0, Sunday=6)
- month: 1-12
- day_of_year: 1-365
- is_weekend: boolean (Saturday=5, Sunday=6)
- season: 0-3 (Winter/Spring/Summer/Fall)
```

#### **Weather Features** (if available)
```python
- T_outside: [°C] outdoor temperature
- wind_speed: [m/s]
- cloud_cover: [0-100%] or categorical (clear/partly/overcast)
- GHI: [W/m²] Global Horizontal Irradiance (for PV)
- precipitation: [mm]
```

#### **Load Features** (measurements)
```python
- P_load: [kW] total electrical load
- P_heat_pump: [kW] heat pump consumption
- P_base_load: [kW] base load (appliances, lights)
- P_EV_charge: [kW] EV charging power
```

#### **Driver Behavior Features** (from FM-c)
```python
- is_home: boolean (True if car/driver at home)
- T_setpoint_pref: [°C] dynamic setpoint (15 or 20)
- departure_time: [hours] time of departure (if predictable)
- arrival_time: [hours] time of arrival
```

---

### B. Engineered Features

#### **Cyclical Encoding** (for temporal features)

Hours and days are cyclical (23:00 is close to 00:00). Use sin/cos encoding:

```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

day_sin = np.sin(2 * np.pi * day_of_year / 365)
day_cos = np.cos(2 * np.pi * day_of_year / 365)
```

**Why:** Preserves cyclical nature. ML models see (sin, cos) pairs instead of linear 0-23.

---

#### **Lagged Features** (autoregressive)

Use past values to predict future:

```python
load_lag_1h = P_load[t-1]           # 1 hour ago
load_lag_24h = P_load[t-24]         # Same hour yesterday
load_lag_168h = P_load[t-168]       # Same hour last week (7×24=168)

pv_lag_24h = P_PV[t-24]             # Yesterday's PV at same hour
```

**Why:** Load/PV often follow daily and weekly patterns. Yesterday's value is strong predictor.

---

#### **Rolling Statistics** (moving averages)

```python
load_rolling_mean_24h = P_load[t-24:t].mean()
load_rolling_std_24h = P_load[t-24:t].std()
load_rolling_max_24h = P_load[t-24:t].max()
```

**Why:** Captures recent trends and volatility.

---

#### **Heating Degree Days (HDD)**

```python
T_base = 18  # °C, balance temperature
HDD = max(0, T_base - T_outside)
```

**Why:** Heat pump load strongly correlated with HDD (how much heating is needed).

---

#### **Solar Features** (for PV forecasting)

```python
solar_elevation_angle = calculate_solar_position(latitude, longitude, timestamp)
solar_azimuth = calculate_solar_azimuth(latitude, longitude, timestamp)

# Clear-sky baseline (Ineichen-Perez model or simpler)
P_PV_clear_sky = P_max × max(0, sin(solar_elevation_angle))
```

**Why:** PV output is directly driven by sun position. Clear-sky is strong baseline.

---

#### **Interaction Features**

Combine two features to capture joint effects:

```python
hour_x_is_home = hour × is_home
# Example: Load at 18:00 is different when home (cooking) vs. away

weekend_x_hour = is_weekend × hour
# Example: Weekend mornings have different load patterns than weekdays

T_outside_x_is_home = T_outside × is_home
# Heating load depends on both temperature AND whether anyone is home
```

**Why:** Captures non-additive effects (interaction terms).

---

#### **Binary Flags**

```python
is_heating_season = (month >= 10) | (month <= 3)  # Oct-Mar
is_daytime = (hour >= 6) & (hour <= 20)
is_peak_hour = (hour >= 17) & (hour <= 21)  # Evening peak
```

**Why:** Helps model learn distinct behavior in different regimes.

---

#### **Derived Load Metrics**

```python
load_per_degree = P_heat_pump / (T_setpoint - T_outside)
# Efficiency metric: how much power per degree of heating

pv_capacity_factor = P_PV / P_PV_max
# PV performance relative to maximum (captures clouds, soiling)
```

**Why:** Normalizes for conditions, helps model generalize.

---

### C. Feature Selection Strategy

Not all features improve the model. Use these techniques:

**1. Correlation Analysis**
```python
correlation_matrix = df.corr()
# Remove features with |corr| < 0.05 with target
```

**2. Feature Importance (Random Forest)**
```python
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
# Keep top 20-30 most important features
```

**3. Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFE
selector = RFE(estimator, n_features_to_select=20)
```

**4. Domain Knowledge**
Always keep features you KNOW matter physically (hour, temperature, is_home).

---

## PART 3: RECOMMENDED FEATURE SET

### For Load Forecasting:

**Essential (always include):**
- hour_sin, hour_cos
- day_of_week
- is_weekend
- T_outside
- is_home
- load_lag_24h
- load_lag_168h

**Recommended (add if available):**
- month
- HDD (heating degree days)
- load_rolling_mean_24h
- hour × is_home (interaction)
- T_setpoint_pref

**Advanced (optional):**
- load_lag_1h (for very short-term forecasting)
- weekend × hour
- is_heating_season
- load_rolling_std_24h

**Total: 12-18 features**

---

### For PV Forecasting:

**Essential (always include):**
- hour_sin, hour_cos
- day_of_year
- solar_elevation_angle
- P_PV_clear_sky (baseline)

**Recommended (add if weather data available):**
- cloud_cover
- GHI (Global Horizontal Irradiance)
- T_outside (panel efficiency effect)
- pv_lag_24h

**Advanced (optional):**
- solar_azimuth
- pv_capacity_factor (derived)
- season

**Total: 5-12 features**

---

## PART 4: IMPLEMENTATION NOTES

### Training Pipeline

```python
# 1. Load historical data
df = load_simulation_data('simulation_results_1year.csv')

# 2. Feature engineering
df = add_temporal_features(df)
df = add_lagged_features(df, lags=[1, 24, 168])
df = add_rolling_features(df, window=24)
df = add_engineered_features(df)

# 3. Train/test split (chronological!)
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# 4. Train model
model = RandomForestRegressor(n_estimators=200, max_depth=20)
model.fit(train[features], train['P_load'])

# 5. Evaluate
predictions = model.predict(test[features])
mae = mean_absolute_error(test['P_load'], predictions)
```

---

## SUMMARY TABLE

| Forecast Type | Recommended Model | Key Features | Expected MAE |
|---------------|-------------------|--------------|--------------|
| Load | Random Forest | Temporal, Lagged, is_home, T_outside | 0.2-0.5 kW |
| PV | Linear Regression | Solar angle, hour, day_of_year | 0.1-0.3 kW |

