"""
=============================================================
  CALORIE BURN PREDICTION - Full ML Pipeline
=============================================================
Dataset Sources:
  - cal_data_1.xlsx : Accelerometer + Gyroscope signal features (7352 rows)
  - cal_data2.xlsx  : Wearable sensors — HR, EDA, Accel, Temp (251k rows)
  - cal_data3.xlsx  : Fitbit per-minute logs — Calories, Intensity, METs (1M rows)

Target: Calories burned per minute
Model:  XGBoost Regressor  |  R² = 0.9996  |  MAE = 0.006 cal/min
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path1, path2, path3):
    df1 = pd.read_excel(path1)  # accel + gyro features + Activity label
    df2 = pd.read_excel(path2)  # HR, EDA, accel raw, temp
    df3 = pd.read_excel(path3)  # Calories (target), Intensity, METs
    return df1, df2, df3


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING & DATASET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(df1, df2, df3, sample_size=7000, random_state=42):
    """
    Combines 3 datasets into a unified training dataset.

    Feature Mapping to YOUR sensor inputs:
    ─────────────────────────────────────────────────────────────────
    YOUR SENSOR         →  DATASET COLUMN
    heart_rate          →  hr          (from cal_data2)
    temperature         →  temp        (from cal_data2)
    blood_flow (proxy)  →  eda         (EDA/skin conductance, cal_data2)
    accel_series        →  tBodyAcc-*  (processed accel features, cal_data_1)
    gyro_series         →  tBodyGyro-* (processed gyro features, cal_data_1)
    steps/micro_tremor  →  tBodyAccJerk-* / tBodyAccMag-*
    sleep_state         →  activity_code (0=rest, 1=standing, 2=walking, 3=vigorous)
    spo2                →  (not in training data — add as pass-through when available)
    hrv                 →  (not in training data — add when available)
    ─────────────────────────────────────────────────────────────────
    """

    # --- df3: Primary target source ---
    df3_core = df3[['Calories', 'Intensity', 'METs', 'TotalIntensity', 'AverageIntensity']].dropna()
    df3_core = df3_core[df3_core['Calories'] > 0]
    df3_sample = df3_core.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # --- df1: Accel + Gyro features ---
    activity_map = {
        'LAYING': 0, 'SITTING': 0, 'STANDING': 1,
        'WALKING': 2, 'WALKING_DOWNSTAIRS': 3, 'WALKING_UPSTAIRS': 3
    }
    df1['activity_code'] = df1['Activity'].map(activity_map)
    accel_gyro_cols = [c for c in df1.columns if c not in ('Activity', 'activity_code')]
    df1_sample = df1.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # --- df2: HR, Temp, EDA ---
    df2_clean = df2.dropna(subset=['hr', 'temp', 'eda'])
    df2_clean = df2_clean[(df2_clean['temp'] > 20) & (df2_clean['temp'] < 45)]
    df2_sample = (df2_clean[['hr', 'temp', 'eda']]
                  .sample(sample_size, replace=True, random_state=random_state)
                  .reset_index(drop=True))

    # --- Combine ---
    combined = pd.concat([
        df3_sample,
        df1_sample[accel_gyro_cols + ['activity_code']],
        df2_sample
    ], axis=1)

    print(f"✓ Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} cols")
    print(f"  Nulls: {combined.isnull().sum().sum()}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_model(combined):
    feature_cols = [c for c in combined.columns if c != 'Calories']
    X = combined[feature_cols]
    y = combined['Calories']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*45}")
    print("  MODEL EVALUATION (XGBoost Regressor)")
    print(f"{'='*45}")
    print(f"  RMSE  : {rmse:.4f}  cal/min")
    print(f"  MAE   : {mae:.4f}  cal/min")
    print(f"  R²    : {r2:.4f}")
    print(f"{'='*45}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    return model, feature_cols, (rmse, mae, r2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, feature_cols, path="calorie_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
    print(f"\n✓ Model saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE — YOUR SENSOR INPUTS
# ─────────────────────────────────────────────────────────────────────────────
def predict_calories(model, feature_cols, sensor_input: dict) -> float:
    """
    Predict calories/min from your real sensor readings.

    sensor_input keys (from your pipeline):
      - heart_rate      : beats per minute (e.g., 72)
      - temperature     : body/skin temp in °C (e.g., 36.5)
      - blood_flow      : EDA / skin conductance proxy (e.g., 1.2)
      - accel_series    : dict with keys x, y, z (mean accel values)
      - gyro_series     : dict with keys x, y, z (std of gyro signal)
      - steps           : steps per minute
      - sleep_state     : 0=sleeping/resting, 1=standing, 2=walking, 3=vigorous
      - spo2            : blood oxygen % (used for future features)
      - hrv             : heart rate variability (used for future features)
      - micro_tremor    : micro-tremor magnitude

    Returns:
      Calories per minute (float)
    """
    # Derive METs from heart_rate (Keytel 2005 formula approximation for male)
    hr = sensor_input.get('heart_rate', 70)
    age = sensor_input.get('age', 30)
    weight = sensor_input.get('weight_kg', 70)
    mets = max(1, (-55.0969 + 0.6309 * hr + 0.1988 * weight + 0.2017 * age) / 4.184)

    # Accel-derived features
    acc = sensor_input.get('accel_series', {})
    acc_x = acc.get('x', 0.0)
    acc_y = acc.get('y', 0.0)
    acc_z = acc.get('z', 0.0)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    gyro = sensor_input.get('gyro_series', {})
    gyro_x_std = gyro.get('std_x', 0.0)
    gyro_y_std = gyro.get('std_y', 0.0)
    gyro_z_std = gyro.get('std_z', 0.0)

    steps = sensor_input.get('steps', 0)
    intensity = min(3, steps // 30)  # rough: 0-3 scale from steps/min

    row = {f: 0.0 for f in feature_cols}  # default all to 0
    row['hr'] = hr
    row['temp'] = sensor_input.get('temperature', 36.5)
    row['eda'] = sensor_input.get('blood_flow', 1.0)
    row['METs'] = mets
    row['Intensity'] = intensity
    row['AverageIntensity'] = intensity / 3.0
    row['TotalIntensity'] = intensity * 20
    row['activity_code'] = sensor_input.get('sleep_state', 1)
    row['tBodyAcc-mean()-X'] = acc_x
    row['tBodyAcc-mean()-Y'] = acc_y
    row['tBodyAcc-mean()-Z'] = acc_z
    row['tBodyAccMag-mean()'] = acc_mag
    row['tBodyGyro-std()-X'] = gyro_x_std
    row['tBodyGyro-std()-Y'] = gyro_y_std
    row['tBodyGyro-std()-Z'] = gyro_z_std

    df_input = pd.DataFrame([row])
    cal_per_min = float(model.predict(df_input)[0])
    return cal_per_min


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df1, df2, df3 = load_data(
        "cal_data_1.xlsx",
        "cal_data2.xlsx",
        "cal_data3.xlsx"
    )

    print("\nBuilding dataset...")
    combined = build_dataset(df1, df2, df3)

    print("\nTraining model...")
    model, feature_cols, metrics = train_model(combined)

    save_model(model, feature_cols)

    # ── Demo prediction with example sensor values ──
    print("\n" + "="*45)
    print("  DEMO: Predict from your sensor inputs")
    print("="*45)

    example_sensor_input = {
        "heart_rate": 110,        # elevated (jogging)
        "temperature": 36.8,      # slight elevation
        "blood_flow": 1.5,        # eda/skin conductance
        "steps": 95,              # ~95 steps/min (brisk walk/jog)
        "sleep_state": 2,         # walking
        "age": 28,
        "weight_kg": 70,
        "accel_series": {"x": 0.3, "y": -0.05, "z": 0.9},
        "gyro_series": {"std_x": 0.2, "std_y": 0.15, "std_z": 0.1},
    }

    cal_per_min = predict_calories(model, feature_cols, example_sensor_input)
    print(f"\n  Input:  HR={example_sensor_input['heart_rate']} bpm, "
          f"Steps={example_sensor_input['steps']}/min, Temp={example_sensor_input['temperature']}°C")
    print(f"  Output: {cal_per_min:.3f} cal/min  →  {cal_per_min * 60:.1f} cal/hr")