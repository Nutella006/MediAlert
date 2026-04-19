"""
=============================================================
  STRESS LEVEL PREDICTION - ML Pipeline v2
  (Trained on actual stress physiological dataset)
=============================================================
Dataset Sources:
  - stress_data1.xlsx : Accel + Gyro signal features (7352 rows)
  - stress_data2.xlsx : Wearable sensors — HR, EDA, TEMP, IBI (251k rows)
  - stress_data3.xlsx : Sleep & activity context (1M rows)

Target: Stress Level  →  0 = Low  |  1 = Medium  |  2 = High
Model:  XGBoost Classifier
        Accuracy = 96.38%  |  F1 = 0.9637

Stress Label Engineering (physiological basis):
  - EDA (electrodermal activity)  → 50% weight  ← strongest stress marker
  - Heart Rate elevated            → 25% weight
  - HRV low (inverted IBI std)    → 15% weight  ← NEW: real HRV from IBI
  - Skin temperature               → 10% weight
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path1, path2, path3):
    df1 = pd.read_excel(path1)   # accel + gyro features
    df2 = pd.read_excel(path2)   # HR, EDA, TEMP, IBI (wearable)
    df3 = pd.read_excel(path3)   # sleep + activity context
    return df1, df2, df3


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENGINEER STRESS LABELS
# ─────────────────────────────────────────────────────────────────────────────
def engineer_stress_labels(df2_clean):
    """
    Physiologically-grounded stress score from real wearable signals:

    Signal         Weight   Why
    ─────────────────────────────────────────────────────
    EDA            50%      Sympathetic NS activation (sweat glands)
    HR             25%      Elevated HR without physical exertion = stress
    HRV (inv.)     15%      Low HRV = high stress (autonomic imbalance)
    Skin Temp      10%      Subtle vasoconstriction/dilation response

    Score → quantile-split into 3 balanced classes (Low / Med / High)
    """
    df = df2_clean.copy()
    df['HRV'] = df['IBI'].rolling(5, min_periods=1).std().fillna(0)
    df['hrv_stress'] = -df['HRV']  # invert: low HRV = high stress

    df['stress_score'] = (
        (df['EDA']        - df['EDA'].mean())        / (df['EDA'].std() + 1e-9)        * 0.50 +
        (df['HR']         - df['HR'].mean())          / (df['HR'].std() + 1e-9)         * 0.25 +
        (df['hrv_stress'] - df['hrv_stress'].mean())  / (df['hrv_stress'].std() + 1e-9) * 0.15 +
        (df['TEMP']       - df['TEMP'].mean())        / (df['TEMP'].std() + 1e-9)       * 0.10
    )
    df['stress_level'] = pd.qcut(df['stress_score'], q=3, labels=[0, 1, 2]).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(df1, df2, df3):
    """
    Feature Mapping to YOUR sensor inputs:
    ─────────────────────────────────────────────────────────────────
    YOUR SENSOR         →  FEATURE
    heart_rate          →  HR
    temperature         →  TEMP (skin temperature)
    blood_flow / EDA    →  EDA   ← TOP stress indicator (50% weight)
    hrv                 →  HRV   (computed from IBI rolling std)
    ir_signal / IBI     →  IBI   (inter-beat interval from IR sensor)
    accel_series        →  accel_mag + tBodyAcc-* features
    gyro_series         →  tBodyGyro-* features
    micro_tremor        →  tBodyAccJerk-* features
    sleep_state         →  SedentaryMinutes / VeryActiveMinutes proxy
    spo2                →  (future — add when labeled data available)
    ─────────────────────────────────────────────────────────────────
    """
    # Clean df2
    df2_clean = df2.dropna(subset=['HR', 'TEMP', 'EDA', ' IBI'])
    df2_clean = df2_clean[(df2_clean['TEMP'] > 20) & (df2_clean['TEMP'] < 45)].copy()
    df2_clean.rename(columns={' IBI': 'IBI'}, inplace=True)
    df2_clean['accel_mag'] = np.sqrt(
        df2_clean['ACC X']**2 + df2_clean['ACC Y']**2 + df2_clean['ACC Z']**2
    )
    df2_clean = engineer_stress_labels(df2_clean)

    n = len(df2_clean)

    # df1: accel + gyro features
    df1_sample = df1.sample(n, replace=True, random_state=42).reset_index(drop=True)

    # df3: sleep and activity context
    df3_clean = df3[['TotalIntensity', 'AverageIntensity',
                     'VeryActiveMinutes', 'SedentaryMinutes']].dropna()
    df3_sample = df3_clean.sample(n, replace=True, random_state=42).reset_index(drop=True)

    core_cols = ['HR', 'TEMP', 'EDA', 'IBI', 'HRV', 'accel_mag', 'stress_level']
    combined = pd.concat([
        df2_clean[core_cols].reset_index(drop=True),
        df1_sample.reset_index(drop=True),
        df3_sample.reset_index(drop=True)
    ], axis=1)

    print(f"✓ Dataset: {combined.shape[0]} rows × {combined.shape[1]} cols | Nulls: {combined.isnull().sum().sum()}")
    print(f"  Stress distribution: {combined['stress_level'].value_counts().sort_index().to_dict()}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_model(combined):
    feature_cols = [c for c in combined.columns if c != 'stress_level']
    X = combined[feature_cols]
    y = combined['stress_level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'='*45}")
    print("  STRESS MODEL v2 — XGBoost Classifier")
    print(f"{'='*45}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred,
          target_names=['Low Stress', 'Medium Stress', 'High Stress']))

    imp = pd.DataFrame({'feature': feature_cols,
                        'importance': model.feature_importances_}
                       ).sort_values('importance', ascending=False)
    print("Top 10 Features:")
    print(imp.head(10).to_string(index=False))

    return model, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, feature_cols, path="stress_model_v2.pkl"):
    with open(path, "wb") as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
    print(f"\n✓ Model saved to: {path}")

def load_model(path="stress_model_v2.pkl"):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved['model'], saved['features']


# ─────────────────────────────────────────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
STRESS_LABELS = {0: "😌 Low Stress", 1: "😐 Medium Stress", 2: "😰 High Stress"}

def predict_stress(model, feature_cols, sensor_input: dict) -> dict:
    """
    Predict stress from real sensor readings.

    sensor_input keys:
      heart_rate     : bpm (e.g., 88)
      temperature    : skin temp °C (e.g., 36.2)
      blood_flow     : EDA / skin conductance (e.g., 1.5)
      hrv            : HRV in seconds (e.g., 0.05) — from IBI std
      ir_signal      : IBI value from IR sensor (e.g., 0.85)
      accel_series   : dict with x, y, z
      gyro_series    : dict with std_x, std_y, std_z
      micro_tremor   : float (AccJerk proxy)
      steps          : steps/min
      sleep_state    : 0=resting, 1=standing, 2=walking, 3=vigorous
    """
    hr   = sensor_input.get('heart_rate', 72)
    temp = sensor_input.get('temperature', 35.5)
    eda  = sensor_input.get('blood_flow', 1.0)
    hrv  = sensor_input.get('hrv', 0.05)
    ibi  = sensor_input.get('ir_signal', 60.0 / max(hr, 1))

    acc = sensor_input.get('accel_series', {})
    ax, ay, az = acc.get('x', 0.0), acc.get('y', 0.0), acc.get('z', 0.0)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    gyro = sensor_input.get('gyro_series', {})
    gx = gyro.get('std_x', 0.0)
    gy = gyro.get('std_y', 0.0)
    gz = gyro.get('std_z', 0.0)

    steps = sensor_input.get('steps', 0)
    intensity = min(3, steps // 30)
    micro_tremor = sensor_input.get('micro_tremor', 0.0)

    row = {f: 0.0 for f in feature_cols}

    # Primary physiological signals
    row['HR']        = hr
    row['TEMP']      = temp
    row['EDA']       = eda
    row['HRV']       = hrv
    row['IBI']       = ibi
    row['accel_mag'] = accel_mag

    # Accel features
    row['tBodyAcc-std()-X'] = sensor_input.get('accel_series', {}).get('std_x', 0.0)
    row['tBodyAcc-std()-Y'] = sensor_input.get('accel_series', {}).get('std_y', 0.0)
    row['tBodyAcc-std()-Z'] = sensor_input.get('accel_series', {}).get('std_z', 0.0)
    row['tBodyAccMag-mean()'] = accel_mag
    row['tBodyAccMag-std()']  = accel_mag * 0.1

    # Gyro features
    row['tBodyGyro-std()-X'] = gx
    row['tBodyGyro-std()-Y'] = gy
    row['tBodyGyro-std()-Z'] = gz

    # Micro-tremor → Jerk features
    row['tBodyAccJerkMag-mean()'] = micro_tremor
    row['tBodyAccJerkMag-std()']  = micro_tremor * 0.8

    # Activity context
    row['TotalIntensity']     = intensity * 20
    row['AverageIntensity']   = intensity / 3.0
    row['VeryActiveMinutes']  = steps * 0.5
    row['SedentaryMinutes']   = max(0, 60 - steps * 0.5)

    df_input = pd.DataFrame([row])
    stress_level = int(model.predict(df_input)[0])
    probs = model.predict_proba(df_input)[0]

    return {
        'stress_level': stress_level,
        'label': STRESS_LABELS[stress_level],
        'probabilities': {
            'low':    round(float(probs[0]), 3),
            'medium': round(float(probs[1]), 3),
            'high':   round(float(probs[2]), 3),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df1, df2, df3 = load_data("stress_data1.xlsx", "stress_data2.xlsx", "stress_data3.xlsx")

    print("\nBuilding dataset...")
    combined = build_dataset(df1, df2, df3)

    print("\nTraining model...")
    model, feature_cols = train_model(combined)
    save_model(model, feature_cols)

    # ── Demo ──
    print("\n" + "="*45)
    print("  DEMO PREDICTIONS")
    print("="*45)

    scenarios = [
        {
            "label": "Calm & resting",
            "input": {"heart_rate": 60, "temperature": 35.1, "blood_flow": 0.65,
                      "hrv": 0.12, "ir_signal": 1.0, "steps": 0, "micro_tremor": 0.0,
                      "accel_series": {"x": 0.01, "y": 0.0, "z": 0.98}}
        },
        {
            "label": "Mildly stressed (work deadline)",
            "input": {"heart_rate": 84, "temperature": 35.6, "blood_flow": 1.3,
                      "hrv": 0.04, "ir_signal": 0.71, "steps": 3, "micro_tremor": 0.1,
                      "accel_series": {"x": 0.05, "y": 0.02, "z": 0.95}}
        },
        {
            "label": "High stress (panic / anxiety)",
            "input": {"heart_rate": 115, "temperature": 36.5, "blood_flow": 1.85,
                      "hrv": 0.01, "ir_signal": 0.52, "steps": 2, "micro_tremor": 0.45,
                      "accel_series": {"x": 0.12, "y": 0.08, "z": 0.88}}
        }
    ]

    for s in scenarios:
        result = predict_stress(model, feature_cols, s['input'])
        print(f"\n  [{s['label']}]")
        print(f"  → {result['label']}")
        print(f"     Low={result['probabilities']['low']:.2f}  "
              f"Med={result['probabilities']['medium']:.2f}  "
              f"High={result['probabilities']['high']:.2f}")