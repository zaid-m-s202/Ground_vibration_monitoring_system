# Ground Vibration Monitoring System (GVMS)
### ESP32 + MPU-6050 · Random Forest · 10-second Batch ML Classification

> **Scope notice:** This system classifies controlled vibration patterns
> (taps, walking, multi-hand strikes) captured with an MPU-6050
> accelerometer on an ESP32. It does **not** detect or predict seismic events.
> All class labels come from filenames, not from firmware thresholds.

---

## Project Structure

```
ml_IDMP/
├── ml/
│   ├── data/                        # CSV recordings (5 classes)
│   │   ├── Gentle-Tap-R.csv         # 1989 real rows
│   │   ├── Hand-Tap-R.csv           # 2000 rows
│   │   ├── Multi-Tap-R.csv          # 2000 rows
│   │   ├── MultiHand-Tap-R.csv      # 2000 rows
│   │   └── Walk1.csv                # 2000 rows
│   ├── models/                      # Saved artefacts (after training)
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── metadata.json
│   ├── outputs/                     # Evaluation results
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── classification_report.txt
│   │   ├── evaluation_report.json
│   │   ├── summary.json
│   │   ├── vibration_score_analysis.json
│   │   └── train.log
│   ├── tests/
│   │   ├── test_preprocessing.py
│   │   ├── test_features.py
│   │   └── test_inference.py
│   ├── preprocessing.py             # Data loading, cleaning, label assignment
│   ├── features.py                  # Windowing + 40-feature extraction
│   ├── train.py                     # RF + SVM training, model selection
│   ├── evaluate.py                  # Metrics, plots, summary
│   ├── inference.py                 # Real-time rolling-buffer inference
│   ├── requirements.txt
│   └── __init__.py
├── dashboard/
│   └── app.py                       # Flask dashboard + REST API
├── firmware/
│   └── gvms_final.ino               # ESP32 Arduino firmware
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r ml/requirements.txt
```

### 2. Train the model
```bash
python -m ml.train
```
Outputs saved to `ml/models/` and `ml/outputs/`:
- `model.pkl` — trained Random Forest
- `scaler.pkl` — fitted StandardScaler
- `metadata.json` — feature schema + label mapping
- `evaluation_report.json` — full per-class metrics

### 3. Run evaluation (plots + report)
```bash
python -m ml.evaluate
```
Generates: `confusion_matrix.png`, `feature_importance.png`, `summary.json`

### 4. Test inference (standalone)
```bash
python ml/inference.py
```

### 5. Start the dashboard
```bash
python dashboard/app.py
# Open http://localhost:5000
```

### 6. Run all tests
```bash
python -m pytest ml/tests/ -v
```

---

## System Architecture

```
ESP32 (gvms_final.ino)
  └── MPU-6050 @ 100 Hz
        └── downsample ÷3 → ~33 Hz
              └── ring buffer (330 samples = 10 s)
                    └── HTTP POST /api/infer/batch (compact [[ax,ay,az,vs],...])
                          └── Flask (dashboard/app.py)
                                └── GVMSInference (ml/inference.py)
                                      └── Random Forest → majority-vote label
                                            └── JSON response → ESP32 OLED + dashboard
```

---

## ML Pipeline Design

### Label Assignment
Labels come **exclusively from filenames**:

| File                  | Class Label     | Rows  |
|-----------------------|-----------------|-------|
| Gentle-Tap-R.csv      | Gentle_Tap      | 1 989 |
| Hand-Tap-R.csv        | Hand_Tap        | 2 000 |
| Multi-Tap-R.csv       | Multi_Tap       | 2 000 |
| MultiHand-Tap-R.csv   | MultiHand_Tap   | 2 000 |
| Walk1.csv             | Walk1           | 2 000 |
| **Total**             |                 | **9 989** |

The `Status` column (NORMAL/ALERT) is a firmware threshold — it is **not**
used as a model input or class target.

### Sensor Data
- **AX, AY, AZ**: Raw MPU-6050 accelerometer readings in LSB units
  (16 384 LSB/g at ±2 g full scale). AZ ≈ 16 384 at rest (gravity).
- **VibrationScore**: Sum of absolute sample-to-sample differences computed
  on the ESP32 before classification. Causal and available at inference time.
  Used in a secondary model variant only; AX/AY/AZ-only baseline is primary.

### Sampling & Downsampling
- Sensor reads at **100 Hz** (`SENSOR_INTERVAL = 10 ms`)
- Firmware downsamples by **÷3** (`ML_DOWNSAMPLE = 3`) → ~33 Hz effective
- 10-second batch = **330 samples** sent per HTTP request

### Windowing
- **Window size**: 50 samples (~1.5 s at 33 Hz) — `ML_WINDOW_SIZE = 50`
- **Step size**: 25 samples (50% overlap, ~0.75 s detection latency)
- A 330-sample batch produces ~11 overlapping windows; the majority-voted
  label across all ready windows is returned as the batch prediction.

### Feature Set (40 features per window)
**Per-axis (AX, AY, AZ) × 12 statistics = 36:**
mean, std, RMS, min, max, peak-to-peak, MAD, energy, skewness, kurtosis,
zero-crossing rate, normalised range

**Magnitude (4):**
magnitude mean / std / max, dynamic-magnitude RMS

Optional: +3 VibrationScore features (mean/std/max) for secondary model (43 total).

### Data-Leakage Prevention
Random-splitting overlapping windows from a single recording inflates test
scores — windows from the same physical event end up in both train and test.

**Strategy:** Temporal split — first 70% of windows per class → train,
last 30% → test. Scaler fitted on training data only.

**Important limitation:** This is a preliminary evaluation. True
session-independent validation requires multiple independent recordings per
class. Current scores reflect within-session generalisation only.

---

## Model Results

| Model                      | Accuracy | Macro-F1 | Train time | Infer latency |
|----------------------------|----------|----------|------------|---------------|
| RandomForest (AX/AY/AZ)    | 1.0000   | 1.0000   | 0.33 s     | ~0.34 ms      |
| SVM (AX/AY/AZ)             | 0.9913   | 0.9913   | 0.01 s     | ~0.01 ms      |
| RandomForest (+VS)         | 1.0000   | 1.0000   | 0.31 s     | ~0.34 ms      |

- Train / test split: 279 / 115 windows per model
- **Selected model:** `RandomForest_AXYZonly` — best macro-F1, no VibrationScore
  leakage risk

**Top 5 features by importance:**

| Rank | Feature   | Importance |
|------|-----------|------------|
| 1    | AY_mad    | 0.1176     |
| 2    | AX_mad    | 0.0778     |
| 3    | AX_std    | 0.0658     |
| 4    | AY_std    | 0.0644     |
| 5    | AY_rms    | 0.0538     |

---

## API Reference (Flask Dashboard)

### `POST /api/infer`
Single-sample inference (accumulates into a rolling buffer):
```json
{"ax": -2020, "ay": 450, "az": 16700, "vibration_score": 0}
```
Response:
```json
{
  "ready": true,
  "label": "Gentle_Tap",
  "label_int": 0,
  "confidence": 0.97,
  "smoothed_label": "Gentle_Tap",
  "inference_ms": 0.45
}
```
`"ready": false` until the rolling buffer has enough samples for one window.

### `POST /api/infer/batch`
Batch inference — accepts either format:
```json
{"samples": [{"ax": -2020, "ay": 450, "az": 16700, "vibration_score": 0}]}
```
or compact (ESP32 format):
```json
{"samples": [[-2020, 450, 16700, 0], ...]}
```
Response:
```json
{"results": [...], "n_processed": 330}
```

### `GET /api/status`
Returns model readiness, name, feature count, window size, timestamp.

### `GET /api/model/info`
Returns full model metadata from `ml/models/metadata.json`.

### `GET /api/history`
Returns last 200 inference results.

### `POST /api/reset`
Clears the rolling buffer.

---

## ESP32 Firmware (`firmware/gvms_final.ino`)

### Required Arduino Libraries
Install via Arduino Library Manager:
- `MPU6050` (Electronic Cats or i2cdevlib)
- `Adafruit SSD1306` + `Adafruit GFX`
- `ArduinoJson` v6
- `SD` (bundled with ESP32 core)

### Before Flashing
1. Set `ML_SERVER_IP` to your PC's LAN IP (`ipconfig` on Windows to find it)
2. Update `WIFI_SSID` / `WIFI_PASSWORD` to match your network
3. Start Flask: `python dashboard/app.py`

### Pin Assignments

| Peripheral | Pin(s)                         |
|------------|--------------------------------|
| MPU-6050   | SDA=GPIO21, SCL=GPIO22         |
| OLED SSD1306 | SDA=GPIO21, SCL=GPIO22 (shared I²C) |
| SD card    | SCK=18, MISO=19, MOSI=23, CS=5 |
| Buzzer     | GPIO27                         |
| Green LED  | GPIO25                         |
| Red LED    | GPIO26                         |

### SD Log Format
```
/earthquake.csv
Time_ms,AX_raw,AY_raw,AZ_raw,Vibration,Status
```

### 10-second Batch Flow
```
readSensor() called every 10 ms (100 Hz)
  → every 3rd sample pushed to batchBuffer (~33 Hz)
  → when batchBuffer reaches 330 samples:
      triggerMLSend() copies buffer to mlSnapshot
      mlHttpTask (Core 0) sends POST /api/infer/batch
      majority-vote label displayed on OLED + dashboard
      new batch starts automatically
```

### OLED Display
- Lines 1–2: Live AX / AY / AZ / VibrationScore
- Line 3: STATUS: NORMAL or STATUS: ALERT
- Line 4: Recording progress (%) or current ML label
- Line 5: Confidence % (shown after each batch completes)
- Line 6: "GROUND VIBRATION" branding

### Firmware Dashboard (`/`)
The ESP32 serves its own rich web dashboard with live waveform, vibration
gauge, ML classification panel, event history, and threshold settings —
accessible directly at the ESP32's IP address on port 80.

---

## Test Results

```
45 passed in 5.71s
```

| Test Module          | Tests | Result |
|----------------------|-------|--------|
| test_preprocessing   | 13    | PASS   |
| test_features        | 16    | PASS   |
| test_inference       | 16    | PASS   |
| **Total**            | **45**| **PASS** |

---

## Known Limitations

1. **Single session per class.** The 1.0 accuracy reflects the model
   distinguishing statistically different distributions within one recording
   session per class, not truly independent unseen sessions. Real-world
   generalisation should be validated with multiple recordings.

2. **Gentle-Tap-R.csv** contains real MPU-6050 data; the other four files
   use statistically realistic simulated data with the same column structure.
   Full real-data recordings are needed for production deployment.

3. **Inference latency** (~0.34 ms per window) is measured on a desktop
   Python interpreter. Server/hardware results will vary.

4. **No calibrated probabilities.** `confidence` is the raw RF posterior
   probability (not Platt-scaled).

5. **Hardware testing** of the ESP32 firmware cannot be verified without
   physical hardware.

6. **IP address is hardcoded** in the firmware (`ML_SERVER_IP`). Update it
   whenever your PC's LAN IP changes.

### Additional Data Required for Production
- At least 5 independent recording sessions per class
- Recordings across different surface types and mounting conditions
- Multiple operators for hand-tap classes

---

## Environment

- **Python** 3.14.7
- **Platform** Windows 10/11 (PowerShell)

| Package       | Min version |
|---------------|-------------|
| numpy         | ≥ 1.24.0    |
| pandas        | ≥ 2.0.0     |
| scikit-learn  | ≥ 1.3.0     |
| matplotlib    | ≥ 3.7.0     |
| seaborn       | ≥ 0.12.0    |
| joblib        | ≥ 1.3.0     |
| flask         | ≥ 3.0.0     |
| flask-cors    | ≥ 4.0.0     |
| scipy         | ≥ 1.11.0    |

Install everything:
```bash
pip install -r ml/requirements.txt
```
