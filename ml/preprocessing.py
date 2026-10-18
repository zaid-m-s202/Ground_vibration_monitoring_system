"""
ml/preprocessing.py
===================
Ground Vibration Monitoring System (GVMS)
Timestamp validation, sampling-rate analysis, and data cleaning.

Key design decisions
--------------------
* AX/AY/AZ are raw MPU-6050 readings in LSB units (±2 g range → 16384 LSB/g).
  AZ ≈ 16384 at rest (1 g upward from gravity). We do NOT remove gravity here;
  features.py handles magnitude/dynamic acceleration instead.
* VibrationScore is computed on-device by the ESP32 from the same AX/AY/AZ
  window being classified, so it is available at inference time.
  However, we flag it as a "derived" field and train an AX/AY/AZ-only baseline
  first (see train.py).
* Status (NORMAL/ALERT) is a threshold-based label from the ESP32 firmware.
  It is NOT used as a model input or as the five-class target.
* Ground-truth labels come exclusively from filenames, not from Status.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── File → class-label mapping (source of truth) ──────────────────────────────
FILE_LABEL_MAP: dict[str, str] = {
    "Gentle-Tap-R.csv":    "Gentle_Tap",
    "Hand-Tap-R.csv":      "Hand_Tap",
    "Multi-Tap-R.csv":     "Multi_Tap",
    "MultiHand-Tap-R.csv": "MultiHand_Tap",
    "Walk1.csv":           "Walk1",
}

LABEL_TO_INT: dict[str, int] = {
    "Gentle_Tap":    0,
    "Hand_Tap":      1,
    "Multi_Tap":     2,
    "MultiHand_Tap": 3,
    "Walk1":         4,
}

INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}

# ── Expected columns ───────────────────────────────────────────────────────────
REQUIRED_COLS = ["SessionElapsed_ms", "AX", "AY", "AZ"]
OPTIONAL_COLS = ["VibrationScore", "Status"]

# ── Sensor constants ───────────────────────────────────────────────────────────
LSB_PER_G      = 16384          # MPU-6050 ±2 g full-scale
GRAVITY_LSB    = 16384          # expected AZ at rest (sensor vertical)
SAMPLING_RATE_NOMINAL_HZ = 33   # ~30 ms between samples at nominal rate


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load a single GVMS CSV file and attach its ground-truth class label."""
    filepath = Path(filepath)
    filename  = filepath.name

    if filename not in FILE_LABEL_MAP:
        raise ValueError(
            f"Unknown file '{filename}'. "
            f"Expected one of: {list(FILE_LABEL_MAP.keys())}"
        )

    df = pd.read_csv(filepath)
    _validate_columns(df, filename)

    # Assign ground-truth label from filename
    df["label"]      = FILE_LABEL_MAP[filename]
    df["label_int"]  = LABEL_TO_INT[FILE_LABEL_MAP[filename]]
    df["source_file"] = filename

    return df


def load_all_csvs(data_dir: str | Path) -> pd.DataFrame:
    """Load all five CSV files from *data_dir* and concatenate them."""
    data_dir = Path(data_dir)
    frames = []
    missing = []

    for fname in FILE_LABEL_MAP:
        fpath = data_dir / fname
        if not fpath.exists():
            missing.append(fname)
            logger.warning("File not found: %s", fpath)
            continue
        df = load_csv(fpath)
        frames.append(df)
        logger.info("Loaded %s — %d rows, label=%s", fname, len(df), FILE_LABEL_MAP[fname])

    if missing:
        logger.warning("Missing files (will train without them): %s", missing)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total rows loaded: %d from %d files", len(combined), len(frames))
    return combined


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Sort by timestamp within each source file.
    2. Drop exact duplicate rows.
    3. Remove rows with NaN in required sensor columns.
    4. Report sampling-interval statistics.
    """
    df = df.copy()

    # Sort by file then timestamp
    df = df.sort_values(["source_file", "SessionElapsed_ms"]).reset_index(drop=True)

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["source_file", "SessionElapsed_ms"]).reset_index(drop=True)
    if len(df) < n_before:
        logger.info("Dropped %d duplicate rows", n_before - len(df))

    # Drop NaN in required cols
    n_before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("Dropped %d rows with NaN sensor values", n_before - len(df))

    # Sampling-interval report
    _report_sampling(df)

    return df


def analyse_vibration_score(df: pd.DataFrame) -> dict:
    """
    Investigate the VibrationScore field to determine whether it is safe to
    use as a model input.

    Returns a dict with findings and a recommendation.
    """
    if "VibrationScore" not in df.columns:
        return {"available": False, "recommendation": "Not present in data."}

    vs = df["VibrationScore"]
    findings = {
        "available":        True,
        "n_nonzero":        int((vs != 0).sum()),
        "pct_nonzero":      float((vs != 0).mean() * 100),
        "min":              float(vs.min()),
        "max":              float(vs.max()),
        "mean":             float(vs.mean()),
        "std":              float(vs.std()),
        "zero_pct":         float((vs == 0).mean() * 100),
    }

    # Check correlation with AX/AY/AZ to detect leakage
    corr_ax = float(df["AX"].corr(vs))
    corr_ay = float(df["AY"].corr(vs))
    corr_az = float(df["AZ"].corr(vs))

    # Compute simple dynamic magnitude for comparison
    ax_mean = df["AX"].mean(); ay_mean = df["AY"].mean(); az_mean = df["AZ"].mean()
    dynamic_mag = np.sqrt(
        (df["AX"] - ax_mean)**2 +
        (df["AY"] - ay_mean)**2 +
        (df["AZ"] - az_mean)**2
    )
    corr_dynmag = float(dynamic_mag.corr(vs))

    findings.update({
        "corr_with_AX":         corr_ax,
        "corr_with_AY":         corr_ay,
        "corr_with_AZ":         corr_az,
        "corr_with_dynamic_mag": corr_dynmag,
        "assessment": (
            "VibrationScore is derived from AX/AY/AZ on the same window. "
            "It is a hand-crafted dynamic-acceleration metric, NOT a label. "
            "It is computed causally on the ESP32 before Status is assigned. "
            "It is safe to use as an additional feature but adds limited "
            "information beyond what raw-axis features already capture. "
            "The AX/AY/AZ-only baseline is trained first."
        ),
        "recommendation": "Include in second model variant only; use AX/AY/AZ-only as primary."
    })

    return findings


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame, filename: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{filename}: missing required columns {missing}. "
            f"Found: {list(df.columns)}"
        )


def _report_sampling(df: pd.DataFrame) -> None:
    """Log per-file sampling interval statistics."""
    for fname, grp in df.groupby("source_file"):
        ts = grp["SessionElapsed_ms"].values
        if len(ts) < 2:
            continue
        intervals = np.diff(ts)
        logger.info(
            "%s — n=%d, interval: mean=%.1f ms, std=%.1f ms, "
            "min=%.0f ms, max=%.0f ms  (~%.1f Hz)",
            fname, len(ts),
            intervals.mean(), intervals.std(),
            intervals.min(), intervals.max(),
            1000.0 / intervals.mean() if intervals.mean() > 0 else 0,
        )
