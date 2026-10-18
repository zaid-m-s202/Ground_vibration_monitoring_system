"""
ml/features.py
==============
Ground Vibration Monitoring System (GVMS)
Sliding-window extraction and feature computation.

Window design
-------------
* Window size : 50 samples  (~1.5 s at 33 Hz nominal rate)
* Step size   : 25 samples  (50 % overlap → detection latency ~0.75 s)
* Both are configurable via WINDOW_SIZE / STEP_SIZE constants.

Feature set (AX/AY/AZ — 3 axes × 12 statistics = 36 features)
---------------------------------------------------------------
Per axis: mean, std, rms, min, max, peak_to_peak, mad (max absolute deviation),
          energy, skewness, kurtosis, zero_crossing_rate, range_norm.

Magnitude features (4 features)
--------------------------------
magnitude_mean, magnitude_std, magnitude_max, dynamic_mag_rms
(magnitude = sqrt(AX²+AY²+AZ²);  dynamic_mag removes gravity component)

VibrationScore features (3 features — optional second model)
------------------------------------------------------------
vs_mean, vs_std, vs_max

Total base features  : 36 + 4 = 40
Total with VS        : 40 + 3 = 43

IMPORTANT: The exact same feature names and order must be used at training
and inference time. FEATURE_NAMES and FEATURE_NAMES_WITH_VS are the
authoritative lists exported by this module.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Window configuration ───────────────────────────────────────────────────────
WINDOW_SIZE = 50   # samples per window
STEP_SIZE   = 25   # step between windows (50 % overlap)

# ── Axes to extract features from ─────────────────────────────────────────────
AXES = ["AX", "AY", "AZ"]

# ── Authoritative feature name lists ──────────────────────────────────────────
_STAT_SUFFIXES = [
    "mean", "std", "rms", "min", "max",
    "peak_to_peak", "mad", "energy",
    "skewness", "kurtosis", "zcr", "range_norm",
]
AXIS_FEATURE_NAMES = [f"{ax}_{s}" for ax in AXES for s in _STAT_SUFFIXES]
MAG_FEATURE_NAMES  = [
    "mag_mean", "mag_std", "mag_max", "dynamic_mag_rms"
]
VS_FEATURE_NAMES   = ["vs_mean", "vs_std", "vs_max"]

FEATURE_NAMES          = AXIS_FEATURE_NAMES + MAG_FEATURE_NAMES          # 40
FEATURE_NAMES_WITH_VS  = FEATURE_NAMES + VS_FEATURE_NAMES                # 43


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int   = STEP_SIZE,
    include_vs: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Slide a window over each recording and extract features.

    Parameters
    ----------
    df          : cleaned DataFrame from preprocessing.validate_and_clean()
    window_size : number of samples per window
    step_size   : step between consecutive windows
    include_vs  : whether to include VibrationScore features

    Returns
    -------
    X           : (n_windows, n_features) float array
    y           : (n_windows,) int array  — label_int
    feature_names : list of feature name strings (length = n_features)
    """
    X_rows, y_rows = [], []

    for fname, grp in df.groupby("source_file"):
        grp = grp.sort_values("SessionElapsed_ms").reset_index(drop=True)
        n = len(grp)

        if n < window_size:
            logger.warning(
                "%s: only %d rows, need %d for one window — skipping",
                fname, n, window_size
            )
            continue

        ax  = grp["AX"].values.astype(np.float64)
        ay  = grp["AY"].values.astype(np.float64)
        az  = grp["AZ"].values.astype(np.float64)
        vs  = grp["VibrationScore"].values.astype(np.float64) \
              if "VibrationScore" in grp.columns else np.zeros(n)
        lbl = int(grp["label_int"].iloc[0])

        wins = 0
        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            feat = _window_features(
                ax[start:end], ay[start:end], az[start:end],
                vs[start:end] if include_vs else None,
            )
            X_rows.append(feat)
            y_rows.append(lbl)
            wins += 1

        logger.debug("%s → %d windows extracted", fname, wins)

    if not X_rows:
        raise ValueError("No windows extracted — check data and window_size.")

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int32)
    feature_names = FEATURE_NAMES_WITH_VS if include_vs else FEATURE_NAMES

    logger.info(
        "Extracted %d windows, %d features, %d classes",
        len(X), X.shape[1], len(np.unique(y))
    )
    return X, y, feature_names


def features_from_buffer(
    ax_buf: np.ndarray,
    ay_buf: np.ndarray,
    az_buf: np.ndarray,
    vs_buf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute features from a single pre-filled window buffer.
    Used by the real-time inference service.

    Parameters
    ----------
    ax_buf, ay_buf, az_buf : 1-D arrays of length WINDOW_SIZE
    vs_buf                 : optional 1-D VibrationScore array

    Returns
    -------
    1-D feature vector (length 40 or 43)
    """
    if len(ax_buf) < WINDOW_SIZE:
        raise ValueError(
            f"Buffer too short: {len(ax_buf)} < WINDOW_SIZE={WINDOW_SIZE}"
        )
    ax = ax_buf[-WINDOW_SIZE:].astype(np.float64)
    ay = ay_buf[-WINDOW_SIZE:].astype(np.float64)
    az = az_buf[-WINDOW_SIZE:].astype(np.float64)
    vs = vs_buf[-WINDOW_SIZE:].astype(np.float64) if vs_buf is not None else None
    return _window_features(ax, ay, az, vs)


# ──────────────────────────────────────────────────────────────────────────────
# Internal computation
# ──────────────────────────────────────────────────────────────────────────────

def _axis_stats(x: np.ndarray) -> list[float]:
    """12 statistics for a single axis window."""
    n    = len(x)
    mean = x.mean()
    std  = x.std() + 1e-9
    rms  = float(np.sqrt(np.mean(x ** 2)))
    xmin = float(x.min())
    xmax = float(x.max())
    p2p  = xmax - xmin
    mad  = float(np.mean(np.abs(x - mean)))
    energy = float(np.sum(x ** 2) / n)

    # Skewness
    skew = float(np.mean(((x - mean) / std) ** 3))
    # Kurtosis (excess)
    kurt = float(np.mean(((x - mean) / std) ** 4) - 3.0)

    # Zero-crossing rate (after mean-removal)
    xc   = x - mean
    zcr  = float(np.sum(np.abs(np.diff(np.sign(xc)))) / (2 * (n - 1)))

    # Range normalised = (max-min) / (|mean| + 1)
    rnorm = p2p / (abs(mean) + 1.0)

    return [mean, std, rms, xmin, xmax, p2p, mad, energy, skew, kurt, zcr, rnorm]


def _window_features(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    vs: Optional[np.ndarray],
) -> np.ndarray:
    """Build the full feature vector for one window."""
    feats = []

    # Axis features
    for x in (ax, ay, az):
        feats.extend(_axis_stats(x))

    # Magnitude features
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    # Dynamic magnitude (remove gravity estimate = mean AZ)
    az_mean = az.mean()
    ax_mean = ax.mean()
    ay_mean = ay.mean()
    dyn_ax  = ax - ax_mean
    dyn_ay  = ay - ay_mean
    dyn_az  = az - az_mean
    dyn_mag = np.sqrt(dyn_ax**2 + dyn_ay**2 + dyn_az**2)

    feats.extend([
        float(mag.mean()),
        float(mag.std()),
        float(mag.max()),
        float(np.sqrt(np.mean(dyn_mag**2))),   # dynamic_mag_rms
    ])

    # Optional VibrationScore features
    if vs is not None:
        feats.extend([float(vs.mean()), float(vs.std()), float(vs.max())])

    return np.array(feats, dtype=np.float64)
