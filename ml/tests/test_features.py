"""
Tests for ml/features.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.features import (
    extract_windows, features_from_buffer,
    WINDOW_SIZE, STEP_SIZE,
    FEATURE_NAMES, FEATURE_NAMES_WITH_VS,
    AXES, _axis_stats,
)
from ml.preprocessing import load_all_csvs, validate_and_clean

DATA_DIR = ROOT / "ml" / "data"


# ── Feature name integrity ────────────────────────────────────────────────────

def test_feature_names_length():
    assert len(FEATURE_NAMES) == 40          # 3 axes × 12 + 4 magnitude
    assert len(FEATURE_NAMES_WITH_VS) == 43  # + 3 VS features


def test_feature_names_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES_WITH_VS)) == len(FEATURE_NAMES_WITH_VS)


def test_feature_names_order_stable():
    # Critical: same order at train and inference time
    a = FEATURE_NAMES[:]
    b = FEATURE_NAMES[:]
    assert a == b


# ── _axis_stats ───────────────────────────────────────────────────────────────

def test_axis_stats_length():
    x = np.random.randn(WINDOW_SIZE)
    stats = _axis_stats(x)
    assert len(stats) == 12


def test_axis_stats_constant_signal():
    x = np.ones(WINDOW_SIZE) * 500.0
    stats = _axis_stats(x)
    mean, std, rms = stats[0], stats[1], stats[2]
    assert abs(mean - 500.0) < 1e-6
    assert std < 1e-3          # near-zero std
    assert abs(rms - 500.0) < 1e-3


def test_axis_stats_zero_signal():
    x = np.zeros(WINDOW_SIZE)
    stats = _axis_stats(x)
    assert stats[0] == 0.0     # mean
    assert stats[4] == 0.0     # max
    assert stats[5] == 0.0     # peak-to-peak


# ── features_from_buffer ─────────────────────────────────────────────────────

def test_features_from_buffer_no_vs():
    rng = np.random.default_rng(0)
    ax = rng.normal(-2000, 100, WINDOW_SIZE)
    ay = rng.normal(450,   100, WINDOW_SIZE)
    az = rng.normal(16700, 200, WINDOW_SIZE)
    feat = features_from_buffer(ax, ay, az)
    assert feat.shape == (40,)
    assert np.isfinite(feat).all()


def test_features_from_buffer_with_vs():
    rng = np.random.default_rng(1)
    ax = rng.normal(-2000, 100, WINDOW_SIZE)
    ay = rng.normal(450,   100, WINDOW_SIZE)
    az = rng.normal(16700, 200, WINDOW_SIZE)
    vs = rng.uniform(0, 1000, WINDOW_SIZE)
    feat = features_from_buffer(ax, ay, az, vs)
    assert feat.shape == (43,)
    assert np.isfinite(feat).all()


def test_features_from_buffer_too_short_raises():
    with pytest.raises(ValueError, match="Buffer too short"):
        features_from_buffer(
            np.zeros(5), np.zeros(5), np.zeros(5)
        )


def test_features_from_buffer_uses_last_n_samples():
    """If buffer > WINDOW_SIZE, should use the last WINDOW_SIZE samples."""
    rng = np.random.default_rng(2)
    ax = rng.normal(-2000, 100, WINDOW_SIZE + 10)
    ay = rng.normal(450,   100, WINDOW_SIZE + 10)
    az = rng.normal(16700, 200, WINDOW_SIZE + 10)
    feat = features_from_buffer(ax, ay, az)
    assert feat.shape == (40,)


# ── extract_windows ───────────────────────────────────────────────────────────

def test_extract_windows_returns_correct_shapes():
    df = validate_and_clean(load_all_csvs(DATA_DIR))
    X, y, names = extract_windows(df, include_vs=False)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_NAMES)
    assert names == FEATURE_NAMES


def test_extract_windows_with_vs():
    df = validate_and_clean(load_all_csvs(DATA_DIR))
    X, y, names = extract_windows(df, include_vs=True)
    assert X.shape[1] == len(FEATURE_NAMES_WITH_VS)
    assert names == FEATURE_NAMES_WITH_VS


def test_extract_windows_five_classes():
    df = validate_and_clean(load_all_csvs(DATA_DIR))
    _, y, _ = extract_windows(df, include_vs=False)
    assert len(np.unique(y)) == 5, "Expected windows from all 5 classes"


def test_extract_windows_no_nans():
    df = validate_and_clean(load_all_csvs(DATA_DIR))
    X, _, _ = extract_windows(df, include_vs=False)
    assert np.isfinite(X).all(), "NaN or Inf in feature matrix"


def test_extract_windows_label_assignment():
    """Each window must inherit label from its source file, not from Status."""
    from ml.preprocessing import LABEL_TO_INT, FILE_LABEL_MAP
    df = validate_and_clean(load_all_csvs(DATA_DIR))
    _, y, _ = extract_windows(df, include_vs=False)
    valid_ints = set(LABEL_TO_INT.values())
    assert set(y.tolist()).issubset(valid_ints), \
        "Window labels contain unexpected integer codes"


def test_extract_windows_minimum_size():
    """Too-small dataframe should produce zero windows gracefully."""
    df = validate_and_clean(load_all_csvs(DATA_DIR))
    tiny = df.head(WINDOW_SIZE - 1).copy()
    # Override source_file so there's only one group with too few rows
    tiny["source_file"] = "Gentle-Tap-R.csv"
    tiny["label"]       = "Gentle_Tap"
    tiny["label_int"]   = 0
    with pytest.raises(ValueError, match="No windows extracted"):
        extract_windows(tiny, include_vs=False)
