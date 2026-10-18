"""
Tests for ml/preprocessing.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing import (
    load_csv, load_all_csvs, validate_and_clean,
    analyse_vibration_score,
    FILE_LABEL_MAP, LABEL_TO_INT, INT_TO_LABEL, REQUIRED_COLS,
)

DATA_DIR = ROOT / "ml" / "data"


# ── Label mapping ──────────────────────────────────────────────────────────────

def test_label_map_has_five_classes():
    assert len(FILE_LABEL_MAP) == 5


def test_label_to_int_is_bijective():
    assert len(LABEL_TO_INT) == len(INT_TO_LABEL)
    for k, v in LABEL_TO_INT.items():
        assert INT_TO_LABEL[v] == k


def test_expected_class_names():
    expected = {"Gentle_Tap", "Hand_Tap", "Multi_Tap", "MultiHand_Tap", "Walk1"}
    assert set(LABEL_TO_INT.keys()) == expected


# ── load_csv ───────────────────────────────────────────────────────────────────

def test_load_csv_assigns_correct_label():
    for fname, label in FILE_LABEL_MAP.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            pytest.skip(f"{fname} not found")
        df = load_csv(fpath)
        assert (df["label"] == label).all(), f"Wrong label for {fname}"
        assert (df["label_int"] == LABEL_TO_INT[label]).all()


def test_load_csv_required_columns_present():
    for fname in FILE_LABEL_MAP:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            pytest.skip(f"{fname} not found")
        df = load_csv(fpath)
        for col in REQUIRED_COLS:
            assert col in df.columns, f"Missing {col} in {fname}"


def test_load_csv_unknown_file_raises():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False,
                                     mode="w") as f:
        f.write("SessionElapsed_ms,AX,AY,AZ,VibrationScore,Status\n")
        f.write("100,-2000,400,16700,0,NORMAL\n")
        tmp = f.name
    with pytest.raises(ValueError, match="Unknown file"):
        load_csv(tmp)
    os.unlink(tmp)


# ── load_all_csvs ──────────────────────────────────────────────────────────────

def test_load_all_csvs_returns_five_labels():
    df = load_all_csvs(DATA_DIR)
    labels = set(df["label"].unique())
    assert labels == set(FILE_LABEL_MAP.values()), \
        f"Expected 5 labels, got: {labels}"


def test_load_all_csvs_no_nans_in_required():
    df = load_all_csvs(DATA_DIR)
    for col in REQUIRED_COLS:
        assert df[col].isna().sum() == 0, f"NaN found in {col}"


# ── validate_and_clean ────────────────────────────────────────────────────────

def test_validate_drops_duplicates():
    df = load_all_csvs(DATA_DIR)
    n_raw = len(df)
    df_dup = pd.concat([df, df.head(10)], ignore_index=True)
    df_clean = validate_and_clean(df_dup)
    assert len(df_clean) <= n_raw


def test_validate_sorted_by_timestamp():
    df = load_all_csvs(DATA_DIR)
    df_clean = validate_and_clean(df)
    for _, grp in df_clean.groupby("source_file"):
        ts = grp["SessionElapsed_ms"].values
        assert (ts[1:] >= ts[:-1]).all(), "Timestamps not sorted"


# ── analyse_vibration_score ───────────────────────────────────────────────────

def test_vs_analysis_returns_dict():
    df = validate_and_clean(load_all_csvs(DATA_DIR))
    result = analyse_vibration_score(df)
    assert isinstance(result, dict)
    assert result["available"] is True
    assert "recommendation" in result


def test_vs_analysis_no_vs_column():
    df = pd.DataFrame({"AX": [1], "AY": [2], "AZ": [3]})
    result = analyse_vibration_score(df)
    assert result["available"] is False
