"""
Tests for ml/inference.py
Verifies model loading, rolling buffer, single + batch inference,
invalid input handling, and train/inference feature schema parity.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference import GVMSInference, RollingBuffer, predict_window
from ml.features import WINDOW_SIZE, FEATURE_NAMES, FEATURE_NAMES_WITH_VS

MODEL_DIR = ROOT / "ml" / "models"


# ── RollingBuffer ─────────────────────────────────────────────────────────────

def test_rolling_buffer_starts_empty():
    buf = RollingBuffer(capacity=10)
    assert not buf.is_full


def test_rolling_buffer_fills_correctly():
    buf = RollingBuffer(capacity=5)
    for i in range(5):
        buf.push(float(i), float(i), float(i))
    assert buf.is_full


def test_rolling_buffer_overwrites_oldest():
    buf = RollingBuffer(capacity=3)
    buf.push(1.0, 1.0, 1.0)
    buf.push(2.0, 2.0, 2.0)
    buf.push(3.0, 3.0, 3.0)
    buf.push(4.0, 4.0, 4.0)   # should overwrite 1.0
    ax, _, _, _ = buf.get_arrays()
    assert ax[0] == pytest.approx(2.0)


def test_rolling_buffer_clear():
    buf = RollingBuffer(capacity=5)
    for i in range(5):
        buf.push(1.0, 1.0, 1.0)
    buf.clear()
    assert not buf.is_full


# ── GVMSInference loading ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """Load inference engine once for all tests."""
    if not (MODEL_DIR / "model.pkl").exists():
        pytest.skip("Model not trained yet — run ml/train.py first")
    return GVMSInference(smoothing_n=3)


def test_engine_is_ready(engine):
    assert engine.is_ready


def test_engine_has_correct_feature_schema(engine):
    """Feature names from loaded model must match training schema."""
    loaded_names = engine._feature_names
    expected = FEATURE_NAMES if not engine._include_vs else FEATURE_NAMES_WITH_VS
    assert loaded_names == expected, \
        f"Feature schema mismatch.\nExpected: {expected}\nGot:      {loaded_names}"


def test_engine_int_to_label_has_five_entries(engine):
    assert len(engine._int_to_label) == 5


# ── Single-sample inference ───────────────────────────────────────────────────

def test_infer_not_ready_before_buffer_full(engine):
    engine.reset()
    result = engine.infer(-2000.0, 450.0, 16700.0, 0.0)
    assert result["ready"] is False
    assert result["label"] is None


def test_infer_ready_after_window_size_samples(engine):
    engine.reset()
    for _ in range(WINDOW_SIZE):
        result = engine.infer(-2000.0, 450.0, 16700.0, 0.0)
    assert result["ready"] is True
    assert result["label"] is not None
    assert isinstance(result["label"], str)


def test_infer_label_is_valid_class(engine):
    engine.reset()
    for _ in range(WINDOW_SIZE):
        engine.infer(-2000.0, 450.0, 16700.0, 0.0)
    result = engine.infer(-2000.0, 450.0, 16700.0, 0.0)
    valid = {"Gentle_Tap", "Hand_Tap", "Multi_Tap", "MultiHand_Tap", "Walk1"}
    assert result["label"] in valid, f"Unexpected label: {result['label']}"


def test_infer_confidence_between_0_and_1(engine):
    engine.reset()
    for _ in range(WINDOW_SIZE + 1):
        result = engine.infer(-2000.0, 450.0, 16700.0, 0.0)
    if result["confidence"] is not None:
        assert 0.0 <= result["confidence"] <= 1.0


def test_infer_smoothed_label_is_valid(engine):
    engine.reset()
    for _ in range(WINDOW_SIZE + 5):
        result = engine.infer(-2000.0, 450.0, 16700.0, 0.0)
    valid = {"Gentle_Tap", "Hand_Tap", "Multi_Tap", "MultiHand_Tap", "Walk1"}
    assert result["smoothed_label"] in valid


def test_infer_inference_ms_is_positive(engine):
    engine.reset()
    for _ in range(WINDOW_SIZE + 1):
        result = engine.infer(-2000.0, 450.0, 16700.0, 0.0)
    assert result["inference_ms"] >= 0.0


# ── predict_window (stateless) ────────────────────────────────────────────────

def test_predict_window_returns_valid_label():
    if not (MODEL_DIR / "model.pkl").exists():
        pytest.skip("Model not trained yet")
    rng = np.random.default_rng(42)
    ax = rng.normal(-2000, 100, WINDOW_SIZE)
    ay = rng.normal(450, 100, WINDOW_SIZE)
    az = rng.normal(16700, 200, WINDOW_SIZE)
    result = predict_window(ax, ay, az)
    valid = {"Gentle_Tap", "Hand_Tap", "Multi_Tap", "MultiHand_Tap", "Walk1"}
    assert result["label"] in valid


def test_predict_window_confidence_range():
    if not (MODEL_DIR / "model.pkl").exists():
        pytest.skip("Model not trained yet")
    rng = np.random.default_rng(7)
    ax = rng.normal(-2000, 100, WINDOW_SIZE)
    ay = rng.normal(450,   100, WINDOW_SIZE)
    az = rng.normal(16700, 200, WINDOW_SIZE)
    result = predict_window(ax, ay, az)
    if result["confidence"] is not None:
        assert 0.0 <= result["confidence"] <= 1.0


# ── Invalid input handling ────────────────────────────────────────────────────

def test_infer_handles_nan_gracefully(engine):
    """NaN inputs should not crash — model should still return a dict."""
    engine.reset()
    try:
        result = engine.infer(float("nan"), 450.0, 16700.0, 0.0)
        # Should either return ready=False or a result without crash
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"NaN input raised unexpected exception: {e}")


def test_infer_handles_extreme_values(engine):
    engine.reset()
    for _ in range(WINDOW_SIZE + 1):
        result = engine.infer(99999.0, -99999.0, 99999.0, 99999.0)
    assert isinstance(result, dict)
    # Should produce some label even for out-of-range values
    if result["ready"]:
        assert result["label"] is not None
