"""
ml/inference.py
===============
Ground Vibration Monitoring System (GVMS)
Real-time inference service.

Architecture
------------
* RollingBuffer  : maintains a sliding window of raw sensor samples.
* GVMSInference  : loads the saved model pipeline, runs prediction on
                   each filled window, applies temporal smoothing.
* predict_window : convenience function for single-shot inference.

Usage (standalone test)
-----------------------
    python ml/inference.py

Usage (from Flask / backend)
-----------------------------
    from ml.inference import GVMSInference
    inf = GVMSInference()
    result = inf.infer(ax, ay, az, vibration_score=vs)
"""

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
ML_DIR    = ROOT / "ml"
MODEL_DIR = ML_DIR / "models"
sys.path.insert(0, str(ROOT))

from ml.features import features_from_buffer, WINDOW_SIZE

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
class RollingBuffer:
    """
    Fixed-capacity FIFO buffer for streaming sensor samples.
    Automatically discards oldest sample when full.
    """

    def __init__(self, capacity: int = WINDOW_SIZE):
        self.capacity = capacity
        self._ax: deque = deque(maxlen=capacity)
        self._ay: deque = deque(maxlen=capacity)
        self._az: deque = deque(maxlen=capacity)
        self._vs: deque = deque(maxlen=capacity)

    def push(self, ax: float, ay: float, az: float, vs: float = 0.0):
        self._ax.append(ax)
        self._ay.append(ay)
        self._az.append(az)
        self._vs.append(vs)

    @property
    def is_full(self) -> bool:
        return len(self._ax) == self.capacity

    def get_arrays(self) -> tuple:
        return (
            np.array(self._ax, dtype=np.float64),
            np.array(self._ay, dtype=np.float64),
            np.array(self._az, dtype=np.float64),
            np.array(self._vs, dtype=np.float64),
        )

    def clear(self):
        self._ax.clear(); self._ay.clear()
        self._az.clear(); self._vs.clear()


# ------------------------------------------------------------------------------
class GVMSInference:
    """
    Stateful inference engine for GVMS.

    Parameters
    ----------
    model_dir       : path to ml/models/
    smoothing_n     : number of consecutive windows to vote over
    include_vs      : whether model expects VibrationScore features
                      (loaded automatically from metadata.json)
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        smoothing_n: int = 3,
    ):
        self.model_dir   = Path(model_dir) if model_dir else MODEL_DIR
        self.smoothing_n = smoothing_n
        self._model      = None
        self._scaler     = None
        self._meta       = None
        self._buffer     = RollingBuffer(WINDOW_SIZE)
        self._recent_preds: deque = deque(maxlen=smoothing_n)
        self._loaded     = False
        self._load()

    # -- Loading ---------------------------------------------------------------

    def _load(self):
        meta_path   = self.model_dir / "metadata.json"
        model_path  = self.model_dir / "model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"

        for p in (meta_path, model_path, scaler_path):
            if not p.exists():
                log.warning("Model artefact missing: %s — inference disabled", p)
                return

        self._meta   = json.loads(meta_path.read_text())
        self._model  = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._loaded = True

        self._include_vs    = self._meta.get("include_vs", False)
        self._int_to_label  = {int(k): v for k, v in self._meta["int_to_label"].items()}
        self._feature_names = self._meta["feature_names"]
        self._window_size   = self._meta.get("window_size", WINDOW_SIZE)
        self._buffer        = RollingBuffer(self._window_size)

        log.info(
            "Model loaded: %s  |  features=%d  |  classes=%d",
            self._meta["model_name"],
            len(self._feature_names),
            len(self._int_to_label),
        )

    @property
    def is_ready(self) -> bool:
        return self._loaded

    # -- Inference -------------------------------------------------------------

    def infer(
        self,
        ax: float, ay: float, az: float,
        vibration_score: float = 0.0,
    ) -> dict:
        """
        Push one sample into the rolling buffer and, if the buffer is full,
        run inference and return a result dict.

        Returns
        -------
        {
          "ready":        bool,       # False until buffer has WINDOW_SIZE samples
          "label":        str | None, # predicted class name
          "label_int":    int | None,
          "confidence":   float | None,  # max class probability
          "smoothed_label": str | None,  # after voting over smoothing_n windows
          "inference_ms": float,
        }
        """
        if not self._loaded:
            return {"ready": False, "label": None, "label_int": None,
                    "confidence": None, "smoothed_label": None, "inference_ms": 0.0,
                    "error": "Model not loaded"}

        self._buffer.push(ax, ay, az, vibration_score)

        if not self._buffer.is_full:
            return {"ready": False, "label": None, "label_int": None,
                    "confidence": None, "smoothed_label": None, "inference_ms": 0.0}

        ax_arr, ay_arr, az_arr, vs_arr = self._buffer.get_arrays()
        vs_arg = vs_arr if self._include_vs else None

        t0 = time.perf_counter()
        feat = features_from_buffer(ax_arr, ay_arr, az_arr, vs_arg)
        feat_scaled = self._scaler.transform(feat.reshape(1, -1))

        pred_int  = int(self._model.predict(feat_scaled)[0])
        infer_ms  = (time.perf_counter() - t0) * 1000

        confidence = None
        if hasattr(self._model, "predict_proba"):
            proba      = self._model.predict_proba(feat_scaled)[0]
            confidence = float(np.max(proba))

        label = self._int_to_label.get(pred_int, f"class_{pred_int}")

        # Temporal smoothing via majority vote
        self._recent_preds.append(pred_int)
        counts = np.bincount(list(self._recent_preds), minlength=len(self._int_to_label))
        smoothed_int   = int(np.argmax(counts))
        smoothed_label = self._int_to_label.get(smoothed_int, f"class_{smoothed_int}")

        return {
            "ready":          True,
            "label":          label,
            "label_int":      pred_int,
            "confidence":     round(confidence, 4) if confidence is not None else None,
            "smoothed_label": smoothed_label,
            "inference_ms":   round(infer_ms, 3),
        }

    def reset(self):
        """Clear buffer and prediction history."""
        self._buffer.clear()
        self._recent_preds.clear()


# ------------------------------------------------------------------------------
def predict_window(
    ax_window: np.ndarray,
    ay_window: np.ndarray,
    az_window: np.ndarray,
    vs_window: Optional[np.ndarray] = None,
    model_dir: Optional[Path] = None,
) -> dict:
    """
    Single-shot inference on a pre-filled window (no state).
    Useful for batch testing.
    """
    mdir = Path(model_dir) if model_dir else MODEL_DIR
    meta   = json.loads((mdir / "metadata.json").read_text())
    model  = joblib.load(mdir / "model.pkl")
    scaler = joblib.load(mdir / "scaler.pkl")
    i2l    = {int(k): v for k, v in meta["int_to_label"].items()}
    inc_vs = meta.get("include_vs", False)

    feat = features_from_buffer(ax_window, ay_window, az_window,
                                 vs_window if inc_vs else None)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    pred_int    = int(model.predict(feat_scaled)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(feat_scaled)[0]
        confidence = float(np.max(proba))

    return {
        "label":      i2l.get(pred_int, f"class_{pred_int}"),
        "label_int":  pred_int,
        "confidence": round(confidence, 4) if confidence is not None else None,
    }


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    inf = GVMSInference(smoothing_n=3)
    if not inf.is_ready:
        print("Model not ready — run ml/train.py first.")
        sys.exit(1)

    print("\n-- GVMS Real-time Inference Demo --")
    print("Simulating 80 sensor samples (Walk pattern)...\n")

    rng = np.random.default_rng(0)
    results = []
    for i in range(80):
        t = i * 30
        phase = (t % 1200) / 1200 * 2 * np.pi
        ax  = -2050 + int(100 * np.sin(phase)) + rng.integers(-50, 50)
        ay  =   950 + int(200 * np.cos(phase)) + rng.integers(-50, 50)
        az  = 16700 + int(150 * np.sin(2 * phase)) + rng.integers(-100, 100)
        vs  = max(0, int(rng.normal(300, 100)))
        res = inf.infer(float(ax), float(ay), float(az), float(vs))
        if res["ready"]:
            results.append(res)

    if results:
        last = results[-1]
        print(f"  Last prediction   : {last['label']}")
        print(f"  Confidence        : {last['confidence']}")
        print(f"  Smoothed label    : {last['smoothed_label']}")
        print(f"  Inference latency : {last['inference_ms']} ms")
        print(f"\n  Predictions across {len(results)} windows:")
        from collections import Counter
        cnt = Counter(r["smoothed_label"] for r in results)
        for lbl, n in cnt.most_common():
            print(f"    {lbl:<25} {n} windows")
    print("\n[OK] Inference demo complete.")
