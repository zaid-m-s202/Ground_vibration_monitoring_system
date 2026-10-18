"""
ml/train.py
===========
Ground Vibration Monitoring System (GVMS)
Training pipeline: Random Forest + SVM, honest evaluation, model saving.

Data-leakage prevention
-----------------------
With one continuous recording per class, random splitting of overlapping
windows from the same recording inflates scores — windows from the same
physical event end up in both train and test sets.

Strategy used here:
  1. For each class we have ONE session recording.
  2. We perform a TEMPORAL split: first 70 % of windows -> train,
     last 30 % -> test.  This ensures the test windows come from the
     *end* of each recording (never seen during training) while
     preserving temporal ordering within each session.
  3. Scaler is fit ONLY on the training windows.
  4. This is labelled "preliminary evaluation" because we have only one
     session per class — true session-independent validation requires
     multiple independent recordings per class.

Usage
-----
    python -m ml.train
    # or
    python ml/train.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -- Path setup -----------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
ML_DIR    = ROOT / "ml"
DATA_DIR  = ML_DIR / "data"
MODEL_DIR = ML_DIR / "models"
OUT_DIR   = ML_DIR / "outputs"

for d in (MODEL_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from ml.preprocessing import (
    load_all_csvs, validate_and_clean,
    analyse_vibration_score, LABEL_TO_INT, INT_TO_LABEL, FILE_LABEL_MAP,
)
from ml.features import extract_windows, WINDOW_SIZE, STEP_SIZE, FEATURE_NAMES, FEATURE_NAMES_WITH_VS

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "train.log", mode="w", encoding="utf-8"),
    ],
)
# Fix Windows console Unicode encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
log = logging.getLogger(__name__)

RANDOM_SEED = 42


# ------------------------------------------------------------------------------
def temporal_train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.30):
    """
    Per-class temporal split: first (1-test_ratio) of windows -> train,
    last test_ratio -> test.  No shuffling — preserves temporal order.
    """
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]  # already in temporal order (sorted by ts)
        n_test  = max(1, int(len(idx) * test_ratio))
        n_train = len(idx) - n_test
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    model, model_name: str, scaler,
    feature_names: list,
):
    """Fit scaler+model, evaluate, return metrics dict."""
    # Fit scaler on training data only
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    log.info("Training %s ...", model_name)
    t0 = time.perf_counter()
    model.fit(X_tr_scaled, y_train)
    train_time = time.perf_counter() - t0
    log.info("  Training time: %.2f s", train_time)

    # Inference timing
    t0 = time.perf_counter()
    y_pred = model.predict(X_te_scaled)
    infer_time = (time.perf_counter() - t0) / len(X_test) * 1000   # ms/sample

    # Metrics
    labels_used = sorted(np.unique(np.concatenate([y_train, y_test])))
    target_names = [INT_TO_LABEL[i] for i in labels_used]
    report = classification_report(
        y_test, y_pred, labels=labels_used, target_names=target_names, output_dict=True
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels_used)
    acc      = float(np.mean(y_pred == y_test))
    cm       = confusion_matrix(y_test, y_pred, labels=labels_used).tolist()

    log.info("  Accuracy : %.4f", acc)
    log.info("  Macro-F1 : %.4f", macro_f1)
    log.info("\n%s", classification_report(
        y_test, y_pred, labels=labels_used, target_names=target_names
    ))

    metrics = {
        "model_name":           model_name,
        "accuracy":             round(acc, 4),
        "macro_f1":             round(macro_f1, 4),
        "train_time_s":         round(train_time, 4),
        "infer_time_ms":        round(infer_time, 4),
        "classification_report": report,
        "confusion_matrix":     cm,
        "confusion_labels":     target_names,
        "n_train":              int(len(X_train)),
        "n_test":               int(len(X_test)),
        "n_features":           int(X_train.shape[1]),
        "window_size":          WINDOW_SIZE,
        "step_size":            STEP_SIZE,
        "evaluation_note": (
            "PRELIMINARY — temporal split within single session per class. "
            "Cross-session validation requires additional recordings."
        ),
    }

    # Feature importance (RF only)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        top_idx  = np.argsort(imp)[::-1][:15]
        top_feat = {feature_names[i]: round(float(imp[i]), 5) for i in top_idx}
        metrics["top_15_features"] = top_feat
        log.info("  Top-5 features: %s", list(top_feat.items())[:5])

    return metrics, y_pred


# ------------------------------------------------------------------------------
def main():
    log.info("=" * 65)
    log.info("  GVMS — ML Training Pipeline")
    log.info("=" * 65)

    # -- 1. Load and clean data ------------------------------------------------
    log.info("Loading CSV files from: %s", DATA_DIR)
    df_raw = load_all_csvs(DATA_DIR)
    df     = validate_and_clean(df_raw)

    # -- 2. VibrationScore investigation --------------------------------------
    log.info("\n-- VibrationScore Investigation --")
    vs_info = analyse_vibration_score(df)
    log.info("  Assessment: %s", vs_info.get("assessment", "N/A"))
    log.info("  Recommendation: %s", vs_info.get("recommendation", "N/A"))
    (OUT_DIR / "vibration_score_analysis.json").write_text(
        json.dumps(vs_info, indent=2)
    )

    # -- 3. Class distribution -------------------------------------------------
    dist = df.groupby("label")["label"].count().to_dict()
    log.info("\n-- Class distribution (rows) --")
    for k, v in sorted(dist.items()):
        log.info("  %-20s : %d rows", k, v)

    # -- 4. Extract windows — AX/AY/AZ only (Baseline Model) -----------------
    log.info("\n-- Extracting windows (AX/AY/AZ only, w=%d, step=%d) --",
             WINDOW_SIZE, STEP_SIZE)
    X_base, y, feat_names_base = extract_windows(df, include_vs=False)

    X_tr_b, X_te_b, y_tr, y_te = temporal_train_test_split(X_base, y)
    log.info("Train windows: %d  |  Test windows: %d", len(X_tr_b), len(X_te_b))

    # -- 5. Train Random Forest (Baseline) ------------------------------------
    log.info("\n-- Model 1: Random Forest (AX/AY/AZ only) --")
    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1,
    )
    scaler_rf = StandardScaler()
    metrics_rf, _ = train_and_evaluate(
        X_tr_b, X_te_b, y_tr, y_te,
        rf_base, "RandomForest_AXYZonly", scaler_rf, feat_names_base,
    )

    # -- 6. Train SVM (Baseline) -----------------------------------------------
    log.info("\n-- Model 2: SVM (AX/AY/AZ only) --")
    svm_base = SVC(kernel="rbf", C=10.0, gamma="scale",
                   probability=True, random_state=RANDOM_SEED)
    scaler_svm = StandardScaler()
    metrics_svm, _ = train_and_evaluate(
        X_tr_b, X_te_b, y_tr, y_te,
        svm_base, "SVM_AXYZonly", scaler_svm, feat_names_base,
    )

    # -- 7. RF with VibrationScore (second variant) ---------------------------
    log.info("\n-- Model 3: Random Forest (AX/AY/AZ + VibrationScore) --")
    X_vs, y_vs, feat_names_vs = extract_windows(df, include_vs=True)
    X_tr_vs, X_te_vs, y_tr_vs, y_te_vs = temporal_train_test_split(X_vs, y_vs)

    rf_vs = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1,
    )
    scaler_rf_vs = StandardScaler()
    metrics_rf_vs, _ = train_and_evaluate(
        X_tr_vs, X_te_vs, y_tr_vs, y_te_vs,
        rf_vs, "RandomForest_with_VS", scaler_rf_vs, feat_names_vs,
    )

    # -- 8. Select best model --------------------------------------------------
    all_metrics = [metrics_rf, metrics_svm, metrics_rf_vs]
    best = max(all_metrics, key=lambda m: m["macro_f1"])
    log.info("\n-- Model comparison --")
    for m in all_metrics:
        log.info(
            "  %-35s  acc=%.4f  macro-F1=%.4f  infer=%.3f ms",
            m["model_name"], m["accuracy"], m["macro_f1"], m["infer_time_ms"]
        )
    log.info("  Selected: %s  (highest macro-F1)", best["model_name"])

    # Map model name -> (model, scaler, feature_names)
    model_map = {
        "RandomForest_AXYZonly":   (rf_base,  scaler_rf,    feat_names_base, False),
        "SVM_AXYZonly":            (svm_base, scaler_svm,   feat_names_base, False),
        "RandomForest_with_VS":    (rf_vs,    scaler_rf_vs, feat_names_vs,   True),
    }
    best_model, best_scaler, best_feats, best_uses_vs = model_map[best["model_name"]]

    # -- 9. Save model artefacts -----------------------------------------------
    joblib.dump(best_model,  MODEL_DIR / "model.pkl")
    joblib.dump(best_scaler, MODEL_DIR / "scaler.pkl")

    metadata = {
        "model_name":         best["model_name"],
        "feature_names":      best_feats,
        "include_vs":         best_uses_vs,
        "label_to_int":       LABEL_TO_INT,
        "int_to_label":       {str(k): v for k, v in INT_TO_LABEL.items()},
        "window_size":        WINDOW_SIZE,
        "step_size":          STEP_SIZE,
        "n_classes":          len(LABEL_TO_INT),
        "training_accuracy":  best["accuracy"],
        "macro_f1":           best["macro_f1"],
        "evaluation_note":    best["evaluation_note"],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved model -> %s", MODEL_DIR / "model.pkl")
    log.info("Saved scaler -> %s", MODEL_DIR / "scaler.pkl")
    log.info("Saved metadata -> %s", MODEL_DIR / "metadata.json")

    # -- 10. Save full evaluation report --------------------------------------
    full_report = {
        "models": all_metrics,
        "selected_model": best["model_name"],
        "vibration_score_analysis": vs_info,
        "class_distribution": dist,
        "data_summary": {
            "total_rows": int(len(df)),
            "n_files": int(df["source_file"].nunique()),
            "labels": list(LABEL_TO_INT.keys()),
        },
    }
    (OUT_DIR / "evaluation_report.json").write_text(json.dumps(full_report, indent=2))
    log.info("Saved evaluation report -> %s", OUT_DIR / "evaluation_report.json")

    log.info("\n[OK] Training complete.")
    return full_report


if __name__ == "__main__":
    main()
