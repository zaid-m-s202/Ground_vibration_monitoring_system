"""
ml/evaluate.py
==============
Ground Vibration Monitoring System (GVMS)
Post-training evaluation: confusion matrix plot, feature-importance plot,
per-class metrics report, and model size/latency summary.

Usage
-----
    python ml/evaluate.py
    (Run AFTER ml/train.py has saved models to ml/models/)
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
)

ROOT      = Path(__file__).resolve().parent.parent
ML_DIR    = ROOT / "ml"
MODEL_DIR = ML_DIR / "models"
OUT_DIR   = ML_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from ml.preprocessing import (
    load_all_csvs, validate_and_clean, INT_TO_LABEL, LABEL_TO_INT,
)
from ml.features import extract_windows

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def load_artefacts():
    """Load saved model, scaler, metadata."""
    meta  = json.loads((MODEL_DIR / "metadata.json").read_text())
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler= joblib.load(MODEL_DIR / "scaler.pkl")
    return model, scaler, meta


def plot_confusion_matrix(cm: np.ndarray, labels: list, path: Path):
    """Save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title("GVMS — Confusion Matrix (Temporal Test Split)", fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved -> %s", path)


def plot_feature_importance(model, feature_names: list, path: Path, top_n: int = 20):
    """Save feature importance bar chart (RF models only)."""
    if not hasattr(model, "feature_importances_"):
        log.info("Model has no feature_importances_ — skipping plot.")
        return

    imp     = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:top_n]
    top_imp = imp[top_idx]
    top_lbl = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_imp[::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_lbl[::-1], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"GVMS — Top {top_n} Feature Importances", fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Feature importance saved -> %s", path)


def model_size_bytes(path: Path) -> int:
    return path.stat().st_size


def main():
    log.info("=" * 60)
    log.info("  GVMS — Evaluation Report Generator")
    log.info("=" * 60)

    # -- Load artefacts --------------------------------------------------------
    model, scaler, meta = load_artefacts()
    feature_names = meta["feature_names"]
    include_vs    = meta["include_vs"]
    window_size   = meta["window_size"]
    step_size     = meta["step_size"]
    int_to_label  = {int(k): v for k, v in meta["int_to_label"].items()}

    log.info("Loaded model : %s", meta["model_name"])
    log.info("Features     : %d  (include_vs=%s)", len(feature_names), include_vs)

    # -- Reload data and extract windows --------------------------------------
    df = validate_and_clean(load_all_csvs(ML_DIR / "data"))
    X, y, _ = extract_windows(df, window_size=window_size,
                               step_size=step_size, include_vs=include_vs)

    # Temporal split (reproduce training split)
    from ml.train import temporal_train_test_split
    _, X_test, _, y_test = temporal_train_test_split(X, y)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # -- Confusion matrix ------------------------------------------------------
    labels_used  = sorted(np.unique(np.concatenate([y_test, y_pred])))
    label_names  = [int_to_label[i] for i in labels_used]
    cm           = confusion_matrix(y_test, y_pred, labels=labels_used)

    plot_confusion_matrix(cm, label_names, OUT_DIR / "confusion_matrix.png")

    # -- Per-class metrics -----------------------------------------------------
    report_str = classification_report(
        y_test, y_pred, labels=labels_used, target_names=label_names
    )
    report_dict = classification_report(
        y_test, y_pred, labels=labels_used, target_names=label_names,
        output_dict=True,
    )
    log.info("\n%s", report_str)
    (OUT_DIR / "classification_report.txt").write_text(report_str)

    # -- Feature importance ---------------------------------------------------
    plot_feature_importance(model, feature_names, OUT_DIR / "feature_importance.png")

    # -- Model size & latency -------------------------------------------------
    model_sz = model_size_bytes(MODEL_DIR / "model.pkl")
    log.info("Model file size: %.1f KB", model_sz / 1024)

    import time
    t0 = time.perf_counter()
    for _ in range(100):
        model.predict(X_test_scaled[:1])
    latency_ms = (time.perf_counter() - t0) / 100 * 1000
    log.info("Single-window inference latency: %.3f ms", latency_ms)

    # -- Summary JSON ---------------------------------------------------------
    summary = {
        "model_name":         meta["model_name"],
        "accuracy":           meta["training_accuracy"],
        "macro_f1":           meta["macro_f1"],
        "model_size_kb":      round(model_sz / 1024, 1),
        "inference_latency_ms": round(latency_ms, 3),
        "n_features":         len(feature_names),
        "window_size":        window_size,
        "step_size":          step_size,
        "per_class_f1":       {
            k: round(v["f1-score"], 4)
            for k, v in report_dict.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix":   cm.tolist(),
        "confusion_labels":   label_names,
        "evaluation_note":    meta["evaluation_note"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary saved -> %s", OUT_DIR / "summary.json")

    # -- Print summary table ---------------------------------------------------
    print("\n" + "=" * 55)
    print("  GVMS Evaluation Summary")
    print("=" * 55)
    print(f"  Model          : {summary['model_name']}")
    print(f"  Accuracy       : {summary['accuracy']:.4f}")
    print(f"  Macro-F1       : {summary['macro_f1']:.4f}")
    print(f"  Model size     : {summary['model_size_kb']} KB")
    print(f"  Infer latency  : {summary['inference_latency_ms']:.3f} ms/window")
    print(f"  Window size    : {window_size} samples")
    print(f"  Step size      : {step_size} samples")
    print("-" * 55)
    print("  Per-class F1:")
    for cls, f1 in summary["per_class_f1"].items():
        print(f"    {cls:<22} {f1:.4f}")
    print("=" * 55)
    print(f"  NOTE: {summary['evaluation_note']}")
    print("=" * 55)

    return summary


if __name__ == "__main__":
    main()
