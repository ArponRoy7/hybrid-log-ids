import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve
)

from src.preprocessing import load_data
from src.feature_engineering import prepare_tfidf
from src.model import predict_scores


def run_evaluation():
    print("🔍 Loading data...")

    data = load_data(
        "data/raw/anomaly_label.csv",
        "data/raw/Event_traces.csv"
    )

    # -------- FEATURES --------
    X, tfidf = prepare_tfidf(data)

    # Dummy semantic (same as training)
    X_sem = np.zeros((X.shape[0], 384))

    X_all = hstack([X, X_sem])
    X_all = csr_matrix(X_all)

    y = data["label"].values

    # -------- LOAD MODELS --------
    print("📦 Loading models...")

    lr = joblib.load("models/lr.pkl")
    iforest = joblib.load("models/iforest.pkl")

    # -------- PREDICTION --------
    print("⚡ Running inference...")

    score = predict_scores(lr, iforest, X_all)

    # -------- THRESHOLD SELECTION --------
    print("🎯 Finding best threshold...")

    prec, rec, thr = precision_recall_curve(y, score)

    PREC_FLOOR = 0.80

    idx = np.where(prec >= PREC_FLOOR)[0]

    if len(idx) > 0:
        j = idx[np.argmax(rec[idx])]
        thr_best = thr[j-1] if (j-1) >= 0 and (j-1) < len(thr) else thr[-1]
    else:
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best = int(np.nanargmax(f1))
        thr_best = thr[max(best-1, 0)] if len(thr) > 0 else 0.5

    y_pred = (score >= thr_best).astype(int)

    # -------- METRICS --------
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    P, R, F1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )

    pr_auc = average_precision_score(y, score)

    acc = (tp + tn) / (tp + tn + fp + fn)
    spec = tn / (tn + fp + 1e-9)

    # -------- PRINT --------
    print("\n=========== EVALUATION RESULT ===========")

    print(f"Threshold (precision >= {PREC_FLOOR}): {thr_best:.4f}\n")

    print(f"Accuracy    : {acc:.3f}")
    print(f"Precision   : {P:.3f}")
    print(f"Recall      : {R:.3f}")
    print(f"F1-score    : {F1:.3f}")
    print(f"PR-AUC      : {pr_auc:.3f}")
    print(f"Specificity : {spec:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Total samples: {len(y)}")


if __name__ == "__main__":
    run_evaluation()