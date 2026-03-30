import numpy as np
import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.preprocessing import load_data
from src.feature_engineering import prepare_tfidf
from src.model import train_models, predict_scores


def run_zero_day():

    print("🔍 Loading data...")
    data = load_data(
        "data/raw/anomaly_label.csv",
        "data/raw/Event_traces.csv"
    )

    # ---------------- NORMAL vs ANOMALY SPLIT ----------------
    normal = data[data["label"] == 0]
    anomaly = data[data["label"] == 1]

    print(f"Normal: {len(normal)}, Anomaly: {len(anomaly)}")

    # ---------------- SIMULATE ZERO-DAY ----------------
    # Split anomalies → half seen, half unseen
    anom_train, anom_test = train_test_split(
        anomaly,
        test_size=0.5,
        random_state=42
    )

    # Training data: normal + partial anomalies
    train_data = pd.concat([normal, anom_train]).reset_index(drop=True)

    # Test data: ONLY unseen anomalies (zero-day)
    test_data = anom_test.reset_index(drop=True)

    print(f"Train size: {len(train_data)}")
    print(f"Zero-day test size: {len(test_data)}")

    # ---------------- FEATURE ENGINEERING ----------------
    print("⚙️ Creating TF-IDF features...")

    X_train, tfidf = prepare_tfidf(train_data)
    X_test = tfidf.transform(
        test_data["events"].apply(lambda x: " ".join(x[:60]))
    )

    # Dummy semantic (same as your pipeline)
    X_sem_train = np.zeros((X_train.shape[0], 384))
    X_sem_test = np.zeros((X_test.shape[0], 384))

    X_train_all = hstack([X_train, X_sem_train])
    X_test_all = hstack([X_test, X_sem_test])

    X_train_all = csr_matrix(X_train_all)
    X_test_all = csr_matrix(X_test_all)

    y_train = train_data["label"].values
    y_test = test_data["label"].values  # all 1s

    # ---------------- TRAIN MODEL ----------------
    print("🚀 Training model...")
    lr, iforest = train_models(X_train_all, y_train)

    # ---------------- PREDICT ----------------
    print("⚡ Testing on zero-day attacks...")
    score = predict_scores(lr, iforest, X_test_all)

    # Use simple threshold (or reuse your thr_best)
    thr = 0.5
    y_pred = (score >= thr).astype(int)

    # ---------------- METRICS ----------------
    tp = np.sum(y_pred == 1)
    fn = np.sum(y_pred == 0)

    recall = tp / (tp + fn + 1e-9)

    print("\n=========== ZERO-DAY RESULT ===========")
    print(f"Detected attacks (TP): {tp}")
    print(f"Missed attacks (FN): {fn}")
    print(f"Zero-Day Recall: {recall:.4f}")


if __name__ == "__main__":
    run_zero_day()