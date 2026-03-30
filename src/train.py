import os
import numpy as np
import joblib

from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data
from src.feature_engineering import prepare_tfidf, compute_sbert
from src.model import train_models


PROCESSED_DIR = "data/processed"


def run_training():

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ---------- CHECK IF PROCESSED DATA EXISTS ----------
    X_path = os.path.join(PROCESSED_DIR, "X_all.npz")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("⚡ Loading processed dataset...")

        X_all = load_npz(X_path)
        y = np.load(y_path)

    else:
        print("🔄 Creating processed dataset...")

        data = load_data(
            "data/raw/anomaly_label.csv",
            "data/raw/Event_traces.csv"
        )

        # ---------- TF-IDF ----------
        X, tfidf = prepare_tfidf(data)

        # ---------- SBERT (OPTIONAL) ----------
        USE_SBERT = False

        if USE_SBERT:
            print("🧠 Computing SBERT embeddings...")
            X_sem = compute_sbert(data, None)
        else:
            print("⚡ Using dummy semantic features...")
            X_sem = np.zeros((X.shape[0], 384))

        # ---------- COMBINE ----------
        X_all = hstack([X, X_sem])
        X_all = csr_matrix(X_all)

        y = data["label"].values

        # ---------- SAVE ----------
        print("💾 Saving processed dataset...")
        save_npz(X_path, X_all)
        np.save(y_path, y)

        joblib.dump(tfidf, "models/tfidf.pkl")

    # ---------- TRAIN ----------
    print("🚀 Training model...")

    tr, te = train_test_split(
        range(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_tr, y_tr = X_all[tr], y[tr]

    lr, iforest = train_models(X_tr, y_tr)

    # ---------- SAVE MODELS ----------
    print("💾 Saving models...")
    joblib.dump(lr, "models/lr.pkl")
    joblib.dump(iforest, "models/iforest.pkl")

    # ---------- 🔥 SAVE IF SCALING (IMPORTANT) ----------
    print("📊 Computing Isolation Forest scaling...")

    s_if_all = -iforest.decision_function(X_all)

    if_min = s_if_all.min()
    if_max = s_if_all.max()

    np.save("models/if_min.npy", if_min)
    np.save("models/if_max.npy", if_max)

    print(f"IF scaling saved: min={if_min:.4f}, max={if_max:.4f}")

    print("✅ Training complete!")


if __name__ == "__main__":
    run_training()