import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Load models
lr = joblib.load("models/lr.pkl")
iforest = joblib.load("models/iforest.pkl")
tfidf = joblib.load("models/tfidf.pkl")

THR = 0.63   # use your thr_best

def predict_sequence(events):
    event_str = " ".join(events[:60])

    X_tfidf = tfidf.transform([event_str])

    X_sem = np.zeros((1, 384))  # same as training

    X_all = hstack([X_tfidf, X_sem])
    X_all = csr_matrix(X_all)

    s_lr = lr.predict_proba(X_all)[0,1]
    s_if = -iforest.decision_function(X_all)[0]
    s_if = float(s_if)

    score = 0.7*s_lr + 0.3*s_if

    pred = int(score >= THR)

    return pred, score