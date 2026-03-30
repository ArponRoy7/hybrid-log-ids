from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import numpy as np

def train_models(X_tr, y_tr):
    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_tr, y_tr)

    iforest = IsolationForest(n_estimators=400, contamination=0.03)
    iforest.fit(X_tr[y_tr==0])

    return lr, iforest

def predict_scores(lr, iforest, X):
    s_lr = lr.predict_proba(X)[:,1]
    s_if = -iforest.decision_function(X)

    s_if = (s_if - s_if.min())/(s_if.max()-s_if.min()+1e-9)

    return 0.7*s_lr + 0.3*s_if