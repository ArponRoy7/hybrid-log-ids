import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re

def prepare_tfidf(data):
    data["event_str"] = data["events"].apply(lambda x: " ".join(x[:60]))
    tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=2)
    X = tfidf.fit_transform(data["event_str"])
    return X, tfidf

def build_template_map(df_tmpl):
    ev = df_tmpl.columns[0]
    txt = df_tmpl.columns[1]
    return dict(zip(df_tmpl[ev].astype(str), df_tmpl[txt].astype(str)))

def events_to_text(events, tmpl_map):
    texts = []
    for e in events[:60]:
        e_id = re.sub(r"\D","",str(e))
        texts.append(tmpl_map.get(e_id,str(e)))
    return texts

def compute_sbert(data, tmpl_map):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(events):
        txt = events_to_text(events, tmpl_map)
        emb = model.encode(txt, convert_to_numpy=True, normalize_embeddings=True)
        return emb.mean(axis=0)

    X_sem = np.vstack([embed(ev) for ev in data["events"]])
    return X_sem