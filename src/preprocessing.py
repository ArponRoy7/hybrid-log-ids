import pandas as pd
import re

def find_col(df, keys):
    for c in df.columns:
        cl = c.lower()
        for k in keys:
            if k in cl:
                return c
    return None

def parse_seq(s):
    s = str(s).strip().strip("[]")
    return [x.strip() for x in s.split(",") if x.strip()]

def load_data(label_file, trace_file):
    df_label = pd.read_csv(label_file)
    df_trace = pd.read_csv(trace_file)

    blk_T = find_col(df_trace, ["block", "blk"])
    feat_T = find_col(df_trace, ["feature"])

    traces = df_trace[[blk_T, feat_T]].copy()
    traces.columns = ["block_id", "sequence"]
    traces["events"] = traces["sequence"].apply(parse_seq)

    blk_L = find_col(df_label, ["block", "blk"])
    lab_L = find_col(df_label, ["label", "anomaly"])

    labels = df_label[[blk_L, lab_L]].copy()
    labels.columns = ["block_id", "label_raw"]

    def to01(x):
        s = str(x).lower()
        if "anom" in s or s in ("1","true","yes"): return 1
        return 0

    labels["label"] = labels["label_raw"].apply(to01)

    data = traces.merge(labels[["block_id","label"]], on="block_id")

    return data