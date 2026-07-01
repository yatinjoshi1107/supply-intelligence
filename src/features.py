"""
MODULE 0b -- FEATURE ENGINEERING FRAMEWORK
Derives the analytic columns every downstream module relies on. Critically:
 - The Stage-A target (completed_docs) and the outcome (registered).
 - EARLY-lifecycle features only for modelling (no document columns leak in).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import load_config


def engineer(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = df.copy()

    # --- Outcome & document progress -----------------------------------------
    reg_stage = cfg["lead_stages"]["registered"]
    df["registered"] = (df["lead_stage"] == reg_stage).astype(int)

    doc_cols = cfg["document_columns"]
    uploaded = cfg["document_uploaded_value"]
    for c in doc_cols:
        if c not in df.columns:
            df[c] = "Not yet uploaded"
    df["docs_uploaded"] = sum((df[c] == uploaded).astype(int) for c in doc_cols)
    n_docs = len(doc_cols)
    # Stage-A target: full document pack complete (the real uncertainty).
    df["completed_docs"] = (df["docs_uploaded"] == n_docs).astype(int)
    df["doc_progress_ratio"] = df["docs_uploaded"] / n_docs

    # --- Contact / connectivity ----------------------------------------------
    conn = cfg["connected_values"]
    df["is_connected"] = df["disposition"].isin(conn).astype(int)

    # --- Sub-disposition buckets (root-cause analytics) ----------------------
    bucket_map = {}
    for bucket, raws in cfg["sub_disposition_map"].items():
        for r in raws:
            bucket_map[r] = bucket
    df["sub_bucket"] = df["sub_disposition"].map(bucket_map).fillna("other")

    # --- Dial intensity & talk-time bands ------------------------------------
    df["total_dials"] = df.get("auto_dials", 0) + df.get("manual_dials", 0)
    df["talk_time"] = df["talk_time"].clip(lower=0)
    df["talk_band"] = pd.cut(
        df["talk_time"],
        bins=[-1, 0, 60, 180, 600, np.inf],
        labels=["none", "0-1min", "1-3min", "3-10min", "10min+"],
    )

    # --- Funnel stage index (for leakage ordering) ---------------------------
    order = {s: i for i, s in enumerate(cfg["lead_stages"]["ordered"])}
    df["stage_index"] = df["lead_stage"].map(order)

    return df


def build_model_matrix(df: pd.DataFrame, cfg: dict):
    """Return (X, y, feature_names) for Stage-A modelling using EARLY features
    only. One-hot encodes categoricals; keeps it interpretable."""
    cat = [c for c in cfg["model"]["features_categorical"] if c in df.columns]
    num = [c for c in cfg["model"]["features_numeric"] if c in df.columns]

    X_cat = pd.get_dummies(df[cat].astype(str), prefix=cat, drop_first=True)
    X_num = df[num].astype(float).reset_index(drop=True)
    X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
    y = df[cfg["model"]["stage_a_target"]].astype(int).values
    return X, y, list(X.columns)
