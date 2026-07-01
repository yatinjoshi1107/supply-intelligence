"""
MODULE 0a -- DATA CLEANING FRAMEWORK
Loads the raw file, renames to canonical columns, applies documented cleaning
rules, and returns a tidy DataFrame. Every rule is logged so leadership can
audit exactly what was changed and why.
"""
from __future__ import annotations
import pandas as pd
from .config import load_config


def _reverse_map(colmap: dict) -> dict:
    """canonical->actual  =>  actual->canonical"""
    return {v: k for k, v in colmap.items()}


def load_and_clean(cfg: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    log = []

    df = pd.read_excel(cfg["paths"]["raw_file"], sheet_name=cfg["paths"]["raw_sheet"])
    log.append(f"Loaded {len(df):,} rows x {df.shape[1]} cols from raw file.")

    # 1. Drop constant / non-informative columns (e.g. all-zero May dials).
    drop = [c for c in cfg["drop_columns"] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        log.append(f"Dropped non-informative columns: {drop}")

    # 2. Rename to canonical names (only those present).
    rev = _reverse_map(cfg["columns"])
    present = {actual: canon for actual, canon in rev.items() if actual in df.columns}
    df = df.rename(columns=present)
    missing = [c for c in cfg["columns"] if c not in df.columns]
    if missing:
        log.append(f"WARNING: expected canonical columns missing from data: {missing}")

    # 3. De-duplicate on lead_id (a lead must be unique).
    before = len(df)
    df = df.drop_duplicates(subset=["lead_id"])
    if len(df) < before:
        log.append(f"Removed {before - len(df)} duplicate lead_id rows.")

    # 4. Type coercions.
    for c in ["talk_time", "auto_dials", "manual_dials"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["created_date", "rm_assign_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 5. Categorical null handling -- keep nulls MEANINGFUL, do not silently fill.
    #    campaign nulls = untagged (often Organic/Google) -> explicit label.
    if "campaign" in df.columns:
        n = df["campaign"].isna().sum()
        df["campaign"] = df["campaign"].fillna("(untagged)")
        log.append(f"Campaign: {n} untagged leads labelled '(untagged)'.")
    #    disposition null = never entered the dialer.
    if "disposition" in df.columns:
        n = df["disposition"].isna().sum()
        df["disposition"] = df["disposition"].fillna("Never Dialed")
        log.append(f"Disposition: {n} leads never dialed -> 'Never Dialed'.")
    if "sub_disposition" in df.columns:
        df["sub_disposition"] = df["sub_disposition"].fillna("Never Dialed")

    # 6. RM assign date null: DO NOT treat as 'unworked'. Data shows some of
    #    these still registered. Flag it instead of dropping.
    if "rm_assign_date" in df.columns:
        df["rm_unassigned_flag"] = df["rm_assign_date"].isna().astype(int)
        log.append(
            f"RM assign date: {int(df['rm_unassigned_flag'].sum())} nulls flagged "
            f"(NOT dropped -- some registered)."
        )

    # 7. Strip whitespace on key string cols.
    for c in ["source", "rm_name", "lead_stage", "allocation_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if verbose:
        print("\n[Module 0a] Data cleaning log")
        for line in log:
            print("  -", line)

    df.attrs["clean_log"] = log
    return df
