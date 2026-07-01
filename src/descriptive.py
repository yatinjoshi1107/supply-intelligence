"""
MODULE 1 -- DESCRIPTIVE ANALYTICS
Executive, RM, Source, Campaign and Funnel dashboards. Pure 'what happened'.
All outputs are returned as DataFrames AND written to outputs/tables.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import safe_div, pct, save_table
from .config import load_config


def executive_kpis(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n = len(df)
    reg = int(df["registered"].sum())
    connected = int(df["is_connected"].sum())
    kpis = {
        "Total Leads": n,
        "Registered": reg,
        "Registration %": pct(safe_div(reg, n)),
        "Connected %": pct(safe_div(connected, n)),
        "Avg Talk Time (s)": round(df["talk_time"].mean(), 1),
        "Avg Auto Dials": round(df.get("auto_dials", pd.Series([0])).mean(), 2),
        "Avg Manual Dials": round(df.get("manual_dials", pd.Series([0])).mean(), 2),
        "Docs-Complete Leads": int(df["completed_docs"].sum()),
        "Doc Completion %": pct(safe_div(df["completed_docs"].sum(), n)),
        "Reg | Docs Complete %": pct(safe_div(reg, max(df["completed_docs"].sum(), 1))),
        "Monthly Target": cfg["business"]["monthly_registration_target"],
        "Gap to Target": cfg["business"]["monthly_registration_target"] - reg,
    }
    return pd.DataFrame([kpis]).T.rename(columns={0: "Value"})


def _group_metrics(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)
    out = pd.DataFrame({
        "leads": g.size(),
        "registrations": g["registered"].sum(),
        "docs_complete": g["completed_docs"].sum(),
        "connected": g["is_connected"].sum(),
        "avg_talk_time": g["talk_time"].mean().round(1),
        "avg_auto_dials": g["auto_dials"].mean().round(2) if "auto_dials" in df else 0,
        "avg_manual_dials": g["manual_dials"].mean().round(2) if "manual_dials" in df else 0,
    })
    out["lead_share_%"] = pct(safe_div(out["leads"], len(df)))
    out["registration_%"] = pct(safe_div(out["registrations"], out["leads"]))
    out["connected_%"] = pct(safe_div(out["connected"], out["leads"]))
    out["doc_completion_%"] = pct(safe_div(out["docs_complete"], out["leads"]))
    return out


def rm_dashboard(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = _group_metrics(df, "rm_name")
    # Disposition-bucket rates per RM.
    for bucket in ["not_interested", "busy", "dnp", "follow_up"]:
        col = (df["sub_bucket"] == bucket).astype(int)
        out[f"{bucket}_%"] = pct(safe_div(df.assign(_b=col).groupby("rm_name")["_b"].sum(), out["leads"]))
    # Incomplete lead share.
    inc = (df["lead_stage"] == "Incomplete Lead").astype(int)
    out["incomplete_%"] = pct(safe_div(df.assign(_i=inc).groupby("rm_name")["_i"].sum(), out["leads"]))
    out = out.sort_values("registrations", ascending=False)
    out["rm_rank_raw"] = range(1, len(out) + 1)
    return out


def source_dashboard(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = _group_metrics(df, "source")
    # Drop-off = did not register.
    out["dropoff_%"] = (100 - out["registration_%"]).round(2)
    return out.sort_values("registration_%", ascending=False)


def campaign_dashboard(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = _group_metrics(df, "campaign")
    out["dropoff_%"] = (100 - out["registration_%"]).round(2)
    out["contribution_%"] = pct(safe_div(out["registrations"], df["registered"].sum()))
    return out.sort_values("registrations", ascending=False)


def funnel(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stage-by-stage funnel with leakage. Uses observable milestones rather
    than only Lead Stage, because the document cliff is the real story."""
    n = len(df)
    stages = [
        ("1. Leads Created", n),
        ("2. RM Assigned", int((df.get("rm_unassigned_flag", 0) == 0).sum())),
        ("3. Connected", int(df["is_connected"].sum())),
        ("4. Any Document Uploaded", int((df["docs_uploaded"] > 0).sum())),
        ("5. All Documents Complete", int(df["completed_docs"].sum())),
        ("6. Registration Requested+", int(df["lead_stage"].isin(["Registration Requested", "Registered"]).sum())),
        ("7. Registered", int(df["registered"].sum())),
    ]
    fdf = pd.DataFrame(stages, columns=["stage", "count"])
    fdf["pct_of_top"] = pct(safe_div(fdf["count"], n))
    fdf["step_conversion_%"] = pct(safe_div(fdf["count"], fdf["count"].shift(1).fillna(n)))
    fdf["leakage_count"] = (fdf["count"].shift(1).fillna(n) - fdf["count"]).astype(int)
    fdf["leakage_%_of_step"] = (100 - fdf["step_conversion_%"]).round(2)
    return fdf


def run(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    t = cfg["paths"]["tables_dir"]
    results = {
        "executive": executive_kpis(df, cfg),
        "rm": rm_dashboard(df, cfg),
        "source": source_dashboard(df, cfg),
        "campaign": campaign_dashboard(df, cfg),
        "funnel": funnel(df, cfg),
    }
    for name, tbl in results.items():
        save_table(tbl, f"{t}/m1_{name}.csv")
    return results
