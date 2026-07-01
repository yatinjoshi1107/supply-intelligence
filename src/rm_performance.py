"""
MODULE 4 -- RM PERFORMANCE MODEL
Never rank RMs on raw conversion (confounded by the source mix they were dealt).
Instead:
  Expected Registrations = sum of model P(register) over the RM's leads
                           (what an average process would achieve on THESE leads)
  Actual Registrations   = observed registrations
  Efficiency Index       = Actual / Expected   (>1 = outperformed the hand dealt)
  RM Difficulty Score    = 100 * (1 - mean lead P(register) / overall mean P)
                           positive = harder-than-average leads
  Capacity Utilization   = RM lead share vs an even split
Requires the scored dataframe from Module 3 (needs 'p_register').
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import safe_div, save_table
from .config import load_config


def run(scored_df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = scored_df
    overall_mean_p = df["p_register"].mean()
    n_rms = df["rm_name"].nunique()
    even_share = len(df) / n_rms

    g = df.groupby("rm_name")
    out = pd.DataFrame({
        "leads": g.size(),
        "expected_registrations": g["p_register"].sum().round(1),
        "actual_registrations": g["registered"].sum(),
        "mean_lead_p": g["p_register"].mean(),
        "avg_talk_time": g["talk_time"].mean().round(1),
        "avg_auto_dials": g["auto_dials"].mean().round(2),
        "avg_manual_dials": g["manual_dials"].mean().round(2),
    })
    out["difference"] = (out["actual_registrations"] - out["expected_registrations"]).round(1)
    out["efficiency_index"] = safe_div(out["actual_registrations"], out["expected_registrations"]).round(3)
    out["difficulty_score"] = (100 * (1 - out["mean_lead_p"] / overall_mean_p)).round(1)
    out["capacity_utilization"] = safe_div(out["leads"], even_share).round(3)

    out["verdict"] = np.select(
        [out["efficiency_index"] >= 1.10, out["efficiency_index"] <= 0.90],
        ["Outperformed", "Underperformed"],
        default="On expectation",
    )
    out = out.drop(columns=["mean_lead_p"]).sort_values("efficiency_index", ascending=False)
    out["efficiency_rank"] = range(1, len(out) + 1)

    save_table(out, f"{cfg['paths']['tables_dir']}/m4_rm_performance.csv")
    return out
