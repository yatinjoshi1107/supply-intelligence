"""
MODULE 6 -- PRESCRIPTIVE MODEL (the point of the whole system)
Turns findings into ranked interventions, each with a formula-based expected
registration uplift and a plain-English rationale. All uplifts are computed
from OBSERVED June rates so leadership can trace every number.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import safe_div, save_table
from .config import load_config


def run(scored_df: pd.DataFrame, rm_perf: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    df = scored_df
    N = len(df)
    reg_now = int(df["registered"].sum())
    comp_rate = df["completed_docs"].mean()
    reg_given_comp = safe_div(reg_now, df["completed_docs"].sum())

    interventions = []

    # 1) MARKETING MIX -- reallocate worst source volume to best.
    srt = df.groupby("source").agg(leads=("registered", "size"),
                                   rate=("registered", "mean"))
    worst = srt["rate"].idxmin(); best = srt["rate"].idxmax()
    shift = int(0.30 * srt.loc[worst, "leads"])   # move 30% of worst-source volume
    uplift_mix = shift * (srt.loc[best, "rate"] - srt.loc[worst, "rate"])
    interventions.append({
        "intervention": f"Shift 30% of {worst} volume to {best}",
        "lever": "Marketing", "expected_uplift": round(uplift_mix, 0),
        "rationale": f"{worst} converts at {srt.loc[worst,'rate']*100:.1f}% vs "
                     f"{best} at {srt.loc[best,'rate']*100:.1f}%. Re-pointing "
                     f"{shift:,} leads captures the gap."})

    # 2) DOCUMENT COMPLETION -- the fulcrum. +5pp completion.
    for pp in (0.05, 0.10):
        uplift = pp * N * reg_given_comp
        interventions.append({
            "intervention": f"Raise document completion by {int(pp*100)}pp "
                            f"({comp_rate*100:.1f}% -> {(comp_rate+pp)*100:.1f}%)",
            "lever": "Operations", "expected_uplift": round(uplift, 0),
            "rationale": f"Registration is ~{reg_given_comp*100:.0f}% once all docs "
                         f"are in. Each extra completed pack yields "
                         f"{reg_given_comp:.2f} registrations."})

    # 3) CONNECTED RATE -- connected leads complete docs far more often.
    conn = df.groupby("is_connected")["completed_docs"].mean()
    if 1 in conn.index and 0 in conn.index:
        gap = conn[1] - conn[0]
        not_conn = int((df["is_connected"] == 0).sum())
        recoverable = int(0.20 * not_conn)   # connect 20% more of the unconnected
        uplift_conn = recoverable * gap * reg_given_comp
        interventions.append({
            "intervention": "Connect 20% more of currently-unconnected leads",
            "lever": "Operations", "expected_uplift": round(uplift_conn, 0),
            "rationale": f"Connected leads complete docs at {conn[1]*100:.1f}% vs "
                         f"{conn[0]*100:.1f}% unconnected. Reaching "
                         f"{recoverable:,} more leads lifts completion."})

    # 4) BOTTOM-QUARTILE RM UPLIFT to median efficiency.
    med_eff = rm_perf["efficiency_index"].median()
    bottom = rm_perf[rm_perf["efficiency_index"] < rm_perf["efficiency_index"].quantile(0.25)]
    uplift_rm = float(((med_eff - bottom["efficiency_index"]) *
                       bottom["expected_registrations"]).clip(lower=0).sum())
    interventions.append({
        "intervention": f"Coach bottom-quartile RMs ({len(bottom)}) to median efficiency",
        "lever": "Operations", "expected_uplift": round(uplift_rm, 0),
        "rationale": f"Bottom-quartile RMs run below median efficiency "
                     f"({med_eff:.2f}) on comparable leads; closing the gap is pure "
                     f"execution, no extra lead spend."})

    # 5) REDUCE DOCUMENT INVALID + REJECTION recovery.
    di = int((df["lead_stage"] == "Document Invalid").sum())
    uplift_di = 0.5 * di * reg_given_comp
    interventions.append({
        "intervention": "Recover 50% of Document-Invalid leads via re-upload prompts",
        "lever": "Operations", "expected_uplift": round(uplift_di, 0),
        "rationale": f"{di} leads reached Document Invalid; salvaging half and "
                     f"getting them to a clean pack recovers registrations."})

    out = (pd.DataFrame(interventions)
           .sort_values("expected_uplift", ascending=False)
           .reset_index(drop=True))
    out["rank"] = range(1, len(out) + 1)
    out["cumulative_uplift"] = out["expected_uplift"].cumsum()
    out["projected_registrations"] = reg_now + out["cumulative_uplift"]

    save_table(out, f"{cfg['paths']['tables_dir']}/m6_interventions.csv", index=False)
    return {"interventions": out, "baseline_registrations": reg_now}
