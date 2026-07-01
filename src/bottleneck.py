"""
MODULE 9 -- OPERATIONAL BOTTLENECK DETECTION
Automatically finds the worst point on every axis and attaches a root-cause
narrative. Deterministic + explainable: every flag traces to a counted metric.
"""
from __future__ import annotations
import pandas as pd
from .utils import safe_div, save_table
from .config import load_config


def run(scored_df: pd.DataFrame, funnel_df: pd.DataFrame,
        rm_perf: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    df = scored_df
    findings = []

    # Largest leakage stage.
    leak = funnel_df.iloc[1:].sort_values("leakage_count", ascending=False).iloc[0]
    findings.append({
        "dimension": "Largest funnel leakage",
        "culprit": leak["stage"],
        "metric": f"{leak['leakage_count']:,} leads lost ({leak['leakage_%_of_step']:.1f}% of step)",
        "root_cause": "Prior-stage effort not converting into this milestone; "
                      "prioritise the single biggest drop for intervention."})

    # Worst RM (by efficiency, not raw).
    worst_rm = rm_perf.sort_values("efficiency_index").iloc[0]
    findings.append({
        "dimension": "Worst RM (efficiency-adjusted)",
        "culprit": worst_rm.name,
        "metric": f"efficiency {worst_rm['efficiency_index']:.2f}, "
                  f"{worst_rm['actual_registrations']:.0f} vs "
                  f"{worst_rm['expected_registrations']:.0f} expected",
        "root_cause": "Underperforming on comparable-difficulty leads -> coaching / "
                      "process issue, not lead quality."})

    # Worst source & campaign (min reg rate, min volume floor).
    src = df.groupby("source").agg(n=("registered", "size"), r=("registered", "mean"))
    worst_src = src[src["n"] >= 100]["r"].idxmin()
    findings.append({
        "dimension": "Worst source", "culprit": worst_src,
        "metric": f"{src.loc[worst_src,'r']*100:.2f}% reg on {int(src.loc[worst_src,'n']):,} leads",
        "root_cause": "Low-intent traffic; audit targeting/creative or cut spend."})

    camp = df.groupby("campaign").agg(n=("registered", "size"), r=("registered", "mean"))
    worst_camp = camp[camp["n"] >= 100]["r"].idxmin()
    findings.append({
        "dimension": "Worst campaign", "culprit": worst_camp,
        "metric": f"{camp.loc[worst_camp,'r']*100:.2f}% reg on {int(camp.loc[worst_camp,'n']):,} leads",
        "root_cause": "Campaign attracts poorly-qualified leads relative to peers."})

    # Sub-disposition concentrations.
    total = len(df)
    for bucket, label in [("not_interested", "Highest Not-Interested"),
                          ("busy", "Highest Busy"),
                          ("dnp", "Highest DNP"),
                          ("follow_up", "Highest Follow-up (unclosed)")]:
        cnt = int((df["sub_bucket"] == bucket).sum())
        findings.append({
            "dimension": label, "culprit": bucket,
            "metric": f"{cnt:,} leads ({safe_div(cnt,total)*100:.1f}% of all)",
            "root_cause": {
                "not_interested": "Intent/targeting mismatch upstream in marketing.",
                "busy": "Call-timing / attempt-cadence problem in the dialer.",
                "dnp": "Reachability problem; test SMS/WhatsApp pre-call nudges.",
                "follow_up": "Leads parked in follow-up without closure; enforce SLA.",
            }[bucket]})

    # Doc-invalid & rejection.
    for stage, label in [("Document Invalid", "Highest Document-Invalid"),
                         ("Rejected", "Highest Rejection")]:
        cnt = int((df["lead_stage"] == stage).sum())
        findings.append({
            "dimension": label, "culprit": stage,
            "metric": f"{cnt:,} leads",
            "root_cause": "Upload quality / eligibility screening gap." if stage == "Document Invalid"
                          else "Eligibility or KYC failures; tighten pre-qualification."})

    out = pd.DataFrame(findings)
    save_table(out, f"{cfg['paths']['tables_dir']}/m9_bottlenecks.csv", index=False)
    return out
