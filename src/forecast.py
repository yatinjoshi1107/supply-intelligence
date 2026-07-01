"""
MODULE 5 -- MONTHLY FORECAST MODEL (scenario propagation, NOT time-series)
We have ONE month of data, so a trained temporal forecaster would be fiction.
Instead the business supplies expected lead volume by source; we propagate it
through OBSERVED June conversion rates by segment. Uncertainty comes from:
  - Wilson intervals on each source's registration proportion, and
  - a Monte-Carlo bootstrap that resamples per-source binomial outcomes.
ASSUMPTIONS (stated in output): (1) June conversion rates hold next month;
(2) leads within a source are exchangeable; (3) RM allocation shares persist.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import wilson_interval, save_table
from .config import load_config


def _source_rates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("source")
    r = pd.DataFrame({"june_leads": g.size(), "june_reg": g["registered"].sum()})
    r["reg_rate"] = r["june_reg"] / r["june_leads"]
    return r


def forecast(df: pd.DataFrame, expected_leads_by_source: dict | None = None,
             cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    rates = _source_rates(df)

    # Default plan = repeat June volumes.
    if expected_leads_by_source is None:
        expected_leads_by_source = rates["june_leads"].to_dict()

    z = {0.90: 1.645, 0.95: 1.96}.get(cfg["forecast"]["confidence_level"], 1.645)
    rows = []
    for src, rate in rates["reg_rate"].items():
        n_plan = int(expected_leads_by_source.get(src, 0))
        succ = int(rates.loc[src, "june_reg"])
        base_n = int(rates.loc[src, "june_leads"])
        lo_r, p_r, hi_r = wilson_interval(succ, base_n, z=z)
        rows.append({
            "source": src, "planned_leads": n_plan,
            "reg_rate": round(rate, 4),
            "expected_registrations": round(n_plan * rate, 1),
            "ci_low": round(n_plan * lo_r, 1),
            "ci_high": round(n_plan * hi_r, 1),
        })
    by_source = pd.DataFrame(rows).sort_values("expected_registrations", ascending=False)

    # Bootstrap total for a combined CI.
    rng = np.random.default_rng(cfg["model"]["random_state"])
    iters = cfg["forecast"]["bootstrap_iterations"]
    totals = np.zeros(iters)
    for src, rate in rates["reg_rate"].items():
        n_plan = int(expected_leads_by_source.get(src, 0))
        totals += rng.binomial(n_plan, min(max(rate, 0), 1), size=iters)
    a = (1 - cfg["forecast"]["confidence_level"]) / 2
    total = {
        "point": round(by_source["expected_registrations"].sum(), 1),
        "ci_low": round(float(np.quantile(totals, a)), 1),
        "ci_high": round(float(np.quantile(totals, 1 - a)), 1),
        "confidence_level": cfg["forecast"]["confidence_level"],
    }

    # Expected registrations by RM: allocate planned leads by June RM share,
    # scaled by each RM's efficiency (actual/expected on June).
    rm_share = df.groupby("rm_name").size() / len(df)
    total_planned = sum(expected_leads_by_source.values())
    rm_eff = (df.groupby("rm_name")["registered"].sum() /
              df.groupby("rm_name")["p_register"].sum()).replace([np.inf, np.nan], 1.0)
    rm_base_rate = df["registered"].mean()
    by_rm = pd.DataFrame({
        "planned_leads": (rm_share * total_planned).round(0),
        "efficiency": rm_eff.round(3),
    })
    by_rm["expected_registrations"] = (
        by_rm["planned_leads"] * rm_base_rate * by_rm["efficiency"]).round(1)
    by_rm = by_rm.sort_values("expected_registrations", ascending=False)

    assumptions = [
        "June per-source conversion rates persist next month.",
        "Leads within a source are exchangeable (i.i.d. Bernoulli).",
        "RM allocation shares and relative efficiency persist.",
        f"Intervals: Wilson (per source) + {iters}-iter binomial bootstrap (total).",
    ]

    t = cfg["paths"]["tables_dir"]
    save_table(by_source, f"{t}/m5_forecast_by_source.csv", index=False)
    save_table(by_rm, f"{t}/m5_forecast_by_rm.csv")
    return {"by_source": by_source, "by_rm": by_rm, "total": total,
            "assumptions": assumptions}
