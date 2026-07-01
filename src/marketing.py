"""
MODULE 8 -- MARKETING INTELLIGENCE
Ranks sources and campaigns on volume, conversion, quality and efficiency, then
recommends a budget reallocation. 'Efficiency' = registrations per 1,000 leads,
a proxy for cost-efficiency (swap in real CPL when available -> ROI per source).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import safe_div, save_table
from .config import load_config


def _rank_block(df, by, quality_col="lead_quality_score"):
    g = df.groupby(by)
    out = pd.DataFrame({
        "leads": g.size(),
        "registrations": g["registered"].sum(),
        "reg_rate_%": (g["registered"].mean() * 100).round(2),
        "avg_quality": g[quality_col].mean().round(1) if quality_col in df else np.nan,
    })
    out["volume_share_%"] = (out["leads"] / len(df) * 100).round(2)
    out["reg_share_%"] = (out["registrations"] / df["registered"].sum() * 100).round(2)
    out["reg_per_1000_leads"] = (safe_div(out["registrations"], out["leads"]) * 1000).round(1)
    return out


def run(scored_df: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    df = scored_df
    src = _rank_block(df, "source").sort_values("reg_per_1000_leads", ascending=False)
    camp = _rank_block(df, "campaign").sort_values("reg_per_1000_leads", ascending=False)

    # Budget reallocation: hold total leads constant, move share toward the
    # most efficient sources. Simple, explainable proportional-to-efficiency mix.
    eff = src["reg_per_1000_leads"].clip(lower=0.1)
    new_share = eff / eff.sum()
    total_leads = int(src["leads"].sum())
    realloc = pd.DataFrame({
        "current_leads": src["leads"],
        "current_share_%": (src["leads"] / total_leads * 100).round(1),
        "recommended_share_%": (new_share * 100).round(1),
        "recommended_leads": (new_share * total_leads).round(0),
        "source_reg_rate": src["reg_rate_%"] / 100,
    })
    realloc["expected_regs_current"] = (realloc["current_leads"] * realloc["source_reg_rate"]).round(0)
    realloc["expected_regs_recommended"] = (realloc["recommended_leads"] * realloc["source_reg_rate"]).round(0)
    uplift = realloc["expected_regs_recommended"].sum() - realloc["expected_regs_current"].sum()

    highlights = {
        "highest_quality_source": src["avg_quality"].idxmax() if "avg_quality" in src else None,
        "highest_conversion_source": src["reg_rate_%"].idxmax(),
        "highest_volume_source": src["leads"].idxmax(),
        "most_efficient_campaign": camp["reg_per_1000_leads"].idxmax(),
        "reallocation_uplift_registrations": round(float(uplift), 0),
    }

    t = cfg["paths"]["tables_dir"]
    save_table(src, f"{t}/m8_source_ranking.csv")
    save_table(camp, f"{t}/m8_campaign_ranking.csv")
    save_table(realloc, f"{t}/m8_budget_reallocation.csv")
    return {"source_ranking": src, "campaign_ranking": camp,
            "reallocation": realloc, "highlights": highlights}
