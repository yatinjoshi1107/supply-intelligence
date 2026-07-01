"""
MODULE 7 -- TARGET ACHIEVEMENT SIMULATOR
Two scenarios side by side (as approved):
  A) REALISTIC MAX at current volume -- best plausible mix + completion.
  B) WHAT IT TAKES to actually hit the 4,000 target -- solve for the required
     completion rate (volume fixed) and required volume (conversion fixed),
     plus a blended path. Honest about feasibility.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import safe_div, save_table
from .config import load_config


def run(scored_df: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    df = scored_df
    N = len(df)
    target = cfg["business"]["monthly_registration_target"]
    reg_now = int(df["registered"].sum())
    comp_rate = df["completed_docs"].mean()
    reg_given_comp = safe_div(reg_now, df["completed_docs"].sum())
    overall_rate = reg_now / N

    # ---- Scenario A: realistic ceiling at current volume --------------------
    # Best-source rate as an aspirational-but-observed conversion ceiling.
    best_source_rate = df.groupby("source")["registered"].mean().max()
    # Assume completion could plausibly double with strong ops (still < best source funnel).
    plausible_comp = min(comp_rate * 2, 0.30)
    realistic_reg = plausible_comp * N * reg_given_comp
    scenario_a = {
        "scenario": "A. Realistic max @ current volume",
        "assumed_completion_rate": round(plausible_comp, 3),
        "assumed_reg_rate": round(plausible_comp * reg_given_comp, 3),
        "projected_registrations": round(realistic_reg, 0),
        "vs_target": round(realistic_reg - target, 0),
    }

    # ---- Scenario B: what it takes to hit 4,000 -----------------------------
    req_comp_rate = safe_div(target, N * reg_given_comp)     # volume fixed
    req_volume = safe_div(target, overall_rate)              # conversion fixed
    # Blended: hit target with a 1.5x completion AND the volume to match.
    blended_comp = min(comp_rate * 1.75, 0.35)
    blended_volume = safe_div(target, blended_comp * reg_given_comp)
    scenario_b = {
        "scenario": "B. Path to 4,000",
        "req_completion_rate_if_volume_fixed": round(req_comp_rate, 3),
        "req_volume_if_conversion_fixed": int(round(req_volume)),
        "blended_completion_rate": round(blended_comp, 3),
        "blended_required_volume": int(round(blended_volume)),
    }

    # ---- Lever sweep (single-lever what-ifs) --------------------------------
    sweep = []
    for delta in [0.02, 0.05]:   # +2pp, +5pp overall conversion
        sweep.append({"lever": f"+{int(delta*100)}pp overall conversion",
                      "new_registrations": round((overall_rate + delta) * N, 0)})
    for mult in [1.25, 1.5, 2.0]:
        sweep.append({"lever": f"x{mult} lead volume (same conversion)",
                      "new_registrations": round(overall_rate * N * mult, 0)})
    for cpp in [0.05, 0.10]:
        sweep.append({"lever": f"+{int(cpp*100)}pp document completion",
                      "new_registrations": round(reg_now + cpp * N * reg_given_comp, 0)})
    sweep_df = pd.DataFrame(sweep)
    sweep_df["gap_to_target"] = (target - sweep_df["new_registrations"]).astype(int)

    summary = pd.DataFrame([
        {"metric": "Business Target", "value": target},
        {"metric": "Current Registrations", "value": reg_now},
        {"metric": "Gap", "value": target - reg_now},
        {"metric": "Current Reg Rate", "value": round(overall_rate, 4)},
        {"metric": "Current Completion Rate", "value": round(comp_rate, 4)},
        {"metric": "Reg | Complete", "value": round(reg_given_comp, 4)},
    ])

    t = cfg["paths"]["tables_dir"]
    save_table(summary, f"{t}/m7_target_summary.csv", index=False)
    save_table(sweep_df, f"{t}/m7_lever_sweep.csv", index=False)
    save_table(pd.DataFrame([scenario_a, scenario_b]).T, f"{t}/m7_scenarios.csv")

    return {"summary": summary, "scenario_a": scenario_a,
            "scenario_b": scenario_b, "sweep": sweep_df}
