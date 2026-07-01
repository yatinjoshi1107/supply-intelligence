#!/usr/bin/env python3
"""
SUPPLY INTELLIGENCE & PREDICTIVE ONBOARDING MODEL -- pipeline orchestrator.
Runs Modules 0-9 in order and writes tables, figures and an executive report.
Usage:  python run_pipeline.py
New month? Point config.paths.raw_file at the new file, set analysis_month,
and re-run. No code changes required.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, ensure_dirs
from src.utils import banner
from src import (data_loader, features, descriptive, lead_quality, predictive,
                 rm_performance, forecast, prescriptive, target_simulator,
                 marketing, bottleneck)


def make_figures(desc, cfg):
    fdir = cfg["paths"]["figures_dir"]
    f = desc["funnel"]
    plt.figure(figsize=(9, 5))
    plt.barh(f["stage"][::-1], f["count"][::-1], color="#2b6cb0")
    plt.title("Onboarding Funnel — leakage at each stage (June)")
    plt.xlabel("Leads"); plt.tight_layout()
    plt.savefig(f"{fdir}/funnel.png", dpi=120); plt.close()

    s = desc["source"].sort_values("registration_%")
    plt.figure(figsize=(7, 4))
    plt.barh(s.index, s["registration_%"], color="#38a169")
    plt.title("Registration % by Source"); plt.xlabel("Registration %")
    plt.tight_layout(); plt.savefig(f"{fdir}/source_reg.png", dpi=120); plt.close()


def main():
    cfg = load_config(); ensure_dirs(cfg)
    print(banner("MODULE 0 — LOAD, CLEAN, ENGINEER"))
    df = data_loader.load_and_clean(cfg)
    df = features.engineer(df, cfg)
    print(f"  Rows after engineering: {len(df):,}")

    print(banner("MODULE 1 — DESCRIPTIVE"))
    desc = descriptive.run(df, cfg)
    print(desc["executive"].to_string())

    print(banner("MODULE 2 — LEAD QUALITY"))
    lq = lead_quality.compute(df, cfg)
    df = lq["df"]
    print("Top quality drivers:\n", lq["weights"].head(8).to_string(index=False))

    print(banner("MODULE 3 — PREDICTIVE (two-stage)"))
    pred = predictive.run(df, cfg)
    df = pred["df"]
    print(pred["comparison"].to_string())
    print("Chosen:", pred["chosen_model"], "—", pred["rationale"])
    print(f"Stage B (reg|complete): {pred['stage_b']['rate']*100:.1f}% "
          f"[{pred['stage_b']['lo']*100:.1f}, {pred['stage_b']['hi']*100:.1f}]")
    print("Risk mix:\n", pred["risk_summary"].to_string())

    print(banner("MODULE 4 — RM PERFORMANCE"))
    rmp = rm_performance.run(df, cfg)
    print(rmp[["leads", "expected_registrations", "actual_registrations",
               "efficiency_index", "difficulty_score", "verdict"]].to_string())

    print(banner("MODULE 5 — FORECAST"))
    fc = forecast.forecast(df, None, cfg)
    print(fc["by_source"].to_string(index=False))
    print("TOTAL:", fc["total"])

    print(banner("MODULE 6 — PRESCRIPTIVE"))
    presc = prescriptive.run(df, rmp, cfg)
    print(presc["interventions"][["rank", "intervention", "expected_uplift",
                                  "projected_registrations"]].to_string(index=False))

    print(banner("MODULE 7 — TARGET SIMULATOR"))
    tgt = target_simulator.run(df, cfg)
    print(tgt["summary"].to_string(index=False))
    print("Scenario A:", tgt["scenario_a"])
    print("Scenario B:", tgt["scenario_b"])
    print(tgt["sweep"].to_string(index=False))

    print(banner("MODULE 8 — MARKETING INTELLIGENCE"))
    mkt = marketing.run(df, cfg)
    print(mkt["source_ranking"].to_string())
    print("Highlights:", mkt["highlights"])

    print(banner("MODULE 9 — BOTTLENECKS"))
    bn = bottleneck.run(df, desc["funnel"], rmp, cfg)
    print(bn.to_string(index=False))

    make_figures(desc, cfg)
    write_report(cfg, desc, pred, rmp, fc, presc, tgt, mkt, bn)
    print(banner("DONE — see outputs/"))


def write_report(cfg, desc, pred, rmp, fc, presc, tgt, mkt, bn):
    r = cfg["paths"]["reports_dir"] + "/executive_report.md"
    ex = desc["executive"]["Value"]
    with open(r, "w") as f:
        f.write(f"# Supply Intelligence — Executive Report ({cfg['project']['analysis_month']})\n\n")
        f.write("## What happened\n")
        f.write(f"- Total leads: **{int(ex['Total Leads']):,}**, Registered: "
                f"**{int(ex['Registered'])}** ({ex['Registration %']}%).\n")
        f.write(f"- Document completion: **{ex['Doc Completion %']}%**; once complete, "
                f"registration runs at **{ex['Reg | Docs Complete %']}%**.\n")
        f.write(f"- Gap to target ({int(ex['Monthly Target'])}): **{int(ex['Gap to Target'])}**.\n\n")
        f.write("## Why (bottlenecks)\n")
        for _, row in bn.head(4).iterrows():
            f.write(f"- **{row['dimension']}**: {row['culprit']} — {row['metric']}. {row['root_cause']}\n")
        f.write("\n## What to do (ranked interventions)\n")
        for _, row in presc["interventions"].iterrows():
            f.write(f"{int(row['rank'])}. {row['intervention']} → +{row['expected_uplift']:.0f} "
                    f"regs (cum {row['projected_registrations']:.0f}).\n")
        f.write(f"\n## Path to target\n- Scenario A (realistic max): "
                f"{tgt['scenario_a']['projected_registrations']:.0f} regs.\n")
        f.write(f"- Scenario B: to hit {int(ex['Monthly Target'])}, either completion "
                f"→ {tgt['scenario_b']['req_completion_rate_if_volume_fixed']*100:.0f}% "
                f"at current volume, or volume → "
                f"{tgt['scenario_b']['req_volume_if_conversion_fixed']:,} at current conversion.\n")
    return r


if __name__ == "__main__":
    main()
