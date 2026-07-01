"""
MODULE 2 -- LEAD QUALITY MODEL (0-100)
The score must be usable on a DAY-ONE lead, so it uses only early-lifecycle
features (never the document columns or the final stage). Weights are NOT
hand-picked: they are the standardized logistic-regression coefficients from
the document-completion model, which makes every weight explainable and
data-driven ('this feature matters this much because it moves completion odds
by X'). Score is a rescaled linear predictor -> monotonic and transparent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .features import build_model_matrix
from .utils import save_table
from .config import load_config


def compute(df: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    X, y, feats = build_model_matrix(df, cfg)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.astype(float))

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(Xs, y)

    # Linear predictor (log-odds) -> min-max rescale to 0-100.
    lp = lr.decision_function(Xs)
    lo, hi = lp.min(), lp.max()
    score = (lp - lo) / (hi - lo + 1e-9) * (
        cfg["lead_quality"]["score_max"] - cfg["lead_quality"]["score_min"]
    ) + cfg["lead_quality"]["score_min"]
    df = df.copy()
    df["lead_quality_score"] = score.round(1)

    # Explainable weight table: standardized coefficient per feature.
    weights = (
        pd.DataFrame({"feature": feats, "std_coefficient": lr.coef_[0]})
        .assign(abs_weight=lambda d: d["std_coefficient"].abs())
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )
    weights["direction"] = np.where(weights["std_coefficient"] >= 0, "raises quality", "lowers quality")
    weights["relative_importance_%"] = (
        weights["abs_weight"] / weights["abs_weight"].sum() * 100
    ).round(2)

    # Aggregations requested by the brief.
    by_rm = df.groupby("rm_name")["lead_quality_score"].mean().round(1).sort_values(ascending=False)
    by_source = df.groupby("source")["lead_quality_score"].mean().round(1).sort_values(ascending=False)
    by_campaign = df.groupby("campaign")["lead_quality_score"].mean().round(1).sort_values(ascending=False)

    t = cfg["paths"]["tables_dir"]
    save_table(weights, f"{t}/m2_quality_weights.csv", index=False)
    save_table(by_rm.to_frame("avg_lead_quality"), f"{t}/m2_quality_by_rm.csv")
    save_table(by_source.to_frame("avg_lead_quality"), f"{t}/m2_quality_by_source.csv")
    save_table(by_campaign.to_frame("avg_lead_quality"), f"{t}/m2_quality_by_campaign.csv")

    return {
        "df": df,
        "weights": weights,
        "by_rm": by_rm,
        "by_source": by_source,
        "by_campaign": by_campaign,
        "model": lr,
        "scaler": scaler,
        "features": feats,
    }
