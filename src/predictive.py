"""
MODULE 3 -- PREDICTIVE MODEL (two-stage, approved framing)
    P(register) = P(complete docs | early features)  x  P(register | complete docs)
Stage A is a real classification problem on genuinely early features (no doc
leakage). Stage B is the observed, near-constant completion->registration rate.
We compare Logistic Regression, Decision Tree, and Gradient Boosting, and
KEEP THE SIMPLEST unless GBM beats it by a configured AUC margin. We report
discrimination (ROC-AUC, PR-AUC) AND calibration (Brier) -- because for a
scenario/forecast use-case, calibrated probabilities matter more than raw
accuracy, and plain accuracy is meaningless on a 9% base rate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from .features import build_model_matrix
from .utils import save_table, wilson_interval
from .config import load_config


def _cv_scores(model, X, y, cfg):
    skf = StratifiedKFold(n_splits=cfg["model"]["cv_folds"], shuffle=True,
                          random_state=cfg["model"]["random_state"])
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
    }, proba


def run(df: pd.DataFrame, cfg: dict | None = None):
    cfg = cfg or load_config()
    rs = cfg["model"]["random_state"]
    X, y, feats = build_model_matrix(df, cfg)
    Xv = X.astype(float).values

    candidates = {
        # NOTE: no class_weight="balanced". For a two-stage EXPECTED-VALUE model,
        # probabilities must be CALIBRATED (sum of P ~= observed positives), not
        # separation-optimised. MLE logistic is calibrated in-sample; balanced
        # weighting would inflate every RM's "expected" and make all 18 look bad.
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                               random_state=rs),
        "GradientBoosting": GradientBoostingClassifier(random_state=rs),
    }

    comparison, oof = {}, {}
    for name, mdl in candidates.items():
        scores, proba = _cv_scores(mdl, Xv, y, cfg)
        comparison[name] = scores
        oof[name] = proba
    comp_df = pd.DataFrame(comparison).T.round(4)

    # Model selection: prefer Logistic (most explainable) unless GBM materially better.
    simplest = "LogisticRegression"
    best_complex = comp_df["roc_auc"].idxmax()
    margin = cfg["model"]["gbm_auc_improvement_threshold"]
    if comp_df.loc[best_complex, "roc_auc"] - comp_df.loc[simplest, "roc_auc"] >= margin \
       and best_complex != simplest:
        chosen = best_complex
        rationale = (f"{best_complex} beats LogisticRegression by "
                     f">= {margin} ROC-AUC; complexity justified.")
    else:
        chosen = simplest
        rationale = (f"LogisticRegression retained: no model beats it by the "
                     f"{margin} ROC-AUC explainability margin.")

    # Fit chosen model on all data for scoring + interpretation.
    final = candidates[chosen].fit(Xv, y)
    p_docs = final.predict_proba(Xv)[:, 1]

    # Stage B: observed completion -> registration rate with 90% Wilson CI.
    comp_mask = df["completed_docs"] == 1
    b_succ = int(df.loc[comp_mask, "registered"].sum())
    b_n = int(comp_mask.sum())
    b_lo, b_rate, b_hi = wilson_interval(b_succ, b_n)

    out = df.copy()
    out["p_complete_docs"] = p_docs.round(4)
    out["p_register"] = (p_docs * b_rate).round(4)

    hi_t = cfg["model"]["risk_bands"]["high"]
    md_t = cfg["model"]["risk_bands"]["medium"]
    out["risk_category"] = np.select(
        [out["p_register"] >= hi_t, out["p_register"] >= md_t],
        ["High Probability", "Medium Probability"],
        default="Low Probability",
    )

    # Interpretability artifact.
    if chosen == "LogisticRegression":
        interp = (pd.DataFrame({"feature": feats, "coefficient": final.coef_[0]})
                  .assign(odds_ratio=lambda d: np.exp(d["coefficient"]).round(3))
                  .sort_values("coefficient", key=abs, ascending=False))
    elif chosen == "DecisionTree":
        interp = pd.DataFrame({"rules": [export_text(final, feature_names=feats)]})
    else:
        interp = (pd.DataFrame({"feature": feats, "importance": final.feature_importances_})
                  .sort_values("importance", ascending=False))

    t = cfg["paths"]["tables_dir"]
    save_table(comp_df, f"{t}/m3_model_comparison.csv")
    save_table(interp, f"{t}/m3_model_interpretation.csv", index=False)
    save_table(out[["lead_id", "p_complete_docs", "p_register", "risk_category"]],
               f"{t}/m3_lead_scores.csv", index=False)

    stage_b = {"rate": b_rate, "lo": b_lo, "hi": b_hi, "n": b_n, "successes": b_succ}
    return {
        "df": out, "comparison": comp_df, "chosen_model": chosen,
        "rationale": rationale, "interpretation": interp,
        "stage_b": stage_b, "model": final, "features": feats,
        "risk_summary": out["risk_category"].value_counts(),
    }
