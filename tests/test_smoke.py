"""Smoke test: full pipeline runs and produces sane, internally-consistent output."""
from src.config import load_config, ensure_dirs
from src import data_loader, features, descriptive, lead_quality, predictive, rm_performance


def test_pipeline_core():
    cfg = load_config(); ensure_dirs(cfg)
    df = features.engineer(data_loader.load_and_clean(cfg, verbose=False), cfg)
    assert len(df) > 0
    # Outcome and target are binary.
    assert set(df["registered"].unique()) <= {0, 1}
    assert set(df["completed_docs"].unique()) <= {0, 1}
    # Lead quality bounded 0-100.
    lq = lead_quality.compute(df, cfg); df = lq["df"]
    assert df["lead_quality_score"].between(0, 100).all()
    # Predictions are probabilities and calibrated (expected ~ actual within 15%).
    pred = predictive.run(df, cfg); df = pred["df"]
    assert df["p_register"].between(0, 1).all()
    exp, act = df["p_register"].sum(), df["registered"].sum()
    assert abs(exp - act) / act < 0.15, f"miscalibrated: exp {exp:.0f} vs act {act}"
    # RM efficiency centered near 1.
    rmp = rm_performance.run(df, cfg)
    assert 0.8 < rmp["efficiency_index"].median() < 1.2
    print("SMOKE OK: exp %.0f vs act %d | median eff %.2f"
          % (exp, act, rmp["efficiency_index"].median()))


if __name__ == "__main__":
    test_pipeline_core()
