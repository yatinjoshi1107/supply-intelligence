# Supply Intelligence & Predictive Onboarding Model

An explainable decision-support system for CEO / COO / Head of Supply / Marketing.
It answers: *what happened, why, what happens next month, why, and what to do
to reach the 4,000/month onboarding target.*

Built on June data (10,550 leads, 18 RMs, 3 sources). Config-driven so July and
August drop in with **zero code changes**.

---

## The one finding that shapes everything

Registration is **near-deterministic given document completion**:

| Documents uploaded | Leads | Registered | Rate |
|---|---|---|---|
| 0–4 | 9,382 | 2 | ~0% |
| 5 | 224 | 2 | 0.9% |
| **6 (all)** | **944** | **706** | **74.8%** |

99.7% of registrations come from the 8.9% who complete all six documents. This is
a **document-completion problem, not a registration problem**, and it drives the
architecture below.

---

## Why the model is built in two stages (not one)

Predicting "will register" *from* document columns is a leaky tautology
(documents ≈ registration). We instead decompose into the two real
uncertainties:

```
P(register) = P(complete all docs | early features)  ×  P(register | complete)
                    └─ Stage A (learned) ─┘              └─ Stage B: 74.8% (observed) ─┘
```

**Stage A** uses only genuinely early features — source, campaign, allocation
type, dialer disposition/sub-bucket, connectivity, talk time, dials — never the
document columns. **Stage B** is the observed completion→registration rate with a
90% Wilson interval: 74.8% [72.4, 77.0].

---

## Modules

| # | Module | Answers | Key output |
|---|---|---|---|
| 0 | Cleaning + feature engineering | — | tidy dataframe, audit log |
| 1 | Descriptive dashboards | What happened? | Exec / RM / Source / Campaign / Funnel |
| 2 | Lead Quality Score (0–100) | How good is a fresh lead? | data-driven feature weights |
| 3 | Two-stage predictive model | Who will register, and why? | P(register), risk bands, model comparison |
| 4 | RM performance (confound-adjusted) | Who really over/under-performs? | efficiency index, difficulty score |
| 5 | Scenario forecast | What happens next month? | regs by source/RM + CIs |
| 6 | Prescriptive engine | What should we do? | ranked interventions + uplift |
| 7 | Target simulator | Can we hit 4,000, and how? | 2 scenarios side by side |
| 8 | Marketing intelligence | Where should budget go? | source/campaign ranking + reallocation |
| 9 | Bottleneck detection | Where is it breaking? | worst-of-every-axis + root cause |

---

## Key formulas

- **Registration probability:** `p_register = p_complete_docs × 0.748`
- **Lead Quality Score:** min-max rescale of the Stage-A logistic linear
  predictor to [0,100]; feature weights = standardized coefficients (so each
  weight is "how much this feature moves completion odds").
- **RM Expected Registrations:** `Σ p_register` over the RM's leads (calibrated,
  so it sums to observed totals).
- **Efficiency Index:** `Actual / Expected` (>1 = beat the hand dealt).
- **RM Difficulty Score:** `100 × (1 − mean_lead_p / overall_mean_p)` (positive =
  harder leads).
- **Forecast CI:** Wilson per source + binomial Monte-Carlo bootstrap for the total.
- **Intervention uplift (documents):** `Δcompletion_pp × total_leads × 0.748`.

Calibration check (in the smoke test): predicted total **707** vs actual **708**.

---

## Model choice & interpretability

Logistic Regression, a depth-4 Decision Tree, and Gradient Boosting are compared
by ROC-AUC, PR-AUC and **Brier score** (calibration matters more than accuracy on
a 9% base rate). All score AUC ≈ 0.96. **Logistic Regression is kept** — it wins
on explainability and nothing beats it by the configured 0.02 AUC margin. We do
**not** use `class_weight="balanced"` for the scoring model because that
decalibrates the expected-value estimates that Modules 4/5 depend on.

---

## Folder structure

```
supply_intelligence/
├── config/config.yaml          # single source of truth (columns, semantics, params)
├── data/raw/                    # drop the monthly xlsx here
│   └── supply_data.xlsx
├── data/processed/
├── outputs/{tables,figures,reports}/
├── src/
│   ├── config.py                # config loader
│   ├── utils.py                 # safe rates, Wilson interval, IO
│   ├── data_loader.py           # M0a cleaning
│   ├── features.py              # M0b feature engineering + model matrix
│   ├── descriptive.py           # M1
│   ├── lead_quality.py          # M2
│   ├── predictive.py            # M3
│   ├── rm_performance.py        # M4
│   ├── forecast.py              # M5
│   ├── prescriptive.py          # M6
│   ├── target_simulator.py      # M7
│   ├── marketing.py             # M8
│   └── bottleneck.py            # M9
├── tests/test_smoke.py
├── run_pipeline.py              # orchestrator
├── requirements.txt
└── README.md
```

---

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py                 # writes outputs/
PYTHONPATH=. python tests/test_smoke.py # sanity checks
```

---

## Dashboard layout (recommended)

1. **Executive strip:** Total Leads · Registered · Reg% · Doc-Completion% ·
   Gap-to-Target — one row of big numbers.
2. **Funnel** (the hero chart): the document cliff should be visually obvious.
3. **Source/Campaign efficiency** bars (registrations per 1,000 leads).
4. **RM quadrant:** Difficulty Score (x) vs Efficiency Index (y) — instantly
   separates "hard leads" from "underperformance."
5. **Target simulator** sliders (completion %, volume, mix) → live projected
   registrations vs the 4,000 line.

---

## Roadmap: adding July & August

1. **Drop-in months:** put each month's file in `data/raw/`, set
   `project.analysis_month`, re-run. No code changes.
2. **Unlock real forecasting:** with ≥3 months, replace the scenario propagation
   in Module 5 with a proper time-series / mixed-effects model that learns
   month-over-month drift and seasonality (impossible with one month today).
3. **Concept-drift monitoring:** track Stage-A AUC/Brier per month; alert if
   source conversion rates shift materially.
4. **RM learning curves:** with panel data, separate an RM's *trend* from a
   single month's noise.
5. **Add cost data (CPL):** turn Module 8's "registrations per 1,000 leads" into
   true cost-per-onboarding and ROI-optimal budget allocation.
6. **Uplift modelling:** once interventions are trialled, move from correlational
   uplift estimates to causal (treatment/control) uplift.

---

## Honest caveats

- **One month of data.** No temporal validation is possible yet; forecasts are
  scenario propagations, not learned time series. Stated in-output.
- **The 4,000 target is a ~5.6× stretch** at current volume (needs 37.9% reg
  rate vs 6.7% today). Module 7 shows both the realistic ceiling and the
  volume/conversion combinations required — it does not pretend the gap is small.
- **Uplift estimates are correlational**, computed from observed June rates.
  Treat as directional sizing, not guarantees, until A/B-tested.
