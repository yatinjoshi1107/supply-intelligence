"""
Supply Intelligence Dashboard
Reads live data from a Google Apps Script web app endpoint, runs the full
9-module predictive pipeline, and displays interactive executive-grade visuals.
Auto-refreshes every 5 minutes + manual refresh button.
"""
import sys
import io
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── make src/ importable ──────────────────────────────────────────────────────
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from src.config import load_config, ensure_dirs
from src import (features, descriptive, lead_quality, predictive,
                 rm_performance, forecast, prescriptive,
                 target_simulator, marketing, bottleneck)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
# Paste your Apps Script web app URL here so ALL users see the same dashboard.
# If left empty, the sidebar will prompt for it.
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzUW2s9LrwPBZ_8KB8rxPyzHQhpkB8FCO-8jMWgzUP39enwIKSqYmbVb0qLXBLZrl8Aeg/exec"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Supply Intelligence", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* card containers */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700;
    color: #1e293b !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] input {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

/* header area */
h1 {
    color: #1e293b !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    padding-bottom: 0 !important;
    margin-bottom: 0.3rem !important;
}
h2, h3 {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* dividers */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* info/warning/success boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 0.92rem;
}

/* expanders */
details {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #fafbfc;
}
details summary {
    font-weight: 600 !important;
}

/* tables */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* tabs */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_gsheet_df(url: str) -> pd.DataFrame:
    url = url.strip()
    if "script.google.com" in url:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise ValueError("Apps Script returned empty data.")
        return df

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Could not parse the URL.")
    sid = match.group(1)
    gid_m = re.search(r"[?&]gid=(\d+)", url)
    gid = gid_m.group(1) if gid_m else "0"

    for tpl in [
        "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&gid={}",
        "https://docs.google.com/spreadsheets/d/{}/export?format=csv&gid={}",
    ]:
        try:
            req = urllib.request.Request(tpl.format(sid, gid),
                                        headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                df = pd.read_csv(io.BytesIO(resp.read()))
            if not df.empty:
                return df
        except Exception:
            continue
    raise ConnectionError("Could not fetch sheet data.")


def _clean(df: pd.DataFrame, cfg: dict, month: str) -> pd.DataFrame:
    drop = [c for c in cfg.get("drop_columns", []) if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    col_map = cfg["columns"].copy()
    for suffix, canon in [("Auto dials", "auto_dials"), ("Manual Dials", "manual_dials")]:
        col_actual = f"{month} {suffix}"
        if col_actual in df.columns:
            col_map[canon] = col_actual
    rev = {v: k for k, v in col_map.items()}
    df = df.rename(columns={a: c for a, c in rev.items() if a in df.columns})

    if "lead_id" in df.columns:
        df = df.drop_duplicates(subset=["lead_id"])

    for c in ["talk_time", "auto_dials", "manual_dials"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["created_date", "rm_assign_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "campaign" in df.columns:
        df["campaign"] = df["campaign"].fillna("(untagged)")
    if "disposition" in df.columns:
        df["disposition"] = df["disposition"].fillna("Never Dialed")
    if "sub_disposition" in df.columns:
        df["sub_disposition"] = df["sub_disposition"].fillna("Never Dialed")

    # RM assign — handle both datetime NaT and string blanks from CSV
    # Use rm_name as primary RM-assignment signal — more reliable than date parsing from CSV.
    # A lead is "unassigned" only if rm_name is genuinely blank.
    if "rm_name" in df.columns:
        blank_rm = df["rm_name"].astype(str).str.strip().isin(["", "nan", "None", "null", "NaN", "na"])
        df["rm_unassigned_flag"] = blank_rm.astype(int)
    elif "rm_assign_date" in df.columns:
        is_blank = df["rm_assign_date"].isna()
        if df["rm_assign_date"].dtype == object:
            is_blank = is_blank | df["rm_assign_date"].astype(str).str.strip().isin(["", "nan", "NaT", "None", "null"])
        df["rm_unassigned_flag"] = is_blank.astype(int)
    else:
        df["rm_unassigned_flag"] = 0  # assume all assigned if column missing

    for c in ["source", "rm_name", "lead_stage", "allocation_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline(url: str, month: str, target: int) -> dict:
    cfg = load_config()
    ensure_dirs(cfg)
    cfg["project"]["analysis_month"] = month
    cfg["business"]["monthly_registration_target"] = target
    cfg["columns"]["auto_dials"] = f"{month} Auto dials"
    cfg["columns"]["manual_dials"] = f"{month} Manual Dials"

    df_raw = load_gsheet_df(url)
    df = _clean(df_raw, cfg, month)
    df = features.engineer(df, cfg)

    desc   = descriptive.run(df, cfg)
    lq     = lead_quality.compute(df, cfg);  df = lq["df"]
    pred   = predictive.run(df, cfg);        df = pred["df"]
    rmp    = rm_performance.run(df, cfg)
    fc     = forecast.forecast(df, None, cfg)
    presc  = prescriptive.run(df, rmp, cfg)
    tgt    = target_simulator.run(df, cfg)
    mkt    = marketing.run(df, cfg)
    bn     = bottleneck.run(df, desc["funnel"], rmp, cfg)

    return dict(df=df, cfg=cfg, desc=desc, lq=lq, pred=pred,
                rmp=rmp, fc=fc, presc=presc, tgt=tgt, mkt=mkt, bn=bn)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 Supply Intelligence")
    st.caption("POSP Onboarding Analytics")
    st.markdown("---")

    if APPS_SCRIPT_URL:
        data_url = APPS_SCRIPT_URL
    else:
        data_url = st.text_input(
            "Apps Script URL",
            placeholder="https://script.google.com/macros/s/.../exec",
        )

    c1, c2 = st.columns(2)
    month = c1.selectbox("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], index=5)
    target = c2.number_input("Target", min_value=100, max_value=100000, value=4000, step=100)

    st.markdown("---")
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Auto-refreshes every 5 min")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Executive Summary", "Funnel Analysis", "Source & Marketing",
         "RM Performance", "Interventions", "Target Simulator",
         "Bottlenecks", "Lead Scores"],
        label_visibility="collapsed",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GATE
# ═══════════════════════════════════════════════════════════════════════════════
if not data_url:
    st.markdown("## 👋 Welcome to Supply Intelligence")
    st.markdown("---")
    st.markdown("### One-time setup (2 minutes)")
    st.markdown("#### Step 1 — Add this script to your Google Sheet")
    st.code('''function doGet(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet()
                .getSheetByName("Base")
             || SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var data = sheet.getDataRange().getValues();
  var csv = data.map(function(row) {
    return row.map(function(cell) {
      var s = String(cell);
      return (s.indexOf(",") > -1 || s.indexOf('"') > -1)
        ? '"' + s.replace(/"/g, '""') + '"' : s;
    }).join(",");
  }).join("\\n");
  return ContentService.createTextOutput(csv)
    .setMimeType(ContentService.MimeType.CSV);
}''', language="javascript")
    st.markdown(
        "1. Open your Google Sheet → **Extensions → Apps Script**\n"
        "2. Delete existing code → paste the above → **Save**\n"
        "3. **Deploy → New deployment** → ⚙️ **Web app**\n"
        "4. Execute as: **Me** · Who has access: **Anyone** → **Deploy**\n"
        "5. Copy the URL → paste in the sidebar"
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD & RUN
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Analyzing..."):
    try:
        data = run_pipeline(data_url, month, target)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

df    = data["df"]
cfg   = data["cfg"]
desc  = data["desc"]
lq    = data["lq"]
pred  = data["pred"]
rmp   = data["rmp"]
fc    = data["fc"]
presc = data["presc"]
tgt   = data["tgt"]
mkt   = data["mkt"]
bn    = data["bn"]

ex = desc["executive"]["Value"]
total_leads = int(ex["Total Leads"])
registered  = int(ex["Registered"])
reg_pct     = float(ex["Registration %"])
gap         = target - registered
doc_pct     = float(ex["Doc Completion %"])
reg_docs    = float(ex["Reg | Docs Complete %"])
conn_pct    = float(ex["Connected %"])

with st.sidebar:
    st.success(f"✅  {total_leads:,} leads loaded")
    st.caption(f"Last refresh: {time.strftime('%H:%M:%S')}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — color for plotly charts
# ═══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe",
           "#00f2fe", "#43e97b", "#fa709a", "#fee140", "#30cfd0"]
SRC_COLORS = {"Organic": "#43e97b", "Google": "#4facfe", "Meta": "#f5576c"}
VERDICT_COLORS = {"Outperformed": "#43e97b", "On expectation": "#fee140", "Underperformed": "#f5576c"}


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.markdown(f"# Executive Summary — {month}")
    st.caption(f"{total_leads:,} leads analyzed · Target {target:,}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Leads", f"{total_leads:,}")
    c2.metric("Registrations", f"{registered:,}", f"{reg_pct:.1f}%")
    c3.metric("Gap to Target", f"{gap:,}")
    c4.metric("Doc Completion", f"{doc_pct:.1f}%")
    c5.metric("Reg | Docs Done", f"{reg_docs:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=registered,
            delta={"reference": target, "valueformat": ","},
            gauge={
                "axis": {"range": [0, target], "tickwidth": 1},
                "bar": {"color": "#667eea", "thickness": 0.7},
                "bgcolor": "#f1f5f9",
                "steps": [
                    {"range": [0, target*0.5], "color": "#fee2e2"},
                    {"range": [target*0.5, target*0.8], "color": "#fef3c7"},
                    {"range": [target*0.8, target], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 3},
                              "thickness": 0.8, "value": target},
            },
            title={"text": "Registrations", "font": {"size": 16}},
            number={"font": {"size": 36, "color": "#1e293b"}},
        ))
        fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        src_df = desc["source"].reset_index()
        fig2 = px.bar(src_df, x="source", y="registration_%", color="source",
                      text_auto=".1f", color_discrete_map=SRC_COLORS)
        fig2.update_layout(showlegend=False, height=280,
                           margin=dict(t=30, b=30), xaxis_title="", yaxis_title="Registration %",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig2.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Key insight card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea18 0%, #764ba218 100%);
                border: 1px solid #c7d2fe; border-radius: 12px; padding: 20px 24px;">
        <h4 style="margin:0 0 8px; color:#4338ca;">Core Insight</h4>
        <p style="margin:0; color:#334155; font-size:1rem; line-height:1.6;">
            <strong>{reg_docs:.0f}%</strong> of leads who upload all 6 documents register
            — but only <strong>{doc_pct:.1f}%</strong> complete their docs.
            This is a <strong>document-completion problem</strong>, not a registration problem.
            Fixing this single step would contribute more than any other intervention combined.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick Numbers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Connected %", f"{conn_pct:.1f}%")
    c2.metric("Avg Talk Time", f"{float(ex['Avg Talk Time (s)']):.0f}s")
    c3.metric("Avg Auto Dials", f"{float(ex['Avg Auto Dials']):.2f}")
    c4.metric("Avg Manual Dials", f"{float(ex['Avg Manual Dials']):.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FUNNEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Funnel Analysis":
    st.markdown("# Funnel Analysis")
    st.caption("Stage-by-stage lead progression and drop-off")

    funnel_df = desc["funnel"]

    fig = go.Figure(go.Funnel(
        y=funnel_df["stage"], x=funnel_df["count"],
        textinfo="value+percent initial",
        marker=dict(color=PALETTE[:len(funnel_df)]),
        connector={"line": {"color": "#e2e8f0", "width": 1}},
    ))
    fig.update_layout(height=420, margin=dict(t=20, b=20, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Leakage Detail")
    disp = funnel_df[["stage","count","pct_of_top","leakage_count","leakage_%_of_step"]].copy()
    disp.columns = ["Stage","Leads","% of Total","Leads Lost","Leakage %"]
    disp["% of Total"] = disp["% of Total"].map("{:.1f}%".format)
    disp["Leakage %"] = disp["Leakage %"].map("{:.1f}%".format)
    st.dataframe(disp.set_index("Stage"), use_container_width=True)

    worst = funnel_df.iloc[1:].sort_values("leakage_count", ascending=False).iloc[0]
    st.warning(f"**Biggest drop:** {worst['stage']} — {int(worst['leakage_count']):,} leads lost ({worst['leakage_%_of_step']:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SOURCE & MARKETING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Source & Marketing":
    st.markdown("# Source & Marketing Intelligence")

    src_df   = desc["source"].reset_index()
    camp_df  = mkt["campaign_ranking"].reset_index()
    realloc  = mkt["reallocation"].reset_index()
    src_rank = mkt["source_ranking"].reset_index()

    t1, t2, t3 = st.tabs(["Registration Rate", "Volume vs Conversion", "Doc Completion"])

    with t1:
        fig = px.bar(src_df, x="source", y="registration_%", color="source",
                     text_auto=".2f", color_discrete_map=SRC_COLORS)
        fig.update_layout(showlegend=False, height=320, xaxis_title="",
                          yaxis_title="Registration %",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig = px.scatter(src_df, x="leads", y="registration_%",
                         size="registrations", color="source", text="source",
                         color_discrete_map=SRC_COLORS,
                         labels={"leads":"Lead Volume", "registration_%":"Registration %"})
        fig.update_traces(textposition="top center")
        fig.update_layout(showlegend=False, height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        fig = px.bar(src_df, x="source", y="doc_completion_%", color="source",
                     text_auto=".1f", color_discrete_map=SRC_COLORS)
        fig.update_layout(showlegend=False, height=320, xaxis_title="",
                          yaxis_title="Doc Completion %",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Campaign Performance")
    if "reg_rate_%" in camp_df.columns and "campaign" in camp_df.columns:
        top_camp = camp_df.head(15).copy()
        fig = px.bar(top_camp, x="reg_rate_%", y="campaign", orientation="h",
                     text_auto=".1f", color="reg_rate_%",
                     color_continuous_scale=["#e2e8f0","#667eea"])
        fig.update_layout(coloraxis_showscale=False, height=450,
                          yaxis={"categoryorder":"total ascending"},
                          xaxis_title="Registration %", yaxis_title="",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Budget Reallocation")
    st.dataframe(realloc, use_container_width=True)
    h = mkt["highlights"]
    st.success(
        f"Reallocation uplift: **+{h.get('reallocation_uplift_registrations', 0):.0f}** registrations  ·  "
        f"Best source: **{h.get('highest_conversion_source', 'N/A')}**  ·  "
        f"Best campaign: **{h.get('most_efficient_campaign', 'N/A')}**"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RM PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "RM Performance":
    st.markdown("# RM Performance")
    st.caption("Efficiency = actual ÷ model-expected registrations (difficulty-adjusted)")

    rm_df = rmp.reset_index()

    col1, col2 = st.columns([2.5, 1])

    with col1:
        fig = px.bar(
            rm_df.sort_values("efficiency_index"),
            x="efficiency_index", y="rm_name", orientation="h",
            color="verdict", color_discrete_map=VERDICT_COLORS,
            text="efficiency_index",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.add_vline(x=1.0, line_dash="dash", line_color="#94a3b8",
                      annotation_text="Baseline (1.0)")
        fig.update_layout(height=max(400, len(rm_df)*32), showlegend=True,
                          xaxis_title="Efficiency Index", yaxis_title="",
                          legend_title="",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        for v in ["Outperformed", "On expectation", "Underperformed"]:
            n = int((rm_df["verdict"] == v).sum())
            icon = {"Outperformed":"🟢","On expectation":"🟡","Underperformed":"🔴"}[v]
            st.metric(f"{icon} {v}", n)

        st.markdown("---")
        st.markdown("##### Bottom 3")
        bot = rm_df.nsmallest(3, "efficiency_index")[["rm_name","efficiency_index","actual_registrations","expected_registrations"]]
        bot.columns = ["RM","Index","Actual","Expected"]
        for _, r in bot.iterrows():
            st.markdown(f"**{r['RM']}** — {r['Index']:.2f}  ({int(r['Actual'])} vs {int(r['Expected'])} exp)")

    st.markdown("---")
    st.subheader("Full Table")
    cols = [c for c in ["rm_name","leads","actual_registrations","expected_registrations",
                        "efficiency_index","difficulty_score","verdict"] if c in rm_df.columns]
    st.dataframe(rm_df[cols].sort_values("efficiency_index", ascending=False).set_index("rm_name"),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INTERVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Interventions":
    st.markdown("# Prescriptive Interventions")
    st.caption("Ranked levers to close the gap — cumulative impact")

    iv = presc["interventions"]

    fig = go.Figure()
    fig.add_bar(x=iv["intervention"], y=iv["expected_uplift"],
                name="Uplift", marker_color="#667eea",
                text=iv["expected_uplift"].map("+{:.0f}".format),
                textposition="outside")
    fig.add_scatter(x=iv["intervention"], y=iv["projected_registrations"],
                    name="Cumulative Total", mode="lines+markers+text",
                    line=dict(color="#f5576c", width=3), marker=dict(size=10),
                    text=iv["projected_registrations"].map("{:,.0f}".format),
                    textposition="top center", yaxis="y2")
    fig.add_hline(y=target, line_dash="dot", line_color="#43e97b",
                  annotation_text=f"Target: {target:,}", yref="y2")
    fig.update_layout(
        yaxis=dict(title="Uplift"), yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420, margin=dict(t=60, b=120), xaxis_tickangle=-18,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    for _, row in iv.iterrows():
        emoji = {"Operations":"⚙️","Marketing":"📣"}.get(str(row.get("lever","")), "📌")
        with st.expander(f"#{int(row['rank'])}  {row['intervention']}  →  +{int(row['expected_uplift'])} reg"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Uplift", f"+{int(row['expected_uplift'])}")
            c2.metric("Cum. Total", f"{int(row['projected_registrations']):,}")
            c3.metric("Lever", f"{emoji} {row.get('lever','')}")
            st.info(str(row.get("rationale","")))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — TARGET SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Target Simulator":
    st.markdown("# Target Simulator")
    st.caption(f"What it takes to hit {target:,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current", f"{registered:,}")
    c2.metric("Target", f"{target:,}")
    c3.metric("Gap", f"{gap:,}")

    st.markdown("---")
    sweep = tgt["sweep"]
    fig = px.bar(sweep, x="lever", y="new_registrations", text_auto=",.0f",
                 color="new_registrations", color_continuous_scale=["#e2e8f0","#667eea"])
    fig.add_hline(y=target, line_dash="dash", line_color="#43e97b",
                  annotation_text=f"Target: {target:,}")
    fig.update_layout(coloraxis_showscale=False, height=360,
                      xaxis_title="", yaxis_title="Projected Registrations",
                      xaxis_tickangle=-15,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sa = tgt["scenario_a"]
    sb = tgt["scenario_b"]
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""<div style="background:#dcfce7; border-radius:12px; padding:20px; border:1px solid #86efac;">
            <h4 style="color:#166534; margin:0 0 8px;">Scenario A — Realistic Max</h4>
            <p style="color:#334155; margin:0; line-height:1.6;">
                Double doc completion → <strong>{:,.0f}</strong> registrations<br/>
                Still <strong>{:,.0f}</strong> short of target at current volume.
            </p></div>""".format(float(sa.get("projected_registrations",0)),
                                 target - float(sa.get("projected_registrations",0))),
                    unsafe_allow_html=True)

    with c2:
        req_comp = float(sb.get("req_completion_rate_if_volume_fixed", 0))
        req_vol  = int(sb.get("req_volume_if_conversion_fixed", 0))
        st.markdown("""<div style="background:#fef3c7; border-radius:12px; padding:20px; border:1px solid #fcd34d;">
            <h4 style="color:#92400e; margin:0 0 8px;">Scenario B — Path to {:,}</h4>
            <p style="color:#334155; margin:0; line-height:1.6;">
                Doc completion → <strong>{:.0f}%</strong> at fixed volume, <em>or</em><br/>
                Lead volume → <strong>{:,}</strong> at current conversion.
            </p></div>""".format(target, req_comp*100, req_vol),
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — BOTTLENECKS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Bottlenecks":
    st.markdown("# Bottleneck Analysis")
    st.caption("Top friction points across funnel, RMs, sources, and campaigns")

    priority = {"Largest funnel leakage", "Worst RM (efficiency-adjusted)", "Worst source"}
    for _, row in bn.iterrows():
        is_crit = row["dimension"] in priority
        color = "#fef2f2" if is_crit else "#fffbeb"
        border = "#fca5a5" if is_crit else "#fcd34d"
        icon = "🔴" if is_crit else "🟡"

        with st.expander(f"{icon}  {row['dimension']}  —  {row['culprit']}"):
            st.markdown(f"""
            <div style="background:{color}; border:1px solid {border}; border-radius:10px;
                        padding:16px; margin-bottom:8px;">
                <strong>Signal:</strong> {row['metric']}<br/>
                <strong>Root cause:</strong> {row['root_cause']}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.dataframe(bn[["dimension","culprit","metric","root_cause"]].set_index("dimension"),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — LEAD SCORES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Lead Scores":
    st.markdown("# Predictive Lead Scores")
    st.caption("P(register) = P(complete docs) × P(register | complete)")

    scores = pred["df"]
    comp   = pred["comparison"]

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Band Distribution")
        if "risk_category" in scores.columns:
            bands = scores["risk_category"].value_counts().reset_index()
            bands.columns = ["Band","Count"]
            fig = px.pie(bands, names="Band", values="Count",
                         color="Band",
                         color_discrete_map={"High Probability":"#43e97b",
                                             "Medium Probability":"#fee140",
                                             "Low Probability":"#f5576c"},
                         hole=0.45)
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Score Distribution")
        if "p_register" in scores.columns:
            fig = px.histogram(scores, x="p_register", nbins=50,
                               color_discrete_sequence=["#667eea"],
                               labels={"p_register":"P(Register)"})
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Comparison")
    comp_r = comp.reset_index()
    mcol = comp_r.columns[0]
    fig = px.bar(comp_r.melt(id_vars=mcol, value_vars=["roc_auc","pr_auc","brier"]),
                 x=mcol, y="value", color="variable", barmode="group",
                 text_auto=".3f",
                 color_discrete_sequence=["#667eea","#4facfe","#f5576c"])
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      legend_title="", xaxis_title="", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Chosen: **{pred['chosen_model']}** — {pred['rationale']}")

    st.markdown("---")
    st.subheader("Top 100 Leads")
    show = [c for c in ["lead_id","p_register","risk_category","source","rm_name"] if c in scores.columns]
    st.dataframe(scores[show].sort_values("p_register", ascending=False).head(100),
                 use_container_width=True)
