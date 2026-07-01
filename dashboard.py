"""
Supply Intelligence Dashboard  v2.0
Reads live data from a Google Apps Script web app endpoint, runs the full
9-module predictive pipeline, and displays interactive executive-grade visuals.
Auto-refreshes every 5 minutes + manual refresh button.
"""
import sys
import io
import re
import urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

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
APPS_SCRIPT_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbzUW2s9LrwPBZ_8KB8rxPyzHQhpkB8FCO-8jMWgzUP39enwIKSqYmbVb0qLXBLZrl8Aeg/exec"
)

# ── Design tokens ─────────────────────────────────────────────────────────────
PRIMARY   = "#6366f1"
SUCCESS   = "#10b981"
WARNING   = "#f59e0b"
DANGER    = "#ef4444"
SLATE_900 = "#0f172a"
SLATE_800 = "#1e293b"
SLATE_700 = "#334155"
SLATE_500 = "#64748b"
SLATE_200 = "#e2e8f0"
SLATE_50  = "#f8fafc"

PALETTE = [PRIMARY, SUCCESS, WARNING, DANGER,
           "#8b5cf6", "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#14b8a6"]

SRC_COLORS = {"Organic": SUCCESS, "Google": PRIMARY, "Meta": DANGER}
VERDICT_COLORS = {
    "Outperformed":    SUCCESS,
    "On expectation":  WARNING,
    "Underperformed":  DANGER,
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
              color=SLATE_700),
    margin=dict(t=40, b=40, l=10, r=10),
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ← must come before any other st call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Supply Intelligence | InsuranceDekho",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 5 minutes (cache TTL matches)
st_autorefresh(interval=300_000, limit=None, key="auto_refresh")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* ── Metric cards ── */
div[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid {SLATE_200};
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(99,102,241,0.04);
    transition: box-shadow 0.2s;
}}
div[data-testid="stMetric"]:hover {{
    box-shadow: 0 4px 20px rgba(99,102,241,0.12);
}}
div[data-testid="stMetric"] label {{
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: {SLATE_500} !important;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: {SLATE_800} !important;
    letter-spacing: -0.5px;
}}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SLATE_800} 0%, {SLATE_900} 100%) !important;
    border-right: 1px solid #1e293b;
}}
section[data-testid="stSidebar"] * {{
    color: #cbd5e1 !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{
    color: #f1f5f9 !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 6px 0 !important;
}}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    color: #e2e8f0 !important;
}}
section[data-testid="stSidebar"] input {{
    background: #334155 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: #334155 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] .stButton button {{
    background: {PRIMARY} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
section[data-testid="stSidebar"] .stButton button:hover {{
    background: #4f46e5 !important;
}}

/* ── Page headers ── */
h1 {{
    color: {SLATE_800} !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px !important;
    font-size: 1.85rem !important;
    margin-bottom: 0.2rem !important;
}}
h2 {{
    color: {SLATE_800} !important;
    font-weight: 700 !important;
    letter-spacing: -0.4px !important;
}}
h3, h4 {{
    color: {SLATE_700} !important;
    font-weight: 600 !important;
}}

/* ── Dividers ── */
hr {{
    border: none;
    border-top: 1px solid {SLATE_200};
    margin: 1.8rem 0;
}}

/* ── Alerts ── */
div[data-testid="stAlert"] {{
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    border-left-width: 4px !important;
}}

/* ── Expanders ── */
details {{
    border: 1px solid {SLATE_200} !important;
    border-radius: 10px !important;
    background: #fafbfc;
    overflow: hidden;
}}
details summary {{
    font-weight: 600 !important;
    padding: 12px 16px !important;
}}
details[open] summary {{
    border-bottom: 1px solid {SLATE_200};
}}

/* ── DataFrames ── */
.stDataFrame {{
    border-radius: 10px !important;
    border: 1px solid {SLATE_200} !important;
    overflow: hidden !important;
}}

/* ── Tabs ── */
button[data-baseweb="tab"] {{
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PRIMARY} !important;
}}

/* ── Caption text ── */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {SLATE_500} !important;
    font-size: 0.82rem !important;
}}

/* ── Section label ── */
.section-label {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {SLATE_500};
    margin-bottom: 12px;
    margin-top: 4px;
}}

/* ── KPI card (custom) ── */
.kpi-card {{
    background: #ffffff;
    border: 1px solid {SLATE_200};
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
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
            req = urllib.request.Request(
                tpl.format(sid, gid), headers={"User-Agent": "Mozilla/5.0"}
            )
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

    # ── CRITICAL: set rm_unassigned_flag BEFORE pd.to_datetime ─────────────────
    # Google Sheets sends JS date strings like
    # "Thu Jun 18 2026 18:46:00 GMT+0530 (India Standard Time)" which
    # pd.to_datetime coerces to NaT, making every row look unassigned if we
    # check after conversion.
    if "rm_assign_date" in df.columns:
        raw = df["rm_assign_date"].astype(str).str.strip()
        is_blank = df["rm_assign_date"].isna() | raw.isin(
            ["", "nan", "NaT", "None", "null", "na", "0"]
        )
        df["rm_unassigned_flag"] = is_blank.astype(int)
    else:
        df["rm_unassigned_flag"] = 0

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

    for c in ["source", "rm_name", "lead_stage", "allocation_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline(url: str, month: str, target: int, version: int = 4) -> dict:
    """version param forces cache invalidation when bumped."""
    cfg = load_config()
    ensure_dirs(cfg)
    cfg["project"]["analysis_month"] = month
    cfg["business"]["monthly_registration_target"] = target
    cfg["columns"]["auto_dials"] = f"{month} Auto dials"
    cfg["columns"]["manual_dials"] = f"{month} Manual Dials"

    df_raw = load_gsheet_df(url)
    df = _clean(df_raw, cfg, month)
    df = features.engineer(df, cfg)

    desc  = descriptive.run(df, cfg)
    lq    = lead_quality.compute(df, cfg);  df = lq["df"]
    pred  = predictive.run(df, cfg);        df = pred["df"]
    rmp   = rm_performance.run(df, cfg)
    fc    = forecast.forecast(df, None, cfg)
    presc = prescriptive.run(df, rmp, cfg)
    tgt   = target_simulator.run(df, cfg)
    mkt   = marketing.run(df, cfg)
    bn    = bottleneck.run(df, desc["funnel"], rmp, cfg)

    return dict(df=df, cfg=cfg, desc=desc, lq=lq, pred=pred,
                rmp=rmp, fc=fc, presc=presc, tgt=tgt, mkt=mkt, bn=bn)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 4px;">
      <div style="font-size:1.15rem; font-weight:800; color:#f1f5f9; letter-spacing:-0.3px;">
        📊 Supply Intelligence
      </div>
      <div style="font-size:0.72rem; color:#94a3b8; font-weight:500; margin-top:2px;">
        POSP Onboarding Analytics · InsuranceDekho
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    data_url = APPS_SCRIPT_URL if APPS_SCRIPT_URL else st.text_input(
        "Apps Script URL",
        placeholder="https://script.google.com/macros/s/.../exec",
    )

    c1, c2 = st.columns(2)
    month = c1.selectbox(
        "Month",
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        index=5,
    )
    target = c2.number_input("Target", min_value=100, max_value=100_000,
                              value=4000, step=100)

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
# GATE — no URL provided
# ═══════════════════════════════════════════════════════════════════════════════
if not data_url:
    st.markdown("## 👋 Welcome to Supply Intelligence")
    st.markdown("---")
    st.markdown("### One-time setup (2 minutes)")
    st.markdown("#### Step 1 — Add this script to your Google Sheet")
    st.code(
        '''function doGet(e) {
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
}''',
        language="javascript",
    )
    st.markdown(
        "1. Open your Google Sheet → **Extensions → Apps Script**\n"
        "2. Delete existing code → paste the above → **Save**\n"
        "3. **Deploy → New deployment** → ⚙️ **Web app**\n"
        "4. Execute as: **Me** · Who has access: **Anyone** → **Deploy**\n"
        "5. Copy the URL → paste in the sidebar"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD & RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Crunching the numbers…"):
    try:
        data = run_pipeline(data_url, month, target, 4)
    except Exception as e:
        st.error(f"**Pipeline error:** {e}")
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
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    st.success(f"✅  {total_leads:,} leads loaded")
    st.caption(f"Last refresh: {ist.strftime('%d %b %H:%M')} IST")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def _chart(**kwargs) -> dict:
    """Merge CHART_LAYOUT with per-chart overrides."""
    base = {**CHART_LAYOUT}
    base.update(kwargs)
    return base


def _funnel_chart(funnel_df: pd.DataFrame) -> go.Figure:
    """
    Beautiful custom horizontal-bar funnel.
    Each stage = full-width bar, width ∝ count.
    Color interpolates from PRIMARY (#6366f1) → SUCCESS (#10b981).
    Drop-off % shown between bars with ▼ indicator.
    Stages with >50 % drop highlighted in red.
    """
    stages = list(funnel_df["stage"])
    counts = list(funnel_df["count"])
    n = len(stages)

    # Interpolate colors: indigo → emerald
    def _lerp_hex(c1, c2, t):
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    bar_colors = [_lerp_hex(PRIMARY, SUCCESS, i / max(n - 1, 1)) for i in range(n)]
    max_count  = max(counts) if counts else 1

    fig = go.Figure()

    for i, (stage, count, color) in enumerate(zip(stages, counts, bar_colors)):
        pct_of_top = count / max_count * 100

        # Bar
        fig.add_trace(go.Bar(
            x=[count],
            y=[stage],
            orientation="h",
            marker=dict(
                color=color,
                line=dict(width=0),
            ),
            text=[f"  {count:,}   ({count/max_count*100:.1f}%)"],
            textposition="inside" if pct_of_top > 20 else "outside",
            textfont=dict(
                color="#ffffff" if pct_of_top > 20 else SLATE_700,
                size=13,
                family="Inter, sans-serif",
            ),
            hovertemplate=(
                f"<b>{stage}</b><br>"
                f"Leads: {count:,}<br>"
                f"% of Total: {count/max_count*100:.1f}%<extra></extra>"
            ),
            showlegend=False,
            name=stage,
        ))

    # Drop-off annotations between bars
    for i in range(1, n):
        prev = counts[i - 1]
        curr = counts[i]
        if prev > 0:
            drop_pct = (prev - curr) / prev * 100
            is_critical = drop_pct > 50
            ann_color = DANGER if is_critical else SLATE_500
            arrow_color = DANGER if is_critical else WARNING
            fig.add_annotation(
                x=max_count * 0.5,
                y=i - 0.5,          # midpoint between bars
                text=(
                    f"<b style='color:{ann_color}'>▼ {drop_pct:.1f}% drop-off</b>"
                    + ("  ⚠️" if is_critical else "")
                ),
                showarrow=False,
                font=dict(size=11, color=ann_color),
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=4,
                xanchor="center",
                yanchor="middle",
            )

    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13, color=SLATE_700, family="Inter, sans-serif"),
            showgrid=False,
            categoryorder="array",
            categoryarray=list(reversed(stages)),
        ),
        xaxis=dict(
            range=[0, max_count * 1.22],
            showgrid=True,
            gridcolor=SLATE_200,
            zeroline=False,
            tickformat=",",
        ),
        bargap=0.45,
        height=max(380, n * 64),
        **_chart(margin=dict(t=20, b=20, l=10, r=80)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.markdown(f"# Executive Summary")
    st.caption(f"{month} · {total_leads:,} leads analyzed · Target {target:,} registrations")

    # ── Top KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Leads", f"{total_leads:,}")
    c2.metric(
        "Registrations",
        f"{registered:,}",
        f"{reg_pct:.1f}% conversion",
        delta_color="normal",
    )
    c3.metric(
        "Gap to Target",
        f"{gap:,}",
        f"{gap/target*100:.1f}% remaining",
        delta_color="inverse",
    )
    c4.metric("Doc Completion", f"{doc_pct:.1f}%")
    c5.metric("Reg | Docs Done", f"{reg_docs:.1f}%")

    st.markdown("---")

    # ── Gauge + Source bar ─────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        gauge_pct = min(registered / target, 1.0)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=registered,
            delta={
                "reference": target,
                "valueformat": ",",
                "increasing": {"color": SUCCESS},
                "decreasing": {"color": DANGER},
            },
            gauge={
                "axis": {
                    "range": [0, target],
                    "tickwidth": 1,
                    "tickcolor": SLATE_200,
                    "tickfont": {"size": 10, "color": SLATE_500},
                },
                "bar": {"color": PRIMARY, "thickness": 0.68},
                "bgcolor": SLATE_50,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, target * 0.5],   "color": "#fef2f2"},
                    {"range": [target * 0.5, target * 0.8], "color": "#fefce8"},
                    {"range": [target * 0.8, target], "color": "#f0fdf4"},
                ],
                "threshold": {
                    "line": {"color": DANGER, "width": 3},
                    "thickness": 0.8,
                    "value": target,
                },
            },
            title={"text": "Registrations vs Target", "font": {"size": 14, "color": SLATE_500}},
            number={"font": {"size": 40, "color": SLATE_800}, "valueformat": ","},
        ))
        fig_gauge.update_layout(
            height=290,
            **_chart(margin=dict(t=60, b=10, l=30, r=30)),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        src_df = desc["source"].reset_index()
        fig_src = px.bar(
            src_df, x="source", y="registration_%",
            color="source", text_auto=".2f",
            color_discrete_map=SRC_COLORS,
            labels={"registration_%": "Registration %", "source": ""},
        )
        fig_src.update_traces(
            textposition="outside",
            marker_line_width=0,
            textfont=dict(size=12, color=SLATE_700),
        )
        fig_src.update_layout(
            showlegend=False,
            height=290,
            xaxis_title="",
            yaxis_title="Registration %",
            title=dict(text="Registration Rate by Source", font=dict(size=14, color=SLATE_500), x=0),
            **_chart(),
        )
        st.plotly_chart(fig_src, use_container_width=True)

    st.markdown("---")

    # ── Core Insight card ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eef2ff 0%, #f0fdf4 100%);
                border: 1px solid #c7d2fe; border-radius: 14px; padding: 22px 28px;
                box-shadow: 0 1px 4px rgba(99,102,241,0.08);">
      <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                  letter-spacing:1px; color:{PRIMARY}; margin-bottom:10px;">
        Core Insight
      </div>
      <p style="margin:0; color:{SLATE_700}; font-size:1rem; line-height:1.75;">
        <strong style="color:{SLATE_800};">{reg_docs:.0f}%</strong> of leads who complete
        all 6 documents go on to register — but only
        <strong style="color:{DANGER};">{doc_pct:.1f}%</strong> of leads ever finish their docs.
        This is a <strong style="color:{SLATE_800};">document-completion problem</strong>,
        not a registration problem. Closing this single gap would unlock more registrations
        than any other lever combined.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Secondary KPIs ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Operational Metrics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Connected %", f"{conn_pct:.1f}%")
    c2.metric("Avg Talk Time", f"{float(ex['Avg Talk Time (s)']):.0f}s")
    c3.metric("Avg Auto Dials", f"{float(ex['Avg Auto Dials']):.2f}")
    c4.metric("Avg Manual Dials", f"{float(ex['Avg Manual Dials']):.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FUNNEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Funnel Analysis":
    st.markdown("# Funnel Analysis")
    st.caption("Stage-by-stage lead progression — bar width shows relative volume, drop-off % between stages")

    funnel_df = desc["funnel"]

    # ── Custom horizontal bar funnel ───────────────────────────────────────────
    fig = _funnel_chart(funnel_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Leakage detail table ───────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Leakage Detail")
        disp = funnel_df[["stage", "count", "pct_of_top",
                           "leakage_count", "leakage_%_of_step"]].copy()
        disp.columns = ["Stage", "Leads", "% of Total", "Leads Lost", "Leakage %"]
        disp["% of Total"] = disp["% of Total"].map("{:.1f}%".format)
        disp["Leakage %"]  = disp["Leakage %"].map("{:.1f}%".format)
        st.dataframe(disp.set_index("Stage"), use_container_width=True)

    with col2:
        st.subheader("Stage Summary")
        top_count = int(funnel_df["count"].iloc[0])
        bot_count = int(funnel_df["count"].iloc[-1])
        overall_drop = (top_count - bot_count) / top_count * 100

        worst = (
            funnel_df.iloc[1:]
            .sort_values("leakage_count", ascending=False)
            .iloc[0]
        )
        worst2 = (
            funnel_df.iloc[1:]
            .sort_values("leakage_%_of_step", ascending=False)
            .iloc[0]
        )

        st.metric("Overall Funnel Drop", f"{overall_drop:.1f}%")
        st.markdown("")
        st.markdown(f"""
        <div style="background:#fef2f2; border:1px solid #fca5a5; border-radius:10px;
                    padding:14px 16px; margin-bottom:10px;">
          <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.8px; color:{DANGER}; margin-bottom:6px;">
            Biggest Volume Loss
          </div>
          <div style="font-weight:700; color:{SLATE_800}; font-size:0.95rem;">
            {worst['stage']}
          </div>
          <div style="color:{SLATE_500}; font-size:0.85rem; margin-top:4px;">
            {int(worst['leakage_count']):,} leads lost
          </div>
        </div>
        <div style="background:#fffbeb; border:1px solid #fcd34d; border-radius:10px;
                    padding:14px 16px;">
          <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.8px; color:{WARNING}; margin-bottom:6px;">
            Highest % Drop
          </div>
          <div style="font-weight:700; color:{SLATE_800}; font-size:0.95rem;">
            {worst2['stage']}
          </div>
          <div style="color:{SLATE_500}; font-size:0.85rem; margin-top:4px;">
            {worst2['leakage_%_of_step']:.1f}% step drop
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Mini stage bar ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Stage Drop-off Waterfall")
    st.caption("How many leads leak at each transition")

    transitions = []
    for i in range(1, len(funnel_df)):
        prev = funnel_df.iloc[i - 1]
        curr = funnel_df.iloc[i]
        leak = int(prev["count"] - curr["count"])
        pct  = leak / prev["count"] * 100 if prev["count"] > 0 else 0
        transitions.append({
            "Transition": f"{prev['stage']} → {curr['stage']}",
            "Leads Lost": leak,
            "Drop %": pct,
        })

    trans_df = pd.DataFrame(transitions)
    fig_wf = px.bar(
        trans_df, x="Transition", y="Leads Lost",
        text="Leads Lost",
        color="Drop %",
        color_continuous_scale=[[0, "#dcfce7"], [0.5, "#fef3c7"], [1.0, "#fee2e2"]],
        labels={"Leads Lost": "Leads Lost at Transition"},
    )
    fig_wf.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker_line_width=0,
    )
    fig_wf.update_coloraxes(colorbar_title="Drop %")
    fig_wf.update_layout(
        xaxis_tickangle=-20,
        height=360,
        xaxis_title="",
        **_chart(),
    )
    st.plotly_chart(fig_wf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SOURCE & MARKETING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Source & Marketing":
    st.markdown("# Source & Marketing Intelligence")
    st.caption("Channel-level conversion, doc completion, and budget reallocation insights")

    src_df   = desc["source"].reset_index()
    camp_df  = mkt["campaign_ranking"].reset_index()
    realloc  = mkt["reallocation"].reset_index()

    # ── Source tabs ────────────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs([
        "📈  Registration Rate",
        "🔵  Volume vs Conversion",
        "📄  Doc Completion",
    ])

    with t1:
        fig = px.bar(
            src_df, x="source", y="registration_%",
            color="source", text_auto=".2f",
            color_discrete_map=SRC_COLORS,
            labels={"registration_%": "Registration %", "source": ""},
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            showlegend=False, height=340,
            xaxis_title="", yaxis_title="Registration %",
            **_chart(),
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig = px.scatter(
            src_df, x="leads", y="registration_%",
            size="registrations", color="source", text="source",
            color_discrete_map=SRC_COLORS,
            labels={"leads": "Lead Volume", "registration_%": "Registration %"},
        )
        fig.update_traces(textposition="top center", marker_line_width=0)
        fig.update_layout(
            showlegend=False, height=340,
            **_chart(),
        )
        # Add quadrant lines
        if len(src_df) > 0:
            med_leads = float(src_df["leads"].median())
            med_reg   = float(src_df["registration_%"].median())
            fig.add_hline(y=med_reg, line_dash="dot", line_color=SLATE_200,
                          annotation_text="Median reg%", annotation_position="bottom right")
            fig.add_vline(x=med_leads, line_dash="dot", line_color=SLATE_200,
                          annotation_text="Median volume", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        fig = px.bar(
            src_df, x="source", y="doc_completion_%",
            color="source", text_auto=".1f",
            color_discrete_map=SRC_COLORS,
            labels={"doc_completion_%": "Doc Completion %", "source": ""},
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            showlegend=False, height=340,
            xaxis_title="", yaxis_title="Doc Completion %",
            **_chart(),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Campaign performance ────────────────────────────────────────────────────
    st.subheader("Campaign Performance")
    st.caption("Top 15 campaigns by registration rate")

    if "reg_rate_%" in camp_df.columns and "campaign" in camp_df.columns:
        top_camp = camp_df.head(15).copy()
        fig = px.bar(
            top_camp, x="reg_rate_%", y="campaign", orientation="h",
            text_auto=".1f",
            color="reg_rate_%",
            color_continuous_scale=[[0, "#e0e7ff"], [1, PRIMARY]],
            labels={"reg_rate_%": "Registration %", "campaign": ""},
        )
        fig.update_traces(
            textposition="outside",
            marker_line_width=0,
        )
        fig.update_layout(
            coloraxis_showscale=False,
            height=460,
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Registration %",
            yaxis_title="",
            **_chart(),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Budget reallocation ─────────────────────────────────────────────────────
    st.subheader("Recommended Budget Reallocation")
    st.caption("Based on cost-per-registration efficiency")
    st.dataframe(realloc, use_container_width=True)

    h = mkt["highlights"]
    cols = st.columns(3)
    cols[0].metric(
        "Reallocation Uplift",
        f"+{h.get('reallocation_uplift_registrations', 0):.0f} reg",
    )
    cols[1].metric(
        "Best Source",
        h.get("highest_conversion_source", "N/A"),
    )
    cols[2].metric(
        "Best Campaign",
        h.get("most_efficient_campaign", "N/A"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RM PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "RM Performance":
    st.markdown("# RM Performance")
    st.caption(
        "Efficiency Index = actual ÷ model-expected registrations, "
        "adjusted for lead quality and difficulty. Index > 1.0 = outperforming."
    )

    rm_df = rmp.reset_index()

    # ── Summary metrics ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    n_out  = int((rm_df["verdict"] == "Outperformed").sum())
    n_on   = int((rm_df["verdict"] == "On expectation").sum())
    n_under = int((rm_df["verdict"] == "Underperformed").sum())
    c1.metric("🟢 Outperformed",   n_out)
    c2.metric("🟡 On Expectation", n_on)
    c3.metric("🔴 Underperformed", n_under)

    st.markdown("---")

    col_chart, col_detail = st.columns([2.5, 1])

    with col_chart:
        fig = px.bar(
            rm_df.sort_values("efficiency_index"),
            x="efficiency_index",
            y="rm_name",
            orientation="h",
            color="verdict",
            color_discrete_map=VERDICT_COLORS,
            text="efficiency_index",
            labels={"efficiency_index": "Efficiency Index", "rm_name": ""},
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            marker_line_width=0,
        )
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color=SLATE_500,
            line_width=1.5,
            annotation_text="Baseline 1.0",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color=SLATE_500,
        )
        fig.update_layout(
            height=max(420, len(rm_df) * 34),
            showlegend=True,
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0,
            ),
            xaxis_title="Efficiency Index",
            yaxis_title="",
            **_chart(),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_detail:
        st.subheader("Bottom 5 RMs")
        bot = rm_df.nsmallest(5, "efficiency_index")[
            ["rm_name", "efficiency_index", "actual_registrations", "expected_registrations"]
        ]
        for _, r in bot.iterrows():
            gap_v = int(r["actual_registrations"]) - int(r["expected_registrations"])
            st.markdown(f"""
            <div style="border:1px solid #fca5a5; border-radius:8px; padding:10px 12px;
                        margin-bottom:8px; background:#fef2f2;">
              <div style="font-weight:700; font-size:0.88rem; color:{SLATE_800};">
                {r['rm_name']}
              </div>
              <div style="color:{DANGER}; font-size:0.82rem; margin-top:2px;">
                Index: {r['efficiency_index']:.2f} &nbsp;·&nbsp;
                {int(r['actual_registrations'])} vs {int(r['expected_registrations'])} exp
                &nbsp;({gap_v:+d})
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Top 5 RMs")
        top5 = rm_df.nlargest(5, "efficiency_index")[
            ["rm_name", "efficiency_index", "actual_registrations"]
        ]
        for _, r in top5.iterrows():
            st.markdown(f"""
            <div style="border:1px solid #6ee7b7; border-radius:8px; padding:10px 12px;
                        margin-bottom:8px; background:#f0fdf4;">
              <div style="font-weight:700; font-size:0.88rem; color:{SLATE_800};">
                {r['rm_name']}
              </div>
              <div style="color:{SUCCESS}; font-size:0.82rem; margin-top:2px;">
                Index: {r['efficiency_index']:.2f} &nbsp;·&nbsp;
                {int(r['actual_registrations'])} registrations
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Full RM Table")
    cols = [c for c in [
        "rm_name", "leads", "actual_registrations", "expected_registrations",
        "efficiency_index", "difficulty_score", "verdict",
    ] if c in rm_df.columns]
    st.dataframe(
        rm_df[cols].sort_values("efficiency_index", ascending=False).set_index("rm_name"),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INTERVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Interventions":
    st.markdown("# Prescriptive Interventions")
    st.caption("Ranked levers to close the registration gap — sorted by expected uplift, showing cumulative impact")

    iv = presc["interventions"]
    baseline = int(presc.get("baseline_registrations", registered))

    # ── KPI summary ────────────────────────────────────────────────────────────
    total_uplift = int(iv["expected_uplift"].sum()) if "expected_uplift" in iv.columns else 0
    best_proj    = int(iv["projected_registrations"].iloc[-1]) if "projected_registrations" in iv.columns else registered + total_uplift
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline",         f"{baseline:,}")
    c2.metric("Max Uplift",       f"+{total_uplift:,}")
    c3.metric("Best-Case Total",  f"{best_proj:,}")
    c4.metric("Still Short",      f"{max(0, target - best_proj):,}")

    st.markdown("---")

    # ── Combined bar + line chart ──────────────────────────────────────────────
    fig = go.Figure()

    # Uplift bars
    fig.add_trace(go.Bar(
        x=iv["intervention"],
        y=iv["expected_uplift"],
        name="Expected Uplift",
        marker=dict(
            color=[PRIMARY] * len(iv),
            line=dict(width=0),
        ),
        text=iv["expected_uplift"].map("+{:.0f}".format),
        textposition="outside",
        textfont=dict(size=11, color=SLATE_700),
        hovertemplate="<b>%{x}</b><br>Uplift: +%{y:.0f} registrations<extra></extra>",
        yaxis="y",
    ))

    # Cumulative line
    fig.add_trace(go.Scatter(
        x=iv["intervention"],
        y=iv["projected_registrations"],
        name="Cumulative Total",
        mode="lines+markers+text",
        line=dict(color=SUCCESS, width=3),
        marker=dict(size=9, color=SUCCESS, line=dict(color="#fff", width=2)),
        text=iv["projected_registrations"].map("{:,.0f}".format),
        textposition="top center",
        textfont=dict(size=11, color=SUCCESS),
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:,.0f}<extra></extra>",
        yaxis="y2",
    ))

    # Target line
    fig.add_hline(
        y=target,
        line_dash="dot",
        line_color=DANGER,
        line_width=2,
        annotation_text=f"Target: {target:,}",
        annotation_position="top right",
        annotation_font_color=DANGER,
        annotation_font_size=11,
        yref="y2",
    )

    fig.update_layout(
        yaxis=dict(title="Uplift (registrations)", showgrid=True, gridcolor=SLATE_200),
        yaxis2=dict(
            title="Cumulative Registrations",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=440,
        xaxis_tickangle=-18,
        **_chart(margin=dict(t=60, b=120, l=60, r=80)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Intervention cards ─────────────────────────────────────────────────────
    st.subheader("Intervention Detail")
    lever_icons = {"Operations": "⚙️", "Marketing": "📣", "Training": "🎓"}

    for _, row in iv.iterrows():
        rank = int(row.get("rank", 0))
        lever = str(row.get("lever", ""))
        icon  = lever_icons.get(lever, "📌")
        uplift = int(row.get("expected_uplift", 0))
        cum    = int(row.get("projected_registrations", 0))

        with st.expander(
            f"#{rank}  {row['intervention']}   →   +{uplift:,} registrations"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Uplift",  f"+{uplift:,}")
            c2.metric("Cumulative Total", f"{cum:,}")
            c3.metric("Lever",            f"{icon} {lever}")
            if row.get("rationale"):
                st.info(str(row["rationale"]))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — TARGET SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Target Simulator":
    st.markdown("# Target Simulator")
    st.caption(f"What it takes to hit {target:,} registrations this month")

    # ── KPIs ───────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current",  f"{registered:,}")
    c2.metric("Target",   f"{target:,}")
    c3.metric("Gap",      f"{gap:,}", delta_color="inverse")
    c4.metric("% Done",   f"{registered/target*100:.1f}%")

    st.markdown("---")

    # ── Lever sweep ────────────────────────────────────────────────────────────
    sweep = tgt["sweep"]
    st.subheader("Lever Sweep")
    st.caption("Projected registrations under each single-lever scenario")

    fig = px.bar(
        sweep, x="lever", y="new_registrations",
        text="new_registrations",
        color="new_registrations",
        color_continuous_scale=[[0, "#e0e7ff"], [1, PRIMARY]],
        labels={"new_registrations": "Projected Registrations", "lever": ""},
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker_line_width=0,
    )
    fig.add_hline(
        y=target,
        line_dash="dash",
        line_color=SUCCESS,
        line_width=2,
        annotation_text=f"Target: {target:,}",
        annotation_font_color=SUCCESS,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        height=380,
        xaxis_tickangle=-18,
        **_chart(),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Scenarios ──────────────────────────────────────────────────────────────
    sa = tgt["scenario_a"]
    sb = tgt["scenario_b"]

    col1, col2 = st.columns(2)

    with col1:
        proj_a = float(sa.get("projected_registrations", 0))
        still_short_a = max(0, target - proj_a)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7);
                    border: 1px solid #86efac; border-radius: 14px; padding: 24px 28px;
                    height: 100%; box-sizing: border-box;">
          <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.8px; color:{SUCCESS}; margin-bottom:10px;">
            Scenario A — Realistic Maximum
          </div>
          <div style="font-size:1.4rem; font-weight:800; color:{SLATE_800}; margin-bottom:8px;">
            {proj_a:,.0f} registrations
          </div>
          <div style="color:{SLATE_700}; font-size:0.9rem; line-height:1.6;">
            By doubling document completion at current lead volume and RM efficiency.
            {'<span style="color:' + SUCCESS + ';font-weight:700;">✓ Hits target!</span>'
             if still_short_a == 0
             else f'Still <strong>{still_short_a:,.0f}</strong> short of target.'}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        req_comp = float(sb.get("req_completion_rate_if_volume_fixed", 0))
        req_vol  = int(sb.get("req_volume_if_conversion_fixed", 0))
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fffbeb, #fef3c7);
                    border: 1px solid #fcd34d; border-radius: 14px; padding: 24px 28px;
                    height: 100%; box-sizing: border-box;">
          <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.8px; color:{WARNING}; margin-bottom:10px;">
            Scenario B — Path to {target:,}
          </div>
          <div style="font-size:1.4rem; font-weight:800; color:{SLATE_800}; margin-bottom:8px;">
            Two routes
          </div>
          <div style="color:{SLATE_700}; font-size:0.9rem; line-height:1.7;">
            📄 Doc completion →
            <strong style="color:{SLATE_800};">{req_comp*100:.0f}%</strong>
            at current lead volume<br/>
            <em>or</em><br/>
            📥 Lead volume →
            <strong style="color:{SLATE_800};">{req_vol:,}</strong>
            at current conversion rate
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — BOTTLENECKS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Bottlenecks":
    st.markdown("# Bottleneck Analysis")
    st.caption(
        "Top friction points across funnel stages, RM performance, sources, and campaigns — "
        "prioritized by revenue impact"
    )

    CRITICAL_DIMS = {"Largest funnel leakage", "Worst RM (efficiency-adjusted)", "Worst source"}

    # ── Summary row ────────────────────────────────────────────────────────────
    n_crit   = sum(1 for _, r in bn.iterrows() if r["dimension"] in CRITICAL_DIMS)
    n_warn   = len(bn) - n_crit
    total_bn = len(bn)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Bottlenecks",  total_bn)
    c2.metric("🔴 Critical",         n_crit)
    c3.metric("🟡 Monitor",          n_warn)

    st.markdown("---")

    # ── Bottleneck cards ───────────────────────────────────────────────────────
    for _, row in bn.iterrows():
        is_crit  = row["dimension"] in CRITICAL_DIMS
        bg_color = "#fef2f2" if is_crit else "#fffbeb"
        bd_color = "#fca5a5" if is_crit else "#fcd34d"
        icon     = "🔴" if is_crit else "🟡"
        label    = "CRITICAL" if is_crit else "MONITOR"
        lbl_col  = DANGER if is_crit else WARNING

        with st.expander(f"{icon}  {row['dimension']}  —  {row['culprit']}"):
            st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {bd_color};
                        border-radius:10px; padding:16px 20px;">
              <div style="font-size:0.65rem; font-weight:700; text-transform:uppercase;
                          letter-spacing:0.8px; color:{lbl_col}; margin-bottom:10px;">
                {label}
              </div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div>
                  <div style="font-size:0.72rem; font-weight:600; color:{SLATE_500};
                              text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
                    Signal
                  </div>
                  <div style="color:{SLATE_800}; font-size:0.9rem;">{row['metric']}</div>
                </div>
                <div>
                  <div style="font-size:0.72rem; font-weight:600; color:{SLATE_500};
                              text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
                    Root Cause
                  </div>
                  <div style="color:{SLATE_800}; font-size:0.9rem;">{row['root_cause']}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Full Bottleneck Table")
    st.dataframe(
        bn[["dimension", "culprit", "metric", "root_cause"]].set_index("dimension"),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — LEAD SCORES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Lead Scores":
    st.markdown("# Predictive Lead Scores")
    st.caption(
        "P(register) = P(complete docs) × P(register | docs complete). "
        "Scores are difficulty-adjusted — same score on harder leads = better RM."
    )

    scores = pred["df"]
    comp   = pred["comparison"]

    # ── Model headline ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eef2ff, #e0e7ff);
                border: 1px solid #c7d2fe; border-radius: 12px; padding: 16px 22px;
                margin-bottom: 1rem;">
      <span style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                   letter-spacing:0.8px; color:{PRIMARY};">Selected Model</span>
      <div style="font-weight:700; font-size:1rem; color:{SLATE_800}; margin-top:4px;">
        {pred.get('chosen_model', 'N/A')}
      </div>
      <div style="color:{SLATE_500}; font-size:0.85rem; margin-top:4px;">
        {pred.get('rationale', '')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score distribution + risk bands ────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Band Distribution")
        if "risk_category" in scores.columns:
            bands = scores["risk_category"].value_counts().reset_index()
            bands.columns = ["Band", "Count"]
            fig = px.pie(
                bands, names="Band", values="Count",
                color="Band",
                color_discrete_map={
                    "High Probability":   SUCCESS,
                    "Medium Probability": WARNING,
                    "Low Probability":    DANGER,
                },
                hole=0.5,
            )
            fig.update_traces(
                textfont_size=13,
                marker_line_color="#fff",
                marker_line_width=2,
            )
            fig.update_layout(
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                **_chart(margin=dict(t=20, b=40, l=10, r=10)),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Score Distribution")
        if "p_register" in scores.columns:
            fig = px.histogram(
                scores, x="p_register", nbins=50,
                color_discrete_sequence=[PRIMARY],
                labels={"p_register": "P(Register)"},
                opacity=0.85,
            )
            fig.update_traces(marker_line_width=0)
            fig.add_vline(
                x=float(scores["p_register"].median()),
                line_dash="dash",
                line_color=SUCCESS,
                line_width=1.5,
                annotation_text=f"Median {scores['p_register'].median():.2f}",
                annotation_font_color=SUCCESS,
            )
            fig.update_layout(
                height=320,
                xaxis_title="P(Register)",
                yaxis_title="Lead Count",
                **_chart(),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Model comparison ───────────────────────────────────────────────────────
    st.subheader("Model Comparison")
    comp_r  = comp.reset_index()
    mcol    = comp_r.columns[0]
    metrics = [c for c in ["roc_auc", "pr_auc", "brier"] if c in comp_r.columns]

    if metrics:
        melted = comp_r.melt(id_vars=mcol, value_vars=metrics)
        fig = px.bar(
            melted, x=mcol, y="value", color="variable",
            barmode="group", text_auto=".3f",
            color_discrete_sequence=[PRIMARY, SUCCESS, DANGER],
            labels={"value": "Score", "variable": "Metric", mcol: "Model"},
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            height=340,
            legend_title="",
            xaxis_title="",
            yaxis_title="Score",
            **_chart(),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Top 100 actionable leads ───────────────────────────────────────────────
    st.subheader("Top 100 High-Priority Leads")
    st.caption("Highest P(register) — prioritize for immediate RM outreach")

    show = [c for c in [
        "lead_id", "p_register", "risk_category", "source", "rm_name",
    ] if c in scores.columns]

    top_leads = scores[show].sort_values("p_register", ascending=False).head(100)

    # Style the dataframe if p_register exists
    if "p_register" in top_leads.columns:
        top_leads["p_register"] = top_leads["p_register"].round(3)

    st.dataframe(top_leads, use_container_width=True)

    # ── Risk summary ───────────────────────────────────────────────────────────
    if "risk_summary" in pred and pred["risk_summary"] is not None:
        st.markdown("---")
        st.subheader("Risk Distribution Summary")
        rs = pred["risk_summary"]
        if isinstance(rs, pd.DataFrame):
            st.dataframe(rs, use_container_width=True)
        elif isinstance(rs, dict):
            st.json(rs)
