"""
dashboard.py  —  LLM Prompt Firewall  /  threat console
Run from project root: streamlit run dashboard.py
"""

import sqlite3, json, math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Prompt Firewall // Threat Console",
    page_icon="⚠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Palette ──────────────────────────────────────────────────────────────────
# Deep charcoal bg  — NOT pure black, NOT GitHub dark
BG        = "#111010"
SURFACE   = "#1a1917"
BORDER    = "#2e2b27"
COPPER    = "#c87941"   # accent — warm amber/copper
COPPER_DIM= "#7a4a28"
TEXT      = "#e8ddd0"
MUTED     = "#7a7168"
RED       = "#c94040"
AMBER     = "#c87941"
GREEN     = "#4e9a6a"

# ─── CSS injection ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Mono', 'Fira Code', monospace !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
section[data-testid="stSidebar"] {{
    background-color: {BG} !important;
}}
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 2px solid {COPPER_DIM};
    border-radius: 2px;
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] > div {{
    color: {MUTED} !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}}
[data-testid="stMetricValue"] > div {{
    color: {TEXT} !important;
    font-size: 1.6rem !important;
    font-weight: 600;
}}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.75rem !important;
}}
[data-testid="stDataFrame"] thead th {{
    background: {SURFACE} !important;
    color: {MUTED} !important;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER} !important;
}}
[data-testid="stDataFrame"] tbody tr:hover td {{
    background: {SURFACE} !important;
}}
div[data-testid="stPlotlyChart"] {{
    border: 1px solid {BORDER};
    border-radius: 2px;
    background: {SURFACE};
}}
.stProgress > div > div > div > div {{
    background-color: {COPPER} !important;
}}
.stProgress > div > div > div {{
    background-color: {BORDER} !important;
}}
h1, h2, h3 {{ color: {TEXT} !important; font-weight: 500; }}
hr {{ border-color: {BORDER} !important; }}
.eyebrow {{
    color: {MUTED};
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.threat-badge-high   {{ color: {RED};   background: #2d1a1a; padding: 1px 7px; border-radius: 2px; font-size: 0.7rem; }}
.threat-badge-medium {{ color: {AMBER}; background: #2d2215; padding: 1px 7px; border-radius: 2px; font-size: 0.7rem; }}
.threat-badge-low    {{ color: {GREEN}; background: #172519; padding: 1px 7px; border-radius: 2px; font-size: 0.7rem; }}
</style>
""", unsafe_allow_html=True)

# ─── Plotly base theme ────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="IBM Plex Mono, monospace", color=MUTED, size=11),
)
AXIS_DEFAULT = dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER, zeroline=False)

# ─── DB load ──────────────────────────────────────────────────────────────────
DB_PATH = Path("logs/firewall.db")

@st.cache_data(ttl=10)
def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM requests ORDER BY id DESC", conn)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["blocked"] = df["blocked"].astype(bool)
    return df

df = load_data()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    f'<p class="eyebrow">llm-prompt-firewall · regex + semantic pipeline</p>',
    unsafe_allow_html=True,
)
st.markdown("### THREAT CONSOLE")

if df.empty:
    st.warning("No data yet — send prompts to /api/v1/analyze first.")
    st.stop()

# ─── Derived stats ────────────────────────────────────────────────────────────
total    = len(df)
blocked  = int(df["blocked"].sum())
allowed  = total - blocked
rate_pct = blocked / total * 100 if total else 0
avg_score= df["injection_score"].mean()
avg_lat  = df["latency_ms"].mean() if "latency_ms" in df.columns else 0
risk_counts = df["risk_level"].value_counts().to_dict()

# ─── Threat meter (signature element) ────────────────────────────────────────
# Segmented bar: green zone (0-30%), amber (30-60%), red (60-100%)
# Copper cursor sits at current block rate
st.markdown('<p class="eyebrow" style="margin-top:1.2rem">threat level indicator</p>', unsafe_allow_html=True)

fig_meter = go.Figure()
segment_colors = [GREEN, GREEN, AMBER, AMBER, AMBER, RED, RED, RED, RED, RED]
for i, c in enumerate(segment_colors):
    fig_meter.add_shape(
        type="rect",
        x0=i*10, x1=i*10+9.2, y0=0, y1=1,
        fillcolor=c, opacity=0.18, line_width=0,
    )
# Cursor line
fig_meter.add_shape(
    type="line",
    x0=rate_pct, x1=rate_pct, y0=-0.2, y1=1.2,
    line=dict(color=COPPER, width=2),
)
fig_meter.add_annotation(
    x=min(rate_pct + 1.5, 96), y=1.35,
    text=f"{rate_pct:.1f}% blocked",
    showarrow=False,
    font=dict(color=COPPER, size=12, family="IBM Plex Mono"),
)
fig_meter.update_layout(
    **LAYOUT_BASE,
    height=60,
    margin=dict(l=0, r=0, t=24, b=0),
    xaxis=dict(range=[0,100], showgrid=False, showticklabels=False, zeroline=False, linecolor="rgba(0,0,0,0)"),
    yaxis=dict(range=[-0.5,2],  showgrid=False, showticklabels=False, zeroline=False, linecolor="rgba(0,0,0,0)"),
    showlegend=False,
)

st.plotly_chart(fig_meter, use_container_width=True, config={"displayModeBar": False})

# ─── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Requests",    f"{total:,}")
c2.metric("Blocked",           f"{blocked:,}")
c3.metric("Allowed",           f"{allowed:,}")
c4.metric("Avg Score",         f"{avg_score:.3f}")
c5.metric("Avg Latency",       f"{avg_lat:.0f} ms")

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ─── Charts row ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

# Score distribution — stepped bar histogram
with col_left:
    st.markdown('<p class="eyebrow">injection score distribution</p>', unsafe_allow_html=True)
    bins  = [round(i*0.1, 1) for i in range(11)]
    labels= [f"{b:.1f}–{b+0.1:.1f}" for b in bins[:-1]]
    counts= pd.cut(df["injection_score"].dropna(), bins=bins, labels=labels, right=False)\
              .value_counts().sort_index()
    bar_colors = []
    for lbl in counts.index:
        lo = float(lbl.split("–")[0])
        bar_colors.append(RED if lo >= 0.6 else (AMBER if lo >= 0.3 else GREEN))
    fig_hist = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values,
        marker_color=bar_colors,
        marker_line_width=0,
    ))
    fig_hist.update_layout(
        **LAYOUT_BASE,
        height=240, bargap=0.06,
        margin=dict(l=40, r=16, t=24, b=40),
        xaxis=AXIS_DEFAULT,
        yaxis=AXIS_DEFAULT,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

# Risk breakdown — horizontal bar
with col_right:
    st.markdown('<p class="eyebrow">requests by risk level</p>', unsafe_allow_html=True)
    levels = ["high", "medium", "low"]
    vals   = [risk_counts.get(l, 0) for l in levels]
    colors = [RED, AMBER, GREEN]
    fig_risk = go.Figure(go.Bar(
        x=vals, y=levels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=vals,
        textposition="outside",
        textfont=dict(color=MUTED, size=11),
    ))
    fig_risk.update_layout(
        **LAYOUT_BASE,
        height=240,
        margin=dict(l=40, r=16, t=24, b=40),
        xaxis=AXIS_DEFAULT,
        yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=BORDER, tickcolor=BORDER, zeroline=False),
    )
    st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

# ─── Volume over time ─────────────────────────────────────────────────────────
if df["timestamp"].notna().any():
    st.markdown('<p class="eyebrow" style="margin-top:1rem">request volume · 1-minute buckets</p>', unsafe_allow_html=True)
    time_df = (
        df.set_index("timestamp")
        .resample("1min")["blocked"]
        .agg(total="count", blocked="sum")
        .fillna(0)
        .reset_index()
    )
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=time_df["timestamp"], y=time_df["total"],
        name="total", mode="lines",
        line=dict(color=MUTED, width=1.5),
        fill="tozeroy", fillcolor="rgba(122,113,104,0.06)",
    ))
    fig_vol.add_trace(go.Scatter(
        x=time_df["timestamp"], y=time_df["blocked"],
        name="blocked", mode="lines",
        line=dict(color=RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(201,64,64,0.08)",
    ))
    fig_vol.update_layout(
        **LAYOUT_BASE,
        height=180,
        margin=dict(l=40, r=16, t=32, b=40),
        xaxis=AXIS_DEFAULT,
        yaxis=AXIS_DEFAULT,
        legend=dict(orientation="h", x=0, y=1.15, font=dict(size=10, color=MUTED)),
    )
    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

# ─── Recent requests table ────────────────────────────────────────────────────
st.markdown('<p class="eyebrow" style="margin-top:1.4rem">recent requests · last 20</p>', unsafe_allow_html=True)

display_cols = ["timestamp","injection_score","regex_score","semantic_score",
                "risk_level","blocked","closest_phrase","latency_ms","prompt_length"]
available = [c for c in display_cols if c in df.columns]
recent = df.head(20)[available].copy()

if "timestamp" in recent.columns:
    recent["timestamp"] = recent["timestamp"].dt.strftime("%H:%M:%S")
for col in ["injection_score","regex_score","semantic_score"]:
    if col in recent.columns:
        recent[col] = recent[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
if "latency_ms" in recent.columns:
    recent["latency_ms"] = recent["latency_ms"].map(lambda x: f"{x:.0f}ms" if pd.notna(x) else "—")
if "blocked" in recent.columns:
    recent["blocked"] = recent["blocked"].map(lambda x: "■ BLOCKED" if x else "· allowed")

st.dataframe(recent, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr/>
<p style='color:{MUTED};font-size:0.65rem;letter-spacing:0.08em;'>
  refreshed {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · 
  github.com/CoderunED/llm-prompt-firewall · 
  day 13/21
</p>
""", unsafe_allow_html=True)
