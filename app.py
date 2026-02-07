"""Sensor Data QC Tool - Interactive quality control for time series data."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from qc.detectors import run_all_detections
from qc.cleaners import clean_series
from qc.report import generate_report

st.set_page_config(
    page_title="Sensor Data QC Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #0369A1 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .app-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        font-size: 1.05rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .metric-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-card .value.accent { color: #0369A1; }
    .metric-card .value.warning { color: #D97706; }
    .metric-card .value.danger { color: #DC2626; }
    .metric-card .value.success { color: #059669; }

    /* Issue badges */
    .issue-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .issue-badge .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .badge-nan { background: #FEF2F2; color: #DC2626; }
    .badge-nan .dot { background: #DC2626; }
    .badge-outlier { background: #F5F3FF; color: #7C3AED; }
    .badge-outlier .dot { background: #7C3AED; }
    .badge-drift { background: #FFF7ED; color: #EA580C; }
    .badge-drift .dot { background: #EA580C; }
    .badge-gap { background: #FFFBEB; color: #D97706; }
    .badge-gap .dot { background: #D97706; }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .section-header .icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
    }
    .icon-blue { background: #DBEAFE; }
    .icon-amber { background: #FEF3C7; }
    .icon-green { background: #D1FAE5; }
    .icon-purple { background: #EDE9FE; }
    .icon-red { background: #FEE2E2; }

    /* Comparison legend */
    .legend-row {
        display: flex;
        gap: 1.5rem;
        margin: 0.5rem 0 0.5rem 0;
        font-size: 0.9rem;
        color: #64748B;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .legend-line {
        width: 24px;
        height: 3px;
        border-radius: 2px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #E2E8F0;
        color: #94A3B8;
        font-size: 0.85rem;
    }
    .app-footer a {
        color: #0369A1;
        text-decoration: none;
        font-weight: 500;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    .sidebar-section {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Upload area */
    .upload-area {
        display: flex;
        gap: 1rem;
        align-items: stretch;
    }

    /* Hide default streamlit header spacing */
    .block-container { padding-top: 1rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", color="#0F172A"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=30, b=30, l=50, r=20),
    xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="white", bordercolor="#E2E8F0",
        font=dict(size=12, color="#0F172A"),
    ),
)
COLORS = ["#0369A1", "#7C3AED", "#059669", "#D97706", "#DC2626"]
ISSUE_COLORS = {
    "Missing Value (NaN)": "#DC2626",
    "Timestamp Gap": "#D97706",
    "Outlier (Z-Score)": "#7C3AED",
    "Outlier (IQR)": "#EC4899",
    "Drift": "#EA580C",
}

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Sensor Data QC Tool</h1>
    <p>Upload a time series CSV, detect anomalies, clean the data, and export results.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    st.markdown('<div class="sidebar-title">Detection</div>', unsafe_allow_html=True)
    outlier_method = st.selectbox("Outlier method", ["Z-Score", "IQR", "Both"], label_visibility="collapsed")
    zscore_threshold = st.slider("Z-Score threshold", 1.0, 5.0, 3.0, 0.1)
    iqr_factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)
    drift_window = st.slider("Drift window (points)", 10, 200, 50, 5)
    drift_threshold = st.slider("Drift threshold (sigma)", 0.5, 5.0, 2.0, 0.1)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">Cleaning</div>', unsafe_allow_html=True)
    outlier_action = st.selectbox("Outlier handling", ["Replace with rolling mean", "Remove (set NaN)"], label_visibility="collapsed")
    rolling_window = st.slider("Rolling window", 3, 50, 10, 1)
    interpolate_max_gap = st.slider("Max gap to interpolate", 1, 20, 5, 1)

# ── Data Input ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="icon icon-blue">📁</div>
    <h2>Data Input</h2>
</div>
""", unsafe_allow_html=True)

col_upload, col_or, col_sample = st.columns([5, 1, 2])

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

with col_or:
    st.markdown("<div style='text-align:center; padding-top:0.5rem; color:#94A3B8; font-weight:600'>or</div>", unsafe_allow_html=True)

with col_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_sensor_data.csv")
    use_sample = st.button("Load sample data", type="primary", use_container_width=True)

df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.session_state["data_source"] = uploaded_file.name
elif use_sample and os.path.exists(sample_path):
    df = pd.read_csv(sample_path)
    st.session_state["df"] = df
    st.session_state["data_source"] = "sample_sensor_data.csv"

if df is None and "df" in st.session_state:
    df = st.session_state["df"]

if df is None:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#94A3B8">
        <p style="font-size:2.5rem; margin-bottom:0.5rem">📊</p>
        <p style="font-size:1.1rem; font-weight:600; color:#64748B">No data loaded</p>
        <p>Upload a CSV file or click <strong>Load sample data</strong> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Auto-detect columns ────────────────────────────────────
datetime_cols = []
for c in df.columns:
    try:
        pd.to_datetime(df[c])
        datetime_cols.append(c)
    except (ValueError, TypeError):
        pass

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not datetime_cols:
    st.error("No datetime column detected. Ensure your CSV has a timestamp column.")
    st.stop()
if not numeric_cols:
    st.error("No numeric columns detected.")
    st.stop()

col_ts, col_vals = st.columns([1, 2])
with col_ts:
    timestamp_col = st.selectbox("Timestamp column", datetime_cols, index=0)
with col_vals:
    value_cols = st.multiselect("Value columns", numeric_cols, default=numeric_cols)

if not value_cols:
    st.warning("Select at least one value column.")
    st.stop()

df[timestamp_col] = pd.to_datetime(df[timestamp_col])

# ── Dataset info metrics ───────────────────────────────────
total_points = len(df)
time_range = df[timestamp_col].iloc[-1] - df[timestamp_col].iloc[0]
total_nan = sum(df[c].isna().sum() for c in value_cols)
nan_pct = total_nan / (total_points * len(value_cols)) * 100

source_name = st.session_state.get("data_source", "unknown")

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Source</div>
        <div class="value" style="font-size:0.95rem">{source_name}</div>
    </div>
    <div class="metric-card">
        <div class="label">Records</div>
        <div class="value accent">{total_points:,}</div>
    </div>
    <div class="metric-card">
        <div class="label">Time span</div>
        <div class="value" style="font-size:1rem">{time_range.days}d {time_range.seconds//3600}h</div>
    </div>
    <div class="metric-card">
        <div class="label">Channels</div>
        <div class="value accent">{len(value_cols)}</div>
    </div>
    <div class="metric-card">
        <div class="label">Missing values</div>
        <div class="value {"danger" if nan_pct > 5 else "warning" if nan_pct > 1 else "success"}">{total_nan} ({nan_pct:.1f}%)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs for main content ──────────────────────────────────
tab_overview, tab_detect, tab_clean, tab_export = st.tabs([
    "Overview", "Detection", "Before vs After", "Export"
])

# ── TAB 1: Overview ────────────────────────────────────────
with tab_overview:
    st.markdown("""
    <div class="section-header">
        <div class="icon icon-blue">📈</div>
        <h2>Raw Data</h2>
    </div>
    """, unsafe_allow_html=True)

    # Stats table
    stats_data = {}
    for col in value_cols:
        series = df[col]
        nan_count = series.isna().sum()
        nan_pct_col = nan_count / len(series) * 100
        stats_data[col] = {
            "Min": f"{series.min():.3f}" if pd.notna(series.min()) else "N/A",
            "Max": f"{series.max():.3f}" if pd.notna(series.max()) else "N/A",
            "Mean": f"{series.mean():.3f}" if pd.notna(series.mean()) else "N/A",
            "Std": f"{series.std():.3f}" if pd.notna(series.std()) else "N/A",
            "NaN": f"{nan_count} ({nan_pct_col:.1f}%)",
        }
    st.dataframe(pd.DataFrame(stats_data).T, use_container_width=True)

    # Raw time series chart
    fig_raw = make_subplots(
        rows=len(value_cols), cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"  {c}" for c in value_cols],
    )
    for i, col in enumerate(value_cols):
        fig_raw.add_trace(
            go.Scatter(
                x=df[timestamp_col], y=df[col],
                mode="lines", name=col,
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f}<extra></extra>",
            ),
            row=i + 1, col=1,
        )
        fig_raw.update_yaxes(
            title_text=col, row=i + 1, col=1,
            gridcolor="#F1F5F9", linecolor="#E2E8F0",
            title_font=dict(size=11),
        )
        fig_raw.update_xaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", row=i + 1, col=1)
    fig_raw.update_layout(
        height=220 * len(value_cols),
        showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "legend")},
    )
    for ann in fig_raw["layout"]["annotations"]:
        ann["font"] = dict(size=12, color="#64748B")
        ann["xanchor"] = "left"
    st.plotly_chart(fig_raw, use_container_width=True)

# ── Run detection & cleaning (shared across tabs) ──────────
issues = run_all_detections(
    df, timestamp_col, value_cols,
    zscore_threshold=zscore_threshold,
    iqr_factor=iqr_factor,
    drift_window=drift_window,
    drift_threshold=drift_threshold,
    outlier_method=outlier_method,
)

df_clean = clean_series(
    df, value_cols,
    outlier_method=outlier_method,
    outlier_action=outlier_action,
    zscore_threshold=zscore_threshold,
    iqr_factor=iqr_factor,
    rolling_window=rolling_window,
    interpolate_max_gap=interpolate_max_gap,
)

# ── TAB 2: Detection ──────────────────────────────────────
with tab_detect:
    st.markdown("""
    <div class="section-header">
        <div class="icon icon-red">🔍</div>
        <h2>Issue Detection</h2>
    </div>
    """, unsafe_allow_html=True)

    if len(issues) == 0:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#059669">
            <p style="font-size:2rem; margin-bottom:0.3rem">✓</p>
            <p style="font-weight:600; font-size:1.1rem">No issues detected</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Issue summary badges
        issue_counts = issues["issue_type"].value_counts()
        badge_map = {
            "Missing Value (NaN)": "badge-nan",
            "Timestamp Gap": "badge-gap",
            "Outlier (Z-Score)": "badge-outlier",
            "Outlier (IQR)": "badge-outlier",
            "Drift": "badge-drift",
        }
        badges_html = ""
        for issue_type, count in issue_counts.items():
            css = badge_map.get(issue_type, "badge-nan")
            badges_html += f'<span class="issue-badge {css}"><span class="dot"></span>{issue_type}: {count}</span>'

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card" style="flex:0 0 auto">
                <div class="label">Total issues</div>
                <div class="value danger">{len(issues)}</div>
            </div>
            <div class="metric-card" style="flex:3; text-align:left">
                <div class="label">Breakdown</div>
                <div style="margin-top:0.3rem">{badges_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Issues chart
        fig_issues = make_subplots(
            rows=len(value_cols), cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[f"  {c}" for c in value_cols],
        )
        for i, col in enumerate(value_cols):
            fig_issues.add_trace(
                go.Scatter(
                    x=df[timestamp_col], y=df[col],
                    mode="lines", name=col,
                    line=dict(color=COLORS[i % len(COLORS)], width=1.2),
                    opacity=0.5,
                    showlegend=False,
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f}<extra></extra>",
                ),
                row=i + 1, col=1,
            )
            col_issues = issues[issues["column"] == col]
            for issue_type, color in ISSUE_COLORS.items():
                type_issues = col_issues[col_issues["issue_type"] == issue_type]
                if len(type_issues) == 0:
                    continue
                issue_timestamps = pd.to_datetime(type_issues["timestamp"])
                issue_values = []
                for ts in issue_timestamps:
                    match = df[df[timestamp_col] == ts]
                    if len(match) > 0:
                        issue_values.append(match[col].iloc[0])
                    else:
                        issue_values.append(None)
                fig_issues.add_trace(
                    go.Scatter(
                        x=issue_timestamps, y=issue_values,
                        mode="markers", name=issue_type,
                        marker=dict(color=color, size=9, symbol="x", line=dict(width=2, color=color)),
                        showlegend=(i == 0),
                        hovertemplate=f"<b>{issue_type}</b><br>%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.3f}}<extra></extra>",
                    ),
                    row=i + 1, col=1,
                )
            fig_issues.update_yaxes(
                title_text=col, row=i + 1, col=1,
                gridcolor="#F1F5F9", linecolor="#E2E8F0",
                title_font=dict(size=11),
            )
            fig_issues.update_xaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", row=i + 1, col=1)

        fig_issues.update_layout(
            height=250 * len(value_cols),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        )
        for ann in fig_issues["layout"]["annotations"]:
            ann["font"] = dict(size=12, color="#64748B")
            ann["xanchor"] = "left"
        st.plotly_chart(fig_issues, use_container_width=True)

        # Issues table
        with st.expander("View all issues", expanded=False):
            st.dataframe(
                issues.style.apply(
                    lambda row: [
                        f"color: {ISSUE_COLORS.get(row['issue_type'], '#0F172A')}"
                    ] * len(row),
                    axis=1,
                ),
                use_container_width=True,
                height=400,
            )

# ── TAB 3: Before vs After ────────────────────────────────
with tab_clean:
    st.markdown("""
    <div class="section-header">
        <div class="icon icon-green">✨</div>
        <h2>Before vs After</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="legend-row">
        <div class="legend-item">
            <div class="legend-line" style="background:#CBD5E1"></div>
            <span>Raw data</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background:#0369A1"></div>
            <span>Cleaned data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig_compare = make_subplots(
        rows=len(value_cols), cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"  {c}" for c in value_cols],
    )
    for i, col in enumerate(value_cols):
        fig_compare.add_trace(
            go.Scatter(
                x=df[timestamp_col], y=df[col],
                mode="lines", name="Raw" if i == 0 else None,
                line=dict(color="#CBD5E1", width=1.5),
                showlegend=(i == 0),
                hovertemplate="Raw: %{y:.3f}<extra></extra>",
            ),
            row=i + 1, col=1,
        )
        fig_compare.add_trace(
            go.Scatter(
                x=df_clean[timestamp_col], y=df_clean[col],
                mode="lines", name="Cleaned" if i == 0 else None,
                line=dict(color="#0369A1", width=1.8),
                showlegend=(i == 0),
                hovertemplate="Cleaned: %{y:.3f}<extra></extra>",
            ),
            row=i + 1, col=1,
        )
        fig_compare.update_yaxes(
            title_text=col, row=i + 1, col=1,
            gridcolor="#F1F5F9", linecolor="#E2E8F0",
            title_font=dict(size=11),
        )
        fig_compare.update_xaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", row=i + 1, col=1)

    fig_compare.update_layout(
        height=230 * len(value_cols),
        showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "legend")},
    )
    for ann in fig_compare["layout"]["annotations"]:
        ann["font"] = dict(size=12, color="#64748B")
        ann["xanchor"] = "left"
    st.plotly_chart(fig_compare, use_container_width=True)

    # Cleaning summary metrics
    st.markdown("#### Cleaning Summary")
    summary_cols = st.columns(len(value_cols))
    for i, col in enumerate(value_cols):
        raw_nan = df[col].isna().sum()
        clean_nan = df_clean[col].isna().sum()
        recovered = raw_nan - clean_nan
        changed = (df[col].fillna(-9999) != df_clean[col].fillna(-9999)).sum()
        pct_recovered = (recovered / raw_nan * 100) if raw_nan > 0 else 0

        with summary_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{col}</div>
                <div style="margin-top:0.5rem">
                    <div style="font-size:0.8rem; color:#64748B">NaN: {raw_nan} → {clean_nan}</div>
                    <div style="font-size:1.3rem; font-weight:700; color:#059669">{recovered} recovered</div>
                    <div style="font-size:0.8rem; color:#64748B">{changed} points modified</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 4: Export ──────────────────────────────────────────
with tab_export:
    st.markdown("""
    <div class="section-header">
        <div class="icon icon-purple">📦</div>
        <h2>Export Results</h2>
    </div>
    """, unsafe_allow_html=True)

    col_csv, col_report = st.columns(2)

    with col_csv:
        st.markdown("""
        <div class="metric-card" style="margin-bottom:1rem">
            <div class="label">Cleaned Dataset</div>
            <div style="font-size:0.9rem; color:#64748B; margin-top:0.3rem">
                CSV with all corrections applied
            </div>
        </div>
        """, unsafe_allow_html=True)
        csv_data = df_clean.to_csv(index=False)
        st.download_button(
            label="Download cleaned CSV",
            data=csv_data,
            file_name="cleaned_sensor_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_report:
        st.markdown("""
        <div class="metric-card" style="margin-bottom:1rem">
            <div class="label">QC Report</div>
            <div style="font-size:0.9rem; color:#64748B; margin-top:0.3rem">
                Summary of all detections and corrections
            </div>
        </div>
        """, unsafe_allow_html=True)
        params = {
            "Outlier method": outlier_method,
            "Z-Score threshold": zscore_threshold,
            "IQR factor": iqr_factor,
            "Outlier action": outlier_action,
            "Rolling window": rolling_window,
            "Interpolation max gap": interpolate_max_gap,
            "Drift detection window": drift_window,
            "Drift threshold": drift_threshold,
        }
        report_text = generate_report(df, df_clean, issues, timestamp_col, value_cols, params)
        st.download_button(
            label="Download QC report",
            data=report_text,
            file_name="qc_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("Preview report"):
        st.code(report_text, language=None)

# ── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built by <a href="https://github.com/Furiatii">Gabriel Furiati</a> ·
    <a href="https://gabrielfuriati.me">Portfolio</a>
</div>
""", unsafe_allow_html=True)
