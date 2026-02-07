"""Sensor Data QC Tool - Interactive quality control for time series data."""

import streamlit as st
import pandas as pd
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
)

# ── Header ──────────────────────────────────────────────────
st.title("Sensor Data QC Tool")
st.markdown("Interactive quality control for time series sensor data. Upload your CSV or try the sample dataset.")

# ── Sidebar: Parameters ────────────────────────────────────
st.sidebar.header("QC Parameters")

outlier_method = st.sidebar.selectbox("Outlier detection method", ["Z-Score", "IQR", "Both"])
zscore_threshold = st.sidebar.slider("Z-Score threshold", 1.0, 5.0, 3.0, 0.1)
iqr_factor = st.sidebar.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)
drift_window = st.sidebar.slider("Drift detection window", 10, 200, 50, 5)
drift_threshold = st.sidebar.slider("Drift threshold (σ)", 0.5, 5.0, 2.0, 0.1)

st.sidebar.header("Cleaning Parameters")
outlier_action = st.sidebar.selectbox("Outlier handling", ["Replace with rolling mean", "Remove (set NaN)"])
rolling_window = st.sidebar.slider("Rolling mean window", 3, 50, 10, 1)
interpolate_max_gap = st.sidebar.slider("Max gap to interpolate", 1, 20, 5, 1)

# ── Data Input ──────────────────────────────────────────────
st.header("1. Data Input")

col_upload, col_sample = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

with col_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_sensor_data.csv")
    use_sample = st.button("Use sample dataset", type="primary")

df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} rows from uploaded file.")
elif use_sample or "df" not in st.session_state:
    if use_sample and os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.session_state["df"] = df
        st.success(f"Loaded {len(df)} rows from sample dataset.")

if df is None and "df" in st.session_state:
    df = st.session_state["df"]

if df is None:
    st.info("Upload a CSV file or click 'Use sample dataset' to get started.")
    st.stop()

st.session_state["df"] = df

# ── Auto-detect columns ────────────────────────────────────
datetime_cols = []
for col in df.columns:
    try:
        pd.to_datetime(df[col])
        datetime_cols.append(col)
    except (ValueError, TypeError):
        pass

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not datetime_cols:
    st.error("No datetime column detected. Ensure your CSV has a timestamp column.")
    st.stop()

if not numeric_cols:
    st.error("No numeric columns detected.")
    st.stop()

timestamp_col = st.selectbox("Timestamp column", datetime_cols, index=0)
value_cols = st.multiselect("Value columns to analyze", numeric_cols, default=numeric_cols)

if not value_cols:
    st.warning("Select at least one value column.")
    st.stop()

# Parse timestamps
df[timestamp_col] = pd.to_datetime(df[timestamp_col])

# ── Raw Data Overview ───────────────────────────────────────
st.header("2. Raw Data Overview")

col_stats, col_chart = st.columns([1, 2])

with col_stats:
    st.subheader("Statistics")
    stats_data = {}
    for col in value_cols:
        series = df[col]
        nan_count = series.isna().sum()
        nan_pct = nan_count / len(series) * 100
        stats_data[col] = {
            "Min": f"{series.min():.3f}" if pd.notna(series.min()) else "N/A",
            "Max": f"{series.max():.3f}" if pd.notna(series.max()) else "N/A",
            "Mean": f"{series.mean():.3f}" if pd.notna(series.mean()) else "N/A",
            "Std": f"{series.std():.3f}" if pd.notna(series.std()) else "N/A",
            "NaN": f"{nan_count} ({nan_pct:.1f}%)",
        }
    st.dataframe(pd.DataFrame(stats_data).T, use_container_width=True)

with col_chart:
    st.subheader("Time Series")
    fig_raw = make_subplots(rows=len(value_cols), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    colors = ["#0369A1", "#0F172A", "#059669", "#D97706"]
    for i, col in enumerate(value_cols):
        fig_raw.add_trace(
            go.Scatter(
                x=df[timestamp_col], y=df[col],
                mode="lines", name=col,
                line=dict(color=colors[i % len(colors)], width=1),
            ),
            row=i + 1, col=1,
        )
        fig_raw.update_yaxes(title_text=col, row=i + 1, col=1)
    fig_raw.update_layout(height=250 * len(value_cols), margin=dict(t=20, b=20))
    st.plotly_chart(fig_raw, use_container_width=True)

# ── Detection ───────────────────────────────────────────────
st.header("3. Issue Detection")

issues = run_all_detections(
    df, timestamp_col, value_cols,
    zscore_threshold=zscore_threshold,
    iqr_factor=iqr_factor,
    drift_window=drift_window,
    drift_threshold=drift_threshold,
    outlier_method=outlier_method,
)

if len(issues) == 0:
    st.success("No issues detected!")
else:
    col_summary, col_table = st.columns([1, 2])

    with col_summary:
        st.subheader("Summary")
        issue_counts = issues["issue_type"].value_counts()
        for issue_type, count in issue_counts.items():
            st.metric(issue_type, count)
        st.metric("Total issues", len(issues))

    with col_table:
        st.subheader("Issues Table")
        st.dataframe(issues, use_container_width=True, height=300)

    # Chart with issues highlighted
    st.subheader("Data with Issues Highlighted")
    issue_colors = {
        "Missing Value (NaN)": "#EF4444",
        "Timestamp Gap": "#F59E0B",
        "Outlier (Z-Score)": "#8B5CF6",
        "Outlier (IQR)": "#EC4899",
        "Drift": "#F97316",
    }
    fig_issues = make_subplots(rows=len(value_cols), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for i, col in enumerate(value_cols):
        fig_issues.add_trace(
            go.Scatter(
                x=df[timestamp_col], y=df[col],
                mode="lines", name=col,
                line=dict(color=colors[i % len(colors)], width=1),
                showlegend=(i == 0),
            ),
            row=i + 1, col=1,
        )
        # Overlay issues
        col_issues = issues[issues["column"] == col]
        for issue_type, color in issue_colors.items():
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
                    marker=dict(color=color, size=8, symbol="x"),
                    showlegend=(i == 0),
                ),
                row=i + 1, col=1,
            )
        fig_issues.update_yaxes(title_text=col, row=i + 1, col=1)
    fig_issues.update_layout(height=300 * len(value_cols), margin=dict(t=20, b=20))
    st.plotly_chart(fig_issues, use_container_width=True)

# ── Cleaning ────────────────────────────────────────────────
st.header("4. Data Cleaning")

df_clean = clean_series(
    df, value_cols,
    outlier_method=outlier_method,
    outlier_action=outlier_action,
    zscore_threshold=zscore_threshold,
    iqr_factor=iqr_factor,
    rolling_window=rolling_window,
    interpolate_max_gap=interpolate_max_gap,
)

# ── Comparison ──────────────────────────────────────────────
st.header("5. Before vs After")

for col in value_cols:
    fig_compare = go.Figure()
    fig_compare.add_trace(
        go.Scatter(
            x=df[timestamp_col], y=df[col],
            mode="lines", name="Raw",
            line=dict(color="#94A3B8", width=1),
            opacity=0.6,
        )
    )
    fig_compare.add_trace(
        go.Scatter(
            x=df_clean[timestamp_col], y=df_clean[col],
            mode="lines", name="Cleaned",
            line=dict(color="#0369A1", width=1.5),
        )
    )
    fig_compare.update_layout(
        title=col,
        height=300,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# Cleaning stats
st.subheader("Cleaning Summary")
summary_data = {}
for col in value_cols:
    raw_nan = df[col].isna().sum()
    clean_nan = df_clean[col].isna().sum()
    changed = (df[col].fillna(-9999) != df_clean[col].fillna(-9999)).sum()
    summary_data[col] = {
        "NaN before": raw_nan,
        "NaN after": clean_nan,
        "NaN recovered": raw_nan - clean_nan,
        "Points modified": changed,
    }
st.dataframe(pd.DataFrame(summary_data).T, use_container_width=True)

# ── Export ──────────────────────────────────────────────────
st.header("6. Export")

col_csv, col_report = st.columns(2)

with col_csv:
    csv_data = df_clean.to_csv(index=False)
    st.download_button(
        label="Download cleaned CSV",
        data=csv_data,
        file_name="cleaned_sensor_data.csv",
        mime="text/csv",
    )

with col_report:
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
    )

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.markdown(
    "Built by [Gabriel Furiati](https://github.com/Furiatii) | "
    "[Portfolio](https://gabrielfuriati.me)"
)
