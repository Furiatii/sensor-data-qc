"""Detection functions for time series quality control."""

import pandas as pd
import numpy as np


def detect_gaps(df: pd.DataFrame, timestamp_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Detect missing timestamps and NaN values.

    Returns a DataFrame with columns: timestamp, column, issue_type, details.
    """
    issues = []

    # Detect irregular timestamp intervals
    timestamps = pd.to_datetime(df[timestamp_col])
    diffs = timestamps.diff()
    if len(diffs.dropna()) > 0:
        median_interval = diffs.median()
        gap_mask = diffs > median_interval * 2
        for idx in df.index[gap_mask]:
            issues.append({
                "timestamp": df[timestamp_col].iloc[idx],
                "column": timestamp_col,
                "issue_type": "Timestamp Gap",
                "details": f"Gap of {diffs.iloc[idx]} (expected ~{median_interval})",
            })

    # Detect NaN values
    for col in value_cols:
        nan_mask = df[col].isna()
        for idx in df.index[nan_mask]:
            issues.append({
                "timestamp": df[timestamp_col].iloc[idx],
                "column": col,
                "issue_type": "Missing Value (NaN)",
                "details": "Value is NaN",
            })

    return pd.DataFrame(issues)


def detect_outliers_zscore(
    df: pd.DataFrame,
    timestamp_col: str,
    value_cols: list[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Detect outliers using z-score method.

    Points with |z-score| > threshold are flagged.
    """
    issues = []
    for col in value_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0:
            continue
        z_scores = (df[col] - mean) / std
        outlier_mask = z_scores.abs() > threshold
        for idx in df.index[outlier_mask]:
            if pd.notna(df[col].iloc[idx]):
                issues.append({
                    "timestamp": df[timestamp_col].iloc[idx],
                    "column": col,
                    "issue_type": "Outlier (Z-Score)",
                    "details": f"Value={df[col].iloc[idx]:.4f}, Z={z_scores.iloc[idx]:.2f} (threshold={threshold})",
                })
    return pd.DataFrame(issues)


def detect_outliers_iqr(
    df: pd.DataFrame,
    timestamp_col: str,
    value_cols: list[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers using IQR method.

    Points outside [Q1 - factor*IQR, Q3 + factor*IQR] are flagged.
    """
    issues = []
    for col in value_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        for idx in df.index[outlier_mask]:
            if pd.notna(df[col].iloc[idx]):
                issues.append({
                    "timestamp": df[timestamp_col].iloc[idx],
                    "column": col,
                    "issue_type": "Outlier (IQR)",
                    "details": f"Value={df[col].iloc[idx]:.4f}, bounds=[{lower:.4f}, {upper:.4f}]",
                })
    return pd.DataFrame(issues)


def detect_drift(
    df: pd.DataFrame,
    timestamp_col: str,
    value_cols: list[str],
    window: int = 50,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Detect drift by comparing local mean to global mean.

    Uses a rolling window. Flags regions where the rolling mean deviates
    from the global mean by more than threshold * global_std.
    """
    issues = []
    for col in value_cols:
        series = df[col].dropna()
        if len(series) < window:
            continue
        global_mean = series.mean()
        global_std = series.std()
        if global_std == 0:
            continue
        rolling_mean = df[col].rolling(window=window, center=True, min_periods=window // 2).mean()
        drift_score = (rolling_mean - global_mean).abs() / global_std
        drift_mask = drift_score > threshold
        # Group consecutive drift points into regions
        if drift_mask.any():
            regions = []
            in_region = False
            start_idx = None
            for idx in range(len(drift_mask)):
                if drift_mask.iloc[idx] and not in_region:
                    in_region = True
                    start_idx = idx
                elif not drift_mask.iloc[idx] and in_region:
                    in_region = False
                    regions.append((start_idx, idx - 1))
            if in_region:
                regions.append((start_idx, len(drift_mask) - 1))

            for start, end in regions:
                mid = (start + end) // 2
                issues.append({
                    "timestamp": df[timestamp_col].iloc[mid],
                    "column": col,
                    "issue_type": "Drift",
                    "details": f"Drift detected from row {start} to {end} ({end - start + 1} points), max deviation={drift_score.iloc[start:end+1].max():.2f}σ",
                })
    return pd.DataFrame(issues)


def run_all_detections(
    df: pd.DataFrame,
    timestamp_col: str,
    value_cols: list[str],
    zscore_threshold: float = 3.0,
    iqr_factor: float = 1.5,
    drift_window: int = 50,
    drift_threshold: float = 2.0,
    outlier_method: str = "Z-Score",
) -> pd.DataFrame:
    """Run all detection methods and return combined issues."""
    results = [detect_gaps(df, timestamp_col, value_cols)]

    if outlier_method == "Z-Score":
        results.append(detect_outliers_zscore(df, timestamp_col, value_cols, zscore_threshold))
    elif outlier_method == "IQR":
        results.append(detect_outliers_iqr(df, timestamp_col, value_cols, iqr_factor))
    else:
        results.append(detect_outliers_zscore(df, timestamp_col, value_cols, zscore_threshold))
        results.append(detect_outliers_iqr(df, timestamp_col, value_cols, iqr_factor))

    results.append(detect_drift(df, timestamp_col, value_cols, drift_window, drift_threshold))

    combined = pd.concat(results, ignore_index=True)
    if len(combined) > 0:
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined
