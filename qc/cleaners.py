"""Cleaning functions for time series quality control."""

import pandas as pd
import numpy as np


def interpolate_gaps(df: pd.DataFrame, value_cols: list[str], max_gap: int = 5) -> pd.DataFrame:
    """Fill NaN gaps using linear interpolation.

    Only fills gaps up to max_gap consecutive NaN values.
    """
    cleaned = df.copy()
    for col in value_cols:
        cleaned[col] = cleaned[col].interpolate(method="linear", limit=max_gap)
    return cleaned


def remove_outliers_zscore(
    df: pd.DataFrame,
    value_cols: list[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Replace z-score outliers with NaN."""
    cleaned = df.copy()
    for col in value_cols:
        series = cleaned[col].dropna()
        if len(series) < 2:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0:
            continue
        z_scores = (cleaned[col] - mean) / std
        cleaned.loc[z_scores.abs() > threshold, col] = np.nan
    return cleaned


def remove_outliers_iqr(
    df: pd.DataFrame,
    value_cols: list[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Replace IQR outliers with NaN."""
    cleaned = df.copy()
    for col in value_cols:
        series = cleaned[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        cleaned.loc[(cleaned[col] < lower) | (cleaned[col] > upper), col] = np.nan
    return cleaned


def replace_outliers_rolling(
    df: pd.DataFrame,
    value_cols: list[str],
    threshold: float = 3.0,
    window: int = 10,
) -> pd.DataFrame:
    """Replace z-score outliers with rolling mean."""
    cleaned = df.copy()
    for col in value_cols:
        series = cleaned[col].dropna()
        if len(series) < 2:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0:
            continue
        z_scores = (cleaned[col] - mean) / std
        outlier_mask = z_scores.abs() > threshold
        rolling = cleaned[col].rolling(window=window, center=True, min_periods=1).mean()
        cleaned.loc[outlier_mask, col] = rolling[outlier_mask]
    return cleaned


def clean_series(
    df: pd.DataFrame,
    value_cols: list[str],
    outlier_method: str = "Z-Score",
    outlier_action: str = "Replace with rolling mean",
    zscore_threshold: float = 3.0,
    iqr_factor: float = 1.5,
    rolling_window: int = 10,
    interpolate_max_gap: int = 5,
) -> pd.DataFrame:
    """Full cleaning pipeline: remove/replace outliers, then interpolate gaps."""
    cleaned = df.copy()

    # Step 1: Handle outliers
    if outlier_action == "Remove (set NaN)":
        if outlier_method == "Z-Score":
            cleaned = remove_outliers_zscore(cleaned, value_cols, zscore_threshold)
        else:
            cleaned = remove_outliers_iqr(cleaned, value_cols, iqr_factor)
    else:
        cleaned = replace_outliers_rolling(cleaned, value_cols, zscore_threshold, rolling_window)

    # Step 2: Interpolate gaps (including NaNs from outlier removal)
    cleaned = interpolate_gaps(cleaned, value_cols, interpolate_max_gap)

    return cleaned
