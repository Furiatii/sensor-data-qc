"""QC report generation."""

import pandas as pd
from datetime import datetime


def generate_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    issues: pd.DataFrame,
    timestamp_col: str,
    value_cols: list[str],
    params: dict,
) -> str:
    """Generate a text summary of the QC process."""
    lines = []
    lines.append("=" * 60)
    lines.append("SENSOR DATA QUALITY CONTROL REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Dataset info
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total records: {len(df_raw)}")
    lines.append(f"Timestamp column: {timestamp_col}")
    lines.append(f"Value columns: {', '.join(value_cols)}")
    if len(df_raw) > 0:
        lines.append(f"Time range: {df_raw[timestamp_col].iloc[0]} to {df_raw[timestamp_col].iloc[-1]}")
    lines.append("")

    # Raw data statistics
    lines.append("RAW DATA STATISTICS")
    lines.append("-" * 40)
    for col in value_cols:
        series = df_raw[col]
        nan_count = series.isna().sum()
        nan_pct = nan_count / len(series) * 100 if len(series) > 0 else 0
        lines.append(f"  {col}:")
        lines.append(f"    Min: {series.min():.4f}" if pd.notna(series.min()) else f"    Min: N/A")
        lines.append(f"    Max: {series.max():.4f}" if pd.notna(series.max()) else f"    Max: N/A")
        lines.append(f"    Mean: {series.mean():.4f}" if pd.notna(series.mean()) else f"    Mean: N/A")
        lines.append(f"    Std: {series.std():.4f}" if pd.notna(series.std()) else f"    Std: N/A")
        lines.append(f"    NaN: {nan_count} ({nan_pct:.1f}%)")
    lines.append("")

    # Issues detected
    lines.append("ISSUES DETECTED")
    lines.append("-" * 40)
    if len(issues) == 0:
        lines.append("  No issues detected.")
    else:
        issue_counts = issues["issue_type"].value_counts()
        for issue_type, count in issue_counts.items():
            lines.append(f"  {issue_type}: {count}")
        lines.append(f"  Total: {len(issues)}")
    lines.append("")

    # Cleaning results
    lines.append("CLEANING RESULTS")
    lines.append("-" * 40)
    for col in value_cols:
        raw_nan = df_raw[col].isna().sum()
        clean_nan = df_clean[col].isna().sum()
        recovered = raw_nan - clean_nan
        raw_valid = df_raw[col].notna().sum()
        clean_valid = df_clean[col].notna().sum()
        changed = (df_raw[col].fillna(-9999) != df_clean[col].fillna(-9999)).sum()
        lines.append(f"  {col}:")
        lines.append(f"    NaN before: {raw_nan} -> after: {clean_nan} ({recovered} recovered)")
        lines.append(f"    Points modified: {changed}")
        lines.append(f"    Valid data: {raw_valid} -> {clean_valid}")
    lines.append("")

    # Parameters used
    lines.append("PARAMETERS")
    lines.append("-" * 40)
    for key, value in params.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("End of report")
    lines.append("=" * 60)

    return "\n".join(lines)
