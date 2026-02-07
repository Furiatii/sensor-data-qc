# Sensor Data QC Tool

Interactive quality control tool for time series sensor data. Upload your CSV, detect anomalies, clean the data, and export results.

**[Live Demo](https://sensor-data-qc.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## What it does

Sensors drift, spike, and drop out. This tool helps you find and fix those problems in any time series CSV:

1. **Upload** your sensor CSV (or use the included sample dataset)
2. **Detect** gaps, outliers (Z-Score / IQR), and drift automatically
3. **Visualize** raw data with flagged issues highlighted on interactive charts
4. **Clean** the data with configurable interpolation and outlier replacement
5. **Compare** before/after side by side
6. **Export** the cleaned CSV and a QC report

All parameters (thresholds, window sizes, methods) are adjustable in the sidebar.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Click "Use sample dataset" to try it immediately.

## Sample dataset

`data/sample_sensor_data.csv` contains 1000 synthetic readings (15-min intervals) with three sensor channels:

| Column | Description |
|--------|-------------|
| `timestamp` | Datetime at 15-min intervals |
| `water_level_m` | Water level (m) with tidal pattern |
| `flow_rate_m3s` | Flow rate (m3/s) correlated with level |
| `temperature_c` | Temperature (C) with diurnal cycle |

Intentional anomalies injected: NaN gaps, spikes, a consecutive missing block, and gradual drift.

## Detection methods

- **Gaps**: missing timestamps and NaN values
- **Outliers (Z-Score)**: points where |z| exceeds the threshold (default 3.0)
- **Outliers (IQR)**: points outside [Q1 - k*IQR, Q3 + k*IQR] (default k=1.5)
- **Drift**: rolling mean deviating from global mean beyond a configurable number of standard deviations

## Cleaning methods

- **Outlier replacement**: rolling mean substitution or removal (set to NaN)
- **Gap interpolation**: linear interpolation for gaps up to a configurable max length

## Project structure

```
sensor-data-qc/
├── app.py                      # Streamlit app
├── qc/
│   ├── detectors.py            # Gap, outlier, and drift detection
│   ├── cleaners.py             # Interpolation and outlier replacement
│   └── report.py               # QC report generation
├── data/
│   └── sample_sensor_data.csv  # Synthetic test data
├── .streamlit/
│   └── config.toml             # Theme configuration
├── requirements.txt
└── README.md
```

## Tech stack

- **Streamlit** for the interactive UI
- **pandas** for data manipulation
- **NumPy** for numerical operations
- **Plotly** for interactive charts

## Use cases

- Environmental monitoring (hydrological stations, weather sensors)
- Industrial sensor validation (process control, equipment monitoring)
- Research data preprocessing (field instruments, lab equipment)

## License

MIT
