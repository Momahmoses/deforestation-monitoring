# Deforestation & Land Use Change Monitoring

A GIS + ML system that uses time-series satellite imagery (NDVI/EVI) to detect illegal logging and land encroachment in Nigerian forest zones and the Congo Basin — generating real-time alerts for forest rangers and conservation agencies.

## Overview

Processes 25 years of NDVI time series (2000–2025) to:
- Detect deforestation events: gradual degradation vs. sudden clearcuts
- Track annual forest cover change per region
- Generate alert maps for immediate ranger response
- Classify pixels as Stable / Degradation / Clearcut

## Features

- **Dual Detection**: Rule-based threshold + ML Random Forest classifier
- **25-Year Trend Analysis**: NDVI time series per forest region
- **Alert Generation**: Interactive Folium map of active deforestation zones
- **Regional Reporting**: Clearcut vs degradation breakdown per region

## Deforestation Classes

| Class | Description | Trigger |
|-------|-------------|---------|
| Stable | No significant change | NDVI Δ > -0.15 |
| Degradation | Gradual forest loss | NDVI Δ ∈ [-0.25, -0.15] |
| Clearcut | Sudden/large clearing | NDVI Δ < -0.25 |

## Project Structure

```
deforestation-monitoring/
├── src/
│   ├── data_ingestion.py     # NDVI time series generation
│   ├── change_detection.py   # Rule-based + ML detection
│   └── visualization.py      # Alert maps and trend charts
├── data/sample/
├── outputs/
├── config.py
├── main.py
└── requirements.txt
```

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```

## Target Regions

| Region | Country | Forest Cover (2000) |
|--------|---------|-------------------|
| Cross River | Nigeria | 72% |
| Ondo | Nigeria | 55% |
| Edo | Nigeria | 61% |
| Congo Basin | DRC/Congo | 88% |
| Cameroon Highlands | Cameroon | 79% |

## Data Sources (Production)

- NDVI/EVI: Landsat 8-9, Sentinel-2 (Google Earth Engine)
- Hansen Global Forest Change v1.10
- FIRMS active fire data (NASA VIIRS/MODIS)

## Author

**MOMAH MOSES .C.**  
Data Scientist & ML Engineer | [GitHub](https://github.com/Momahmoses)
