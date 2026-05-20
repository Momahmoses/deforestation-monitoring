# Deforestation & Land Use Change Monitoring

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Time-series satellite imagery analysis to track illegal logging, land encroachment, and deforestation across Cross River State, Congo Basin, and Niger Delta — generating automated alerts for forest rangers and policy makers.

---

## Problem Statement

Nigeria loses ~350,000 hectares of forest annually — one of the world's worst deforestation rates. The Niger Delta mangroves and Cross River rainforests are critically threatened by illegal logging and agricultural encroachment. This system provides near-real-time detection and alert generation.

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-Year NDVI Analysis | Time-series 2015–2024 from Landsat-8 / Sentinel-2 |
| Land-Use Classification | Random Forest — Forest / Vegetation / Agriculture / Deforested |
| Automated Alert Generation | Critical / High / Moderate severity alerts for rangers |
| Forest Coverage Trends | Annual loss rates by region and forest type |
| Change Detection Map | Interactive map with deforestation alert markers |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Remote Sensing | Landsat-8, Sentinel-2 NDVI/EVI |
| Machine Learning | scikit-learn (Random Forest) |
| Geospatial | GeoPandas, Folium, Rasterio |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Data | pandas, NumPy |

---

## Project Structure

```
deforestation-monitoring/
├── src/
│   ├── data_loader.py     # Time-series satellite data ingestion and alert detection
│   ├── model.py           # Land-use classification, forest loss computation
│   └── visualize.py       # Change map, trend charts, land-use pie charts
├── data/raw/              # Landsat/Sentinel rasters, shapefiles
├── models/                # Saved Random Forest classifier
├── config.yaml            # NDVI thresholds, alert levels, region config
├── main.py                # Pipeline entry point
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/deforestation-monitoring.git
cd deforestation-monitoring
pip install -r requirements.txt
python main.py
```

---

## Data Sources

- Global Forest Watch (Hansen tree cover loss rasters)
- Sentinel-2 Level-2A surface reflectance
- Landsat-8 Collection 2 imagery
- NESREA National Environmental Standards Agency shapefiles
- REDD+ Nigeria forest reference level data

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
