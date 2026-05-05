# Deforestation & Land Use Change Monitoring

Time-series satellite imagery (Google Earth Engine / Landsat-8) to track illegal logging and land encroachment, generating automated alerts for forest rangers across Cross River State, Congo Basin, and Niger Delta.

## Features
- Multi-year NDVI time-series analysis (2015–2024)
- Random Forest land-use classification (Forest / Vegetation / Agriculture / Deforested)
- Automated deforestation alert generation (Critical / High / Moderate)
- Forest coverage trend analysis by region
- Interactive change detection map with alert markers

## Project Structure
```
deforestation-monitoring/
├── src/
│   ├── data_loader.py     # Time-series data generation and alert detection
│   ├── model.py           # Land-use classification, forest loss computation
│   └── visualize.py       # Change map, trend charts, land-use pie charts
├── data/raw/              # Landsat/Sentinel rasters, shapefiles
├── models/                # Saved classifier
├── outputs/               # Maps, alert reports, charts
├── config.yaml
├── main.py
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Data Sources
| Layer | Source |
|-------|--------|
| Satellite imagery | Landsat-8 / Sentinel-2 via Google Earth Engine |
| Forest boundaries | Hansen Global Forest Watch |
| Protected areas | WDPA (UNEP-WCMC) |
| Road network | OpenStreetMap |

## Output
- `outputs/deforestation_map.html` — interactive change detection map
- `outputs/deforestation_alerts.csv` — georeferenced deforestation alerts
- `outputs/ndvi_trend.png` — forest coverage trend by region
- `outputs/land_use_change.png` — baseline vs current land-use comparison

## Author
MOMAH MOSES .C.
