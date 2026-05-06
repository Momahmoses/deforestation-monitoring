import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

STUDY_REGIONS = {
    "Cross_River": {"lat": 5.8702, "lon": 8.5988, "forest_cover_pct_2000": 72},
    "Ondo": {"lat": 7.2508, "lon": 5.2103, "forest_cover_pct_2000": 55},
    "Edo": {"lat": 6.3350, "lon": 5.6037, "forest_cover_pct_2000": 61},
    "Congo_Basin": {"lat": -1.5, "lon": 23.5, "forest_cover_pct_2000": 88},
    "Cameroon_Highlands": {"lat": 5.5, "lon": 10.5, "forest_cover_pct_2000": 79},
}

YEARS = list(range(2000, 2026))

# Change detection thresholds
NDVI_CHANGE_THRESHOLD = -0.15  # significant drop = deforestation
ALERT_CHANGE_THRESHOLD = -0.25  # severe drop = active clearing

FEATURE_COLS = [
    "ndvi_t0", "ndvi_t1", "ndvi_change",
    "evi_t0", "evi_t1", "evi_change",
    "slope_deg", "elevation_m", "distance_to_road_km",
    "distance_to_settlement_km", "rainfall_anomaly",
]
TARGET_COL = "deforestation_class"  # 0=stable, 1=degradation, 2=clearcut
