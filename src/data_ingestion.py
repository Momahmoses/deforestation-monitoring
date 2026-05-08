"""Generate synthetic time-series NDVI/EVI data for deforestation monitoring."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SAMPLE_DIR, STUDY_REGIONS, YEARS, FEATURE_COLS, TARGET_COL


def generate_timeseries(n_pixels: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = list(STUDY_REGIONS.keys())
    region_labels = rng.choice(regions, size=n_pixels)

    lats, lons = [], []
    for r in region_labels:
        info = STUDY_REGIONS[r]
        offset = 0.8
        lats.append(info["lat"] + rng.uniform(-offset, offset))
        lons.append(info["lon"] + rng.uniform(-offset, offset))

    # Generate NDVI time series (2000-2025)
    n_years = len(YEARS)
    # Base NDVI per pixel (forest = high)
    base_ndvi = rng.uniform(0.55, 0.85, n_pixels)

    # Deforestation events: some pixels experience sudden drops
    defor_class = rng.choice([0, 1, 2], size=n_pixels, p=[0.60, 0.25, 0.15])
    defor_year_idx = rng.integers(5, n_years - 1, size=n_pixels)

    ndvi_series = []
    for i in range(n_pixels):
        ts = []
        ndvi = base_ndvi[i]
        for yr_idx in range(n_years):
            if defor_class[i] == 2 and yr_idx == defor_year_idx[i]:  # clearcut
                ndvi = rng.uniform(0.05, 0.25)
            elif defor_class[i] == 1 and yr_idx >= defor_year_idx[i]:  # gradual degradation
                ndvi = max(ndvi - rng.uniform(0.02, 0.05), 0.15)
            ts.append(ndvi + rng.normal(0, 0.02))
        ndvi_series.append(ts)

    ndvi_arr = np.clip(np.array(ndvi_series), 0.05, 0.95)

    # Extract t0 and t1 (latest pair) + features
    ndvi_t0 = ndvi_arr[:, -2]
    ndvi_t1 = ndvi_arr[:, -1]
    ndvi_change = ndvi_t1 - ndvi_t0
    evi_t0 = ndvi_t0 * 0.85 + rng.normal(0, 0.03, n_pixels)
    evi_t1 = ndvi_t1 * 0.85 + rng.normal(0, 0.03, n_pixels)
    evi_change = evi_t1 - evi_t0

    df = pd.DataFrame({
        "latitude": lats, "longitude": lons, "region": region_labels,
        "ndvi_t0": ndvi_t0.clip(0, 1), "ndvi_t1": ndvi_t1.clip(0, 1),
        "ndvi_change": ndvi_change,
        "evi_t0": evi_t0.clip(0, 1), "evi_t1": evi_t1.clip(0, 1),
        "evi_change": evi_change,
        "slope_deg": rng.exponential(5, n_pixels).clip(0, 35),
        "elevation_m": rng.uniform(20, 1200, n_pixels),
        "distance_to_road_km": rng.exponential(10, n_pixels).clip(0.1, 80),
        "distance_to_settlement_km": rng.exponential(15, n_pixels).clip(0.1, 100),
        "rainfall_anomaly": rng.normal(0, 80, n_pixels),
        TARGET_COL: defor_class,
    })
    # Store time series columns
    for yr_idx, yr in enumerate(YEARS):
        df[f"ndvi_{yr}"] = ndvi_arr[:, yr_idx]
    return df


def load_or_generate():
    path = os.path.join(SAMPLE_DIR, "deforestation_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = generate_timeseries()
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Generated {len(df):,} pixel records")
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
