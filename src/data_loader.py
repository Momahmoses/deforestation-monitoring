import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


LAND_USE_CLASSES = {
    0: "Dense Forest",
    1: "Sparse Vegetation",
    2: "Agriculture",
    3: "Deforested/Bare",
    4: "Water Body",
    5: "Urban"
}


def generate_time_series_dataset(n_pixels=5000, n_years=10, seed=42):
    np.random.seed(seed)
    regions = ["Cross River", "Congo Basin", "Niger Delta"]
    lats = np.random.uniform(3.5, 7.0, n_pixels)
    lons = np.random.uniform(5.0, 14.5, n_pixels)

    baseline_ndvi = np.random.beta(a=5, b=2, size=n_pixels) * 0.9
    baseline_class = np.where(
        baseline_ndvi > 0.6, 0,
        np.where(baseline_ndvi > 0.3, 1, 3)
    )

    records = []
    for year in range(2015, 2015 + n_years):
        deforestation_rate = np.random.uniform(-0.05, 0.01, n_pixels)
        ndvi = np.clip(baseline_ndvi + deforestation_rate * (year - 2014), 0.05, 0.95)
        current_class = np.where(
            ndvi > 0.6, 0,
            np.where(ndvi > 0.3, 1,
            np.where(ndvi > 0.1, 2, 3))
        )
        for i in range(n_pixels):
            records.append({
                "pixel_id": f"PIX_{i:05d}",
                "region": regions[i % 3],
                "latitude": lats[i],
                "longitude": lons[i],
                "year": year,
                "ndvi": ndvi[i],
                "evi": ndvi[i] * 0.85 + np.random.normal(0, 0.02),
                "land_use_class": current_class[i],
                "land_use_label": LAND_USE_CLASSES.get(current_class[i], "Unknown"),
                "rainfall_mm": np.random.gamma(3, 80),
                "temperature_c": np.random.normal(27, 3),
                "distance_to_road_km": np.random.exponential(10),
                "distance_to_settlement_km": np.random.exponential(15),
                "slope_deg": np.random.exponential(5),
                "elevation_m": np.random.uniform(0, 500),
            })

    df = pd.DataFrame(records)
    df["baseline_ndvi"] = df.groupby("pixel_id")["ndvi"].transform("first")
    df["ndvi_change"] = df["ndvi"] - df["baseline_ndvi"]
    return df


def get_latest_snapshot(df):
    return df[df["year"] == df["year"].max()].copy()


def detect_alerts(df, config):
    thresholds = config["alert_thresholds"]
    alerts = df[df["ndvi_change"] <= thresholds["ndvi_change"]].copy()
    alerts["alert_severity"] = pd.cut(
        alerts["ndvi_change"],
        bins=[-np.inf, -0.3, -0.2, thresholds["ndvi_change"]],
        labels=["Critical", "High", "Moderate"]
    )
    return alerts
