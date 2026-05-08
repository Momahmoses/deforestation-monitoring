"""
Synthetic time-series dataset for deforestation and land-use change monitoring.
Each record represents a 500m x 500m forest grid cell observed across multiple years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_CELLS = 3000
YEARS = list(range(2015, 2024))

REGIONS = [
    ("Cross River", 5.86, 8.60), ("Edo", 6.34, 5.63),
    ("Ondo", 7.10, 5.05), ("Congo Basin", -0.23, 21.76),
    ("Cameroon Highlands", 5.96, 10.15), ("Ogun", 7.00, 3.35),
]

DEFORESTATION_DRIVERS = [
    "Agricultural Expansion", "Illegal Logging", "Charcoal Production",
    "Infrastructure Development", "Mining", "Subsistence Farming",
]


def generate_deforestation_dataset(n_cells: int = N_CELLS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic forest monitoring dataset with annual observations.

    Returns
    -------
    pd.DataFrame
        Annual forest-cover records per cell with change detection flags.
    """
    rng = np.random.default_rng(seed)
    records = []

    for cell_id in range(n_cells):
        region, lat, lon = REGIONS[rng.integers(0, len(REGIONS))]
        lat_j = lat + rng.normal(0, 0.5)
        lon_j = lon + rng.normal(0, 0.5)
        initial_cover = float(np.clip(rng.beta(5, 2), 0.3, 1.0))
        is_protected_area = int(rng.random() < 0.25)
        proximity_settlement_km = max(0.2, float(rng.exponential(15)))
        primary_driver = DEFORESTATION_DRIVERS[rng.integers(0, len(DEFORESTATION_DRIVERS))]

        annual_loss_rate = float(rng.beta(1.5, 15)) * (0.3 if is_protected_area else 1.0)
        cover = initial_cover

        for year in YEARS:
            ndvi = float(np.clip(cover * 0.85 + rng.normal(0, 0.03), 0.05, 0.95))
            rainfall_mm = max(100, float(rng.normal(1600 - (year - 2015) * 10, 200)))
            fire_events = int(rng.poisson(0.3 if cover > 0.6 else 0.9))
            encroachment_events = int(rng.poisson(max(0, (1 - cover) * 3)))
            cover_loss = float(annual_loss_rate * rng.uniform(0.5, 1.5))
            new_cover = float(max(0.02, cover - cover_loss))
            deforestation_alert = int(cover_loss > 0.02)

            records.append({
                "cell_id": cell_id, "region": region, "year": year,
                "latitude": round(lat_j, 6), "longitude": round(lon_j, 6),
                "forest_cover_pct": round(cover * 100, 2),
                "ndvi": round(ndvi, 4),
                "annual_cover_loss_pct": round(cover_loss * 100, 3),
                "rainfall_mm": round(rainfall_mm, 1),
                "fire_events": fire_events,
                "encroachment_events": encroachment_events,
                "is_protected_area": is_protected_area,
                "proximity_settlement_km": round(proximity_settlement_km, 2),
                "primary_driver": primary_driver,
                "deforestation_alert": deforestation_alert,
            })
            cover = new_cover

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_deforestation_dataset()
    path = output_dir / "deforestation_data.csv"
    df.to_csv(path, index=False)
    return path
