"""Main pipeline: Deforestation & Land Use Change Monitoring."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion import load_or_generate
from src.change_detection import (
    rule_based_detection, train_ml_detector,
    compute_forest_loss_timeseries, forest_loss_chart
)
from src.visualization import create_deforestation_map, annual_loss_bar


def main():
    print("=" * 60)
    print("  Deforestation & Land Use Change Monitoring")
    print("  Target: Cross River, Ondo, Edo (Nigeria) + Congo Basin")
    print("=" * 60)

    print("\n[1/5] Loading time-series NDVI data...")
    gdf = load_or_generate()
    print(f"  {len(gdf):,} pixels | Regions: {gdf['region'].nunique()}")
    clearcut_pct = (gdf["deforestation_class"] == 2).mean()
    print(f"  Clearcut pixels (ground truth): {clearcut_pct:.1%}")

    print("\n[2/5] Rule-based change detection...")
    gdf = rule_based_detection(gdf)
    rule_clearcut = (gdf["rule_class"] == 2).sum()
    print(f"  Rule-based clearcuts detected: {rule_clearcut:,}")

    print("\n[3/5] Training ML change detector...")
    clf, scaler, X_test, y_test, test_gdf = train_ml_detector(gdf)

    print("\n[4/5] Generating NDVI trend analysis...")
    ts_df = compute_forest_loss_timeseries(gdf)
    forest_loss_chart(ts_df)

    print("\n[5/5] Generating deforestation alert maps...")
    import numpy as np
    X_all = gdf[["ndvi_t0", "ndvi_t1", "ndvi_change", "evi_t0", "evi_t1", "evi_change",
                  "slope_deg", "elevation_m", "distance_to_road_km",
                  "distance_to_settlement_km", "rainfall_anomaly"]].fillna(0).values
    X_all_s = scaler.transform(X_all)
    all_preds = clf.predict(X_all_s)
    create_deforestation_map(gdf, all_preds)
    annual_loss_bar(gdf, all_preds)

    clearcut_pred = (all_preds == 2).sum()
    print(f"\n  ML-detected clearcut events: {clearcut_pred:,} ({clearcut_pred/len(all_preds):.1%})")
    print("\n✓ Pipeline complete. Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
