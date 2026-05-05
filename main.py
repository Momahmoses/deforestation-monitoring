import os
from src.data_loader import load_config, generate_time_series_dataset, get_latest_snapshot, detect_alerts
from src.model import train, predict_land_use, compute_forest_loss, save_model
from src.visualize import plot_change_map, plot_ndvi_trend, plot_land_use_change


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = load_config("config.yaml")
    print(f"[1/5] Config loaded — targets: {', '.join(config['targets'])}")

    df = generate_time_series_dataset(n_pixels=2000, n_years=10)
    print(f"[2/5] {len(df)} pixel-year records generated ({df['year'].nunique()} years, {df['pixel_id'].nunique()} pixels)")

    model, metrics = train(df, config)
    print(f"[3/5] Land use classifier trained — Accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report"])
    save_model(model, "models/land_use_model.pkl")

    snapshot = get_latest_snapshot(df)
    snapshot = predict_land_use(model, snapshot)

    alerts = detect_alerts(snapshot, config)
    alerts.to_csv(config["output"]["alert_report"], index=False)

    trend = compute_forest_loss(df)
    print(f"[4/5] {len(alerts)} deforestation alerts generated")
    print(f"      Forest coverage trend (latest year, by region):")
    latest_trend = trend[trend["year"] == trend["year"].max()][["region", "forest_coverage_pct"]]
    print(latest_trend.to_string(index=False))

    plot_change_map(snapshot, alerts, config["output"]["change_map"])
    plot_ndvi_trend(trend, config["output"]["trend_chart"])
    plot_land_use_change(df, config["output"]["land_use_chart"])
    print("[5/5] All outputs saved to /outputs/")
    print("\nDone.")


if __name__ == "__main__":
    main()
