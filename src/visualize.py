import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap


LAND_COLORS = {
    "Dense Forest": "#1a9641",
    "Sparse Vegetation": "#a6d96a",
    "Agriculture": "#fdae61",
    "Deforested/Bare": "#d7191c",
    "Water Body": "#2166ac",
    "Urban": "#969696"
}
ALERT_COLORS = {"Critical": "#d73027", "High": "#fc8d59", "Moderate": "#fee090"}


def plot_change_map(snapshot_df, alerts_df, output_path):
    center = [snapshot_df["latitude"].mean(), snapshot_df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    for _, row in snapshot_df.sample(min(600, len(snapshot_df))).iterrows():
        color = LAND_COLORS.get(str(row.get("predicted_label", "Dense Forest")), "#ccc")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4, color=color, fill=True, fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>Region:</b> {row['region']}<br>"
                f"<b>Land Use:</b> {row.get('predicted_label', 'N/A')}<br>"
                f"<b>NDVI:</b> {row['ndvi']:.3f}<br>"
                f"<b>NDVI Change:</b> {row['ndvi_change']:.3f}",
                max_width=220
            )
        ).add_to(m)

    for _, row in alerts_df.iterrows():
        color = ALERT_COLORS.get(str(row.get("alert_severity", "Moderate")), "#fee090")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8, color=color, fill=True, fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>ALERT — {row.get('alert_severity', 'N/A')}</b><br>"
                f"Region: {row['region']}<br>"
                f"NDVI Change: {row['ndvi_change']:.3f}",
                max_width=200
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Deforestation map saved: {output_path}")


def plot_ndvi_trend(trend_df, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for region in trend_df["region"].unique():
        sub = trend_df[trend_df["region"] == region]
        ax.plot(sub["year"], sub["forest_coverage_pct"], marker="o", label=region)
    ax.set_title("Forest Coverage % Over Time by Region")
    ax.set_xlabel("Year")
    ax.set_ylabel("Forest Coverage (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_land_use_change(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    snapshot_2015 = df[df["year"] == 2015]["land_use_label"].value_counts()
    snapshot_2024 = df[df["year"] == df["year"].max()]["land_use_label"].value_counts()

    colors = [LAND_COLORS.get(k, "#ccc") for k in snapshot_2015.index]
    axes[0].pie(snapshot_2015.values, labels=snapshot_2015.index, colors=colors, autopct="%1.1f%%")
    axes[0].set_title("Land Use 2015 (Baseline)")

    colors2 = [LAND_COLORS.get(k, "#ccc") for k in snapshot_2024.index]
    axes[1].pie(snapshot_2024.values, labels=snapshot_2024.index, colors=colors2, autopct="%1.1f%%")
    axes[1].set_title(f"Land Use {df['year'].max()} (Current)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
