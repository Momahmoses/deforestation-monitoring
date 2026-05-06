"""Folium deforestation alert maps."""

import folium
from folium.plugins import HeatMap
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUTS_DIR, STUDY_REGIONS

CLASS_COLORS = {"stable": "#27ae60", "degradation": "#f39c12", "clearcut": "#c0392b"}
CLASS_MAP = {0: "stable", 1: "degradation", 2: "clearcut"}


def create_deforestation_map(gdf: gpd.GeoDataFrame, predictions: np.ndarray):
    center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    for (_, row), pred in zip(gdf.iterrows(), predictions):
        cls = CLASS_MAP[pred]
        if cls != "stable":
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4 if cls == "degradation" else 6,
                color=CLASS_COLORS[cls], fill=True, fill_opacity=0.7,
                popup=f"Region: {row['region']}<br>Change: {cls.upper()}<br>"
                      f"NDVI Δ: {row['ndvi_change']:.3f}",
            ).add_to(m)

    for region, info in STUDY_REGIONS.items():
        folium.Marker(
            [info["lat"], info["lon"]],
            popup=f"<b>{region.replace('_',' ')}</b><br>Forest 2000: {info['forest_cover_pct_2000']}%",
            icon=folium.Icon(color="green", icon="tree-deciduous"),
        ).add_to(m)

    legend_html = """<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
    padding:12px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);font-size:12px;">
    <b>Deforestation Alert</b><br>
    <span style="color:#27ae60;">&#9632;</span> Stable<br>
    <span style="color:#f39c12;">&#9632;</span> Degradation<br>
    <span style="color:#c0392b;">&#9632;</span> Clearcut</div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    out = os.path.join(OUTPUTS_DIR, "deforestation_alert_map.html")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    m.save(out)
    print(f"Alert map saved → {out}")


def annual_loss_bar(gdf: gpd.GeoDataFrame, predictions: np.ndarray):
    gdf = gdf.copy()
    gdf["pred_class"] = [CLASS_MAP[p] for p in predictions]
    summary = gdf.groupby(["region", "pred_class"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [CLASS_COLORS.get(c, "gray") for c in summary.columns]
    summary.plot(kind="bar", ax=ax, color=colors, width=0.7)
    ax.set_title("Deforestation Classification by Region")
    ax.set_ylabel("Pixel Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "region_loss_summary.png"), dpi=150)
    plt.close()
    print("Region loss chart saved.")
