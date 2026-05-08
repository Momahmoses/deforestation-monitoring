"""Change detection algorithms for deforestation monitoring."""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURE_COLS, TARGET_COL, OUTPUTS_DIR, YEARS, NDVI_CHANGE_THRESHOLD, ALERT_CHANGE_THRESHOLD

CLASS_NAMES = ["stable", "degradation", "clearcut"]


def rule_based_detection(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["rule_class"] = 0
    gdf.loc[gdf["ndvi_change"] < NDVI_CHANGE_THRESHOLD, "rule_class"] = 1
    gdf.loc[gdf["ndvi_change"] < ALERT_CHANGE_THRESHOLD, "rule_class"] = 2
    return gdf


def train_ml_detector(gdf: gpd.GeoDataFrame):
    X = gdf[FEATURE_COLS].fillna(0).values
    y = gdf[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_ts_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
    clf.fit(X_tr_s, y_train)
    y_pred = clf.predict(X_ts_s)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(OUTPUTS_DIR, "deforestation_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.pkl"))
    return clf, scaler, X_ts_s, y_test, gdf.iloc[len(X_train):]


def compute_forest_loss_timeseries(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    year_cols = [f"ndvi_{yr}" for yr in YEARS if f"ndvi_{yr}" in gdf.columns]
    yearly_mean = gdf.groupby("region")[[c for c in year_cols]].mean()
    records = []
    for region in yearly_mean.index:
        for col in year_cols:
            yr = int(col.split("_")[1])
            records.append({"region": region, "year": yr, "mean_ndvi": yearly_mean.loc[region, col]})
    return pd.DataFrame(records)


def forest_loss_chart(ts_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    regions = ts_df["region"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))
    for region, color in zip(regions, colors):
        d = ts_df[ts_df["region"] == region]
        ax.plot(d["year"], d["mean_ndvi"], label=region.replace("_", " "), color=color, lw=2, marker="o", ms=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean NDVI")
    ax.set_title("Forest NDVI Trend (2000–2025) — Deforestation Monitoring")
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, "ndvi_trend.png"), dpi=150)
    plt.close()
    print("NDVI trend chart saved.")
