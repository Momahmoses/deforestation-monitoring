"""
Deforestation & Land Use Change Monitoring
===========================================
Time-series satellite imagery analysis platform detecting illegal logging,
land encroachment, and forest cover decline across Cross River, Congo Basin,
and West-Central Africa. Generates real-time deforestation alerts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_deforestation_dataset, YEARS
from model import (
    FEATURE_COLS, METRICS_PATH, MODEL_PATH,
    load_model, predict_alert, save_model, train,
)

st.set_page_config(page_title="Deforestation Monitoring", page_icon="🌳", layout="wide")

SEVERITY_COLORS = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}
ALERT_COLORS = {1: "#E74C3C", 0: "#2ECC71"}


@st.cache_resource(show_spinner="Processing satellite time-series data…")
def get_model_and_data():
    df = generate_deforestation_dataset()
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        pipeline = load_model()
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        pipeline, metrics = train(df)
        save_model(pipeline, metrics)
    df["alert_probability"] = pipeline.predict_proba(df[FEATURE_COLS])[:, 1]
    return pipeline, metrics, df


pipeline, metrics, df = get_model_and_data()

with st.sidebar:
    st.title("🌳 Deforestation Monitor")
    st.caption("Africa Forest Watch Platform")
    st.divider()
    page = st.radio("Navigation", ["Overview", "Forest Cover Map", "Time-Series Analysis", "Alert Detector", "Model Performance"], label_visibility="collapsed")
    st.divider()
    region_filter = st.multiselect("Filter by Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    year_filter = st.select_slider("Year", options=YEARS, value=(YEARS[0], YEARS[-1]))

df_f = df[(df["region"].isin(region_filter)) & (df["year"].between(*year_filter))]

if page == "Overview":
    st.title("Deforestation & Land Use Change Monitoring")
    st.markdown("Real-time forest cover monitoring using time-series satellite imagery across Cross River, Congo Basin, Cameroon Highlands, and West Africa.")
    st.divider()

    total_loss = df_f["annual_cover_loss_pct"].sum()
    alerts = df_f["deforestation_alert"].sum()
    protected_alerts = df_f[(df_f["deforestation_alert"] == 1) & (df_f["is_protected_area"] == 1)].shape[0]
    mean_cover = df_f.groupby("year")["forest_cover_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df_f):,}")
    c2.metric("Deforestation Alerts", f"{alerts:,}", f"{alerts/len(df_f):.1%} of cells", delta_color="inverse")
    c3.metric("Alerts in Protected Areas", f"{protected_alerts:,}", delta_color="inverse")
    c4.metric("Model ROC-AUC", f"{metrics['roc_auc_test']:.4f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        annual_loss = df_f.groupby("year")["annual_cover_loss_pct"].mean().reset_index()
        fig = px.area(annual_loss, x="year", y="annual_cover_loss_pct", color_discrete_sequence=["#E74C3C"],
                      title="Mean Annual Forest Cover Loss (%)", labels={"annual_cover_loss_pct": "Loss (%)"}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        driver_counts = df_f[df_f["deforestation_alert"] == 1]["primary_driver"].value_counts().reset_index()
        fig2 = px.bar(driver_counts, x="count", y="primary_driver", orientation="h",
                      color="count", color_continuous_scale="Reds",
                      title="Alert Count by Deforestation Driver", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        region_loss = df_f.groupby(["region", "year"])["forest_cover_pct"].mean().reset_index()
        fig3 = px.line(region_loss, x="year", y="forest_cover_pct", color="region",
                       title="Forest Cover Trend by Region (%)", height=380,
                       labels={"forest_cover_pct": "Forest Cover (%)"})
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.scatter(df_f.sample(min(1500, len(df_f)), random_state=42),
                          x="proximity_settlement_km", y="annual_cover_loss_pct",
                          color="deforestation_alert", color_discrete_map=ALERT_COLORS,
                          opacity=0.4, title="Proximity to Settlement vs Cover Loss", height=380)
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Forest Cover Map":
    st.title("Forest Cover Map")
    latest_year = df_f["year"].max()
    latest = df_f[df_f["year"] == latest_year].copy()
    latest["alert_label"] = latest["deforestation_alert"].map({1: "Alert", 0: "Stable"})
    fig = px.scatter_mapbox(
        latest.sample(min(1500, len(latest)), random_state=42),
        lat="latitude", lon="longitude",
        color="forest_cover_pct", size="annual_cover_loss_pct", size_max=14,
        color_continuous_scale="Greens",
        hover_name="region",
        hover_data={"forest_cover_pct": ":.1f", "annual_cover_loss_pct": ":.3f", "primary_driver": True, "latitude": False, "longitude": False},
        mapbox_style="carto-positron", zoom=3,
        center={"lat": 5.0, "lon": 8.0},
        title=f"Forest Cover Map — {latest_year}", height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Time-Series Analysis":
    st.title("Multi-Year Time-Series Analysis")
    region_sel = st.selectbox("Select Region", sorted(df["region"].unique()))
    region_df = df[df["region"] == region_sel]
    annual = region_df.groupby("year").agg(
        mean_cover=("forest_cover_pct", "mean"),
        mean_loss=("annual_cover_loss_pct", "mean"),
        total_alerts=("deforestation_alert", "sum"),
        mean_ndvi=("ndvi", "mean"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=annual["year"], y=annual["mean_cover"], name="Mean Forest Cover (%)", line={"color": "#27AE60", "width": 2.5}))
    fig.add_trace(go.Bar(x=annual["year"], y=annual["total_alerts"], name="Annual Alerts", marker_color="#E74C3C", yaxis="y2", opacity=0.6))
    fig.update_layout(
        title=f"Forest Cover Trend & Alert Frequency — {region_sel}",
        yaxis={"title": "Forest Cover (%)"},
        yaxis2={"title": "Alert Count", "overlaying": "y", "side": "right"},
        height=450, legend={"x": 0.01, "y": 0.99},
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Alert Detector":
    st.title("Real-Time Deforestation Alert Detector")
    col1, col2 = st.columns(2)
    with col1:
        forest_cover_pct = st.slider("Current Forest Cover (%)", 5.0, 100.0, 65.0, 0.5)
        ndvi = st.slider("NDVI", 0.05, 0.95, 0.60, 0.01)
        rainfall_mm = st.slider("Annual Rainfall (mm)", 200.0, 3000.0, 1500.0, 50.0)
        fire_events = st.number_input("Fire Events This Year", 0, 20, 1)
    with col2:
        encroachment_events = st.number_input("Encroachment Events", 0, 15, 0)
        is_protected = st.checkbox("Protected Area?", value=False)
        proximity_settlement_km = st.slider("Distance to Nearest Settlement (km)", 0.2, 80.0, 12.0, 0.5)
        year = st.selectbox("Year", YEARS, index=len(YEARS) - 1)

    if st.button("Run Alert Detection", type="primary"):
        result = predict_alert(pipeline, {
            "forest_cover_pct": forest_cover_pct, "ndvi": ndvi, "rainfall_mm": rainfall_mm,
            "fire_events": fire_events, "encroachment_events": encroachment_events,
            "is_protected_area": int(is_protected), "proximity_settlement_km": proximity_settlement_km,
            "year": year,
        })
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Alert Probability", f"{result['alert_probability']:.2%}")
        r2.metric("Alert Status", "ALERT TRIGGERED" if result["alert_triggered"] else "No Alert")
        r3.metric("Severity", result["severity"])
        if result["alert_triggered"]:
            st.error("🚨 Deforestation alert triggered. Notify forest rangers for immediate ground verification.")
        else:
            st.success("No deforestation event detected for this observation window.")

elif page == "Model Performance":
    st.title("Model Performance — Random Forest Classifier")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", f"{metrics['roc_auc_test']:.4f}")
    m2.metric("CV AUC", f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    m3.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m4.metric("Alert F1-Score", f"{metrics['f1_alert']:.4f}")
    cm = metrics["confusion_matrix"]
    fig = go.Figure(go.Heatmap(
        z=np.array(cm), x=["No Alert", "Alert"], y=["No Alert", "Alert"],
        colorscale="Greens", showscale=False,
        text=[[str(v) for v in row] for row in cm], texttemplate="%{text}", textfont={"size": 18},
    ))
    fig.update_layout(title="Confusion Matrix", height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full Metrics"):
        st.json(metrics)
