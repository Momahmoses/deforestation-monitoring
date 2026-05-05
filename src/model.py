import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


FEATURES = [
    "ndvi", "evi", "ndvi_change", "rainfall_mm", "temperature_c",
    "distance_to_road_km", "distance_to_settlement_km", "slope_deg", "elevation_m"
]
TARGET = "land_use_class"
CLASS_NAMES = ["Dense Forest", "Sparse Vegetation", "Agriculture", "Deforested/Bare"]


def train(df, config):
    snapshot = df[df["year"] == df["year"].max()].copy()
    valid = snapshot[snapshot[TARGET].isin([0, 1, 2, 3])]

    X = valid[FEATURES]
    y = valid[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"],
        n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }
    return model, metrics


def predict_land_use(model, df):
    df = df.copy()
    df["predicted_class"] = model.predict(df[FEATURES])
    df["predicted_label"] = df["predicted_class"].map({
        0: "Dense Forest", 1: "Sparse Vegetation",
        2: "Agriculture", 3: "Deforested/Bare"
    })
    return df


def compute_forest_loss(df):
    by_region_year = df.groupby(["region", "year"]).apply(
        lambda g: pd.Series({
            "forest_pixels": (g["land_use_class"] == 0).sum(),
            "deforested_pixels": (g["land_use_class"] == 3).sum(),
            "total_pixels": len(g)
        })
    ).reset_index()
    by_region_year["forest_coverage_pct"] = (
        by_region_year["forest_pixels"] / by_region_year["total_pixels"] * 100
    )
    return by_region_year


def save_model(model, path):
    joblib.dump(model, path)
