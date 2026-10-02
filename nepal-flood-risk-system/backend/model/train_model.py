"""
Trains an XGBoost flood-risk classifier for the Kathmandu Valley.

REAL DATA USED:
 - 9 named, real flood-susceptible river/confluence zones from published
   flood susceptibility research on the Kathmandu Valley watershed
 - Real recorded 1993 extreme rainfall (540mm/24hr, 65mm/hr peak intensity)
 - Real published valley elevation range (1198-2733m ASL)

APPROXIMATED (flagged, not fabricated as real):
 - Per-cell elevation/slope surface, same reasoning as the Hyderabad model:
   this sandbox has no live network access to Google Earth Engine/SRTM, so a
   physically-informed terrain proxy is used (low near real river/confluence
   points, high toward valley rim, scaled into the REAL published elevation
   range). Swap in backend/gee_integration.py's get_real_elevation() /
   get_real_slope() for genuine SRTM data in production — zero other changes.
"""
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.real_flood_points import REAL_FLOOD_ZONES, BBOX, REAL_ELEVATION_RANGE_M

FLOOD_XY = np.array([(lon, lat) for _, lon, lat, _ in REAL_FLOOD_ZONES])

# Valley rim (real geographic feature — the ring of hills enclosing Kathmandu
# Valley) used as high-ground anchors. Approximate rim points at the cardinal
# hill ranges (Shivapuri north, Phulchowki south, Nagarjun west, Bhaktapur
# hills east) — real named hills, coordinates approximate to the ridge line.
VALLEY_RIM_ANCHORS = [
    ("Shivapuri (north rim)", 85.3800, 27.8100),
    ("Phulchowki (south rim)", 85.3900, 27.5900),
    ("Nagarjun (west rim)", 85.2400, 27.7500),
    ("Bhaktapur hills (east rim)", 85.4700, 27.6900),
]
RIM_XY = np.array([(lon, lat) for _, lon, lat in VALLEY_RIM_ANCHORS])


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def min_dist_km(lon, lat, anchors_xy):
    return min(haversine_km(lon, lat, a[0], a[1]) for a in anchors_xy)


def estimate_elevation(lon, lat):
    """Physically-informed elevation proxy for the Kathmandu Valley (see docstring)."""
    d_low = min_dist_km(lon, lat, FLOOD_XY)
    d_high = min_dist_km(lon, lat, RIM_XY)
    low_pull = math.exp(-d_low / 1.5)
    high_pull = math.exp(-d_high / 3.0)
    base = REAL_ELEVATION_RANGE_M["valley_floor_avg"]
    span = REAL_ELEVATION_RANGE_M["max"] - REAL_ELEVATION_RANGE_M["min"]
    elev = base - (span * 0.35 * low_pull) + (span * 0.65 * high_pull)
    return float(np.clip(elev, REAL_ELEVATION_RANGE_M["min"], REAL_ELEVATION_RANGE_M["max"]))


def build_grid(step=0.005):
    lons = np.arange(BBOX["min_lon"], BBOX["max_lon"], step)
    lats = np.arange(BBOX["min_lat"], BBOX["max_lat"], step)
    return [(lon, lat) for lat in lats for lon in lons]


def compute_features(lon, lat, rainfall_mm):
    elev = estimate_elevation(lon, lat)
    eps = 0.001
    e_dx = estimate_elevation(lon + eps, lat) - elev
    e_dy = estimate_elevation(lon, lat + eps) - elev
    slope = math.sqrt(e_dx ** 2 + e_dy ** 2) / eps

    dist_to_river_km = min_dist_km(lon, lat, FLOOD_XY)
    return {
        "lon": lon, "lat": lat,
        "elevation_m": elev,
        "slope": slope,
        "dist_to_known_flood_zone_km": dist_to_river_km,
        "rainfall_mm": rainfall_mm,
    }


def make_training_set():
    """
    Positive samples: the 9 real named flood-susceptible zones, replicated
    across realistic monsoon rainfall intensities up to the real 1993 record
    (540mm/24hr). Negative samples: grid cells far from any known zone.
    """
    rainfall_scenarios = [50, 100, 180, 300, 420, 540.0]
    rows = []
    for _, lon, lat, _ in REAL_FLOOD_ZONES:
        for rf in rainfall_scenarios:
            f = compute_features(lon, lat, rf)
            f["label"] = 1 if rf >= 100 else 0
            rows.append(f)

    rng = np.random.default_rng(42)
    for lon, lat in build_grid(step=0.006):
        if min_dist_km(lon, lat, FLOOD_XY) < 0.9:
            continue
        rf = rng.choice(rainfall_scenarios)
        f = compute_features(lon, lat, float(rf))
        f["label"] = 0
        rows.append(f)

    return pd.DataFrame(rows)


def train():
    df = make_training_set()
    feature_cols = ["elevation_m", "slope", "dist_to_known_flood_zone_km", "rainfall_mm"]
    X, y = df[feature_cols], df["label"]

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42,
    )
    model.fit(X, y)

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "flood_xgb.joblib"))
    with open(os.path.join(out_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    print(f"Trained on {len(df)} samples ({int(y.sum())} positive / {len(y)-int(y.sum())} negative)")
    print(f"Train accuracy: {model.score(X, y):.3f}")
    print("Feature importances:", dict(zip(feature_cols, model.feature_importances_.round(3).tolist())))
    return model, feature_cols


if __name__ == "__main__":
    train()
