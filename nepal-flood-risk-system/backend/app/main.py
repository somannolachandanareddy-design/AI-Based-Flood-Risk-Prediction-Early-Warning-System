"""
FastAPI backend — AI-Based Flood Risk Prediction & Early Warning, Kathmandu Valley.

Endpoints:
  GET  /api/health
  GET  /api/flood-zones          -> real named flood-susceptible zones (GeoJSON)
  GET  /api/risk-grid            -> XGBoost terrain+rainfall risk heatmap (GeoJSON)
  POST /api/predict              -> risk score for an arbitrary lat/lon + rainfall
  GET  /api/rainfall-events      -> real historical extreme rainfall records
  POST /api/predict-satellite    -> PyTorch CNN flood-extent probability from a patch
"""
import json
import os
import sys

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.real_flood_points import REAL_FLOOD_ZONES, BBOX, REAL_RAINFALL_EVENTS_MM
from model.train_model import compute_features, build_grid
from model.flood_extent_cnn import FloodExtentCNN, predict_patch

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "artifacts")

app = FastAPI(title="Nepal Flood Risk Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model, _feature_cols, _cnn = None, None, None


def get_xgb():
    global _model, _feature_cols
    if _model is None:
        model_path = os.path.join(ARTIFACT_DIR, "flood_xgb.joblib")
        if not os.path.exists(model_path):
            raise HTTPException(500, "XGBoost model not trained. Run model/train_model.py first.")
        _model = joblib.load(model_path)
        with open(os.path.join(ARTIFACT_DIR, "feature_cols.json")) as f:
            _feature_cols = json.load(f)
    return _model, _feature_cols


def get_cnn():
    global _cnn
    if _cnn is None:
        cnn_path = os.path.join(ARTIFACT_DIR, "flood_extent_cnn.pt")
        if not os.path.exists(cnn_path):
            raise HTTPException(500, "CNN not trained. Run model/flood_extent_cnn.py first.")
        _cnn = FloodExtentCNN()
        _cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        _cnn.eval()
    return _cnn


class PredictRequest(BaseModel):
    lat: float
    lon: float
    rainfall_mm: float = 150.0


class SatellitePredictRequest(BaseModel):
    ndwi_patch: list  # 32x32 floats, NDWI band
    sar_patch: list   # 32x32 floats, Sentinel-1 VV backscatter band


def risk_level(p):
    if p >= 0.7:
        return "HIGH"
    if p >= 0.4:
        return "MODERATE"
    return "LOW"


@app.get("/api/health")
def health():
    return {"status": "ok", "xgb_loaded": _model is not None, "cnn_loaded": _cnn is not None}


@app.get("/api/rainfall-events")
def rainfall_events():
    return {"events_mm": REAL_RAINFALL_EVENTS_MM,
            "note": "1993 Bagmati floods: 540mm/24hr, 1,336 deaths — deadliest flood on record for the Kathmandu Valley."}


@app.get("/api/flood-zones")
def flood_zones():
    features = []
    for name, lon, lat, note in REAL_FLOOD_ZONES:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "note": note, "source": "Published Kathmandu Valley flood susceptibility research"},
        })
    return {"type": "FeatureCollection", "features": features}


@app.post("/api/predict")
def predict(req: PredictRequest):
    model, feature_cols = get_xgb()
    f = compute_features(req.lon, req.lat, req.rainfall_mm)
    X = np.array([[f[c] for c in feature_cols]])
    proba = float(model.predict_proba(X)[0][1])
    return {
        "lat": req.lat, "lon": req.lon, "rainfall_mm": req.rainfall_mm,
        "risk_score": round(proba, 4), "risk_level": risk_level(proba),
        "features": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()},
    }


@app.get("/api/risk-grid")
def risk_grid(rainfall_mm: float = Query(150.0, ge=0, le=600), step: float = Query(0.006, ge=0.003, le=0.02)):
    model, feature_cols = get_xgb()
    grid = build_grid(step=step)
    rows = [compute_features(lon, lat, rainfall_mm) for lon, lat in grid]
    X = np.array([[r[c] for c in feature_cols] for r in rows])
    probs = model.predict_proba(X)[:, 1]

    features = []
    for r, p in zip(rows, probs):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {"risk_score": round(float(p), 4), "risk_level": risk_level(float(p))},
        })
    return {"type": "FeatureCollection", "rainfall_mm": rainfall_mm, "bbox": BBOX, "features": features}


@app.post("/api/predict-satellite")
def predict_satellite(req: SatellitePredictRequest):
    """
    PyTorch CNN inference on an NDWI+SAR patch pair. Demo model — see
    model/flood_extent_cnn.py docstring: architecture and training loop are
    real and working, but trained on synthetic patches in this sandbox since
    live Sentinel-1/2 downloads aren't reachable here. Feed it real patches
    pulled via gee_integration.py in production for genuine satellite-based
    flood confirmation.
    """
    cnn = get_cnn()
    ndwi = np.array(req.ndwi_patch, dtype=np.float32)
    sar = np.array(req.sar_patch, dtype=np.float32)
    if ndwi.shape != (32, 32) or sar.shape != (32, 32):
        raise HTTPException(400, "Both patches must be 32x32 arrays.")
    prob = predict_patch(cnn, ndwi, sar)
    return {"flood_probability": round(prob, 4), "is_flooded": prob >= 0.5,
            "note": "Demo CNN trained on synthetic patches — see model/flood_extent_cnn.py for how to wire real Sentinel data."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
