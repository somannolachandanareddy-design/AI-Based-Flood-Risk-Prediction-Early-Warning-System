CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS flood_zones (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    note TEXT,
    source TEXT NOT NULL DEFAULT 'Published Kathmandu Valley flood susceptibility research',
    geom GEOMETRY(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_flood_zones_geom ON flood_zones USING GIST (geom);

CREATE TABLE IF NOT EXISTS rainfall_events (
    id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    location TEXT NOT NULL,
    rainfall_mm NUMERIC NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id SERIAL PRIMARY KEY,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    rainfall_mm NUMERIC NOT NULL,
    geom GEOMETRY(Point, 4326) NOT NULL,
    risk_score NUMERIC NOT NULL,
    risk_level TEXT NOT NULL,
    elevation_m NUMERIC,
    slope NUMERIC,
    source_model TEXT DEFAULT 'xgboost_terrain_rainfall'
);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_geom ON risk_predictions USING GIST (geom);

CREATE TABLE IF NOT EXISTS satellite_flood_detections (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    geom GEOMETRY(Point, 4326) NOT NULL,
    flood_probability NUMERIC NOT NULL,
    source_model TEXT DEFAULT 'pytorch_sentinel_cnn',
    sentinel_pass_date DATE
);
CREATE INDEX IF NOT EXISTS idx_satellite_detections_geom ON satellite_flood_detections USING GIST (geom);

INSERT INTO rainfall_events (event_date, location, rainfall_mm, notes) VALUES
    ('1993-07-20', 'Kathmandu Valley', 540.0, '1993 Bagmati floods, 1,336 deaths, deadliest on record for the valley')
ON CONFLICT DO NOTHING;
