"""
PRODUCTION satellite data integration — Google Earth Engine (Sentinel-1/2 + SRTM).

This sandbox environment has no outbound network access to earthengine.googleapis.com,
so this module cannot run here. It is provided so that, on your own machine/server
(with `earthengine-api` installed and `earthengine authenticate` run once), you can
swap the approximated terrain in model/train_model.py for the real thing with almost
no code change — same feature names, same model, same API.

Install:
    pip install earthengine-api google-auth

Auth (one-time, opens a browser):
    earthengine authenticate

Then set GEE_PROJECT env var to your Earth Engine cloud project id.
"""
import os
import ee

_initialized = False


def init_gee():
    global _initialized
    if not _initialized:
        ee.Initialize(project=os.environ.get("GEE_PROJECT"))
        _initialized = True


def get_real_elevation(lon: float, lat: float) -> float:
    """Real SRTM 30m elevation (metres) for a single point."""
    init_gee()
    point = ee.Geometry.Point([lon, lat])
    dem = ee.Image("USGS/SRTMGL1_003")
    value = dem.sample(point, 30).first().get("elevation").getInfo()
    return float(value)


def get_real_slope(lon: float, lat: float) -> float:
    """Real terrain slope (degrees) derived from SRTM at a point."""
    init_gee()
    point = ee.Geometry.Point([lon, lat])
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    value = slope.sample(point, 30).first().get("slope").getInfo()
    return float(value)


def get_latest_sentinel1_flood_extent(bbox: dict, days_back: int = 5):
    """
    Real-time surface-water / flood extent from Sentinel-1 SAR (works day or
    night, through cloud cover — critical for flood monitoring during storms).
    Returns a water-mask image clipped to bbox; threshold VV backscatter to
    detect open water / inundation.
    """
    init_gee()
    region = ee.Geometry.Rectangle(
        [bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]]
    )
    end = ee.Date(ee.Date.now())
    start = end.advance(-days_back, "day")
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )
    latest = collection.sort("system:time_start", False).first()
    vv = latest.select("VV")
    # Open water/flooded surfaces have low VV backscatter (smooth surface)
    water_mask = vv.lt(-17)
    return water_mask.clip(region)


def get_latest_sentinel2_ndwi(bbox: dict, days_back: int = 10):
    """
    Real-time Normalized Difference Water Index from Sentinel-2 optical
    imagery — complements Sentinel-1 SAR on cloud-free days for validating
    surface water extent.
    """
    init_gee()
    region = ee.Geometry.Rectangle(
        [bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]]
    )
    end = ee.Date(ee.Date.now())
    start = end.advance(-days_back, "day")
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    img = collection.first()
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return ndwi.clip(region)
