import { useEffect, useState, useCallback } from "react";
import { MapContainer, TileLayer, CircleMarker, Marker, Popup, Tooltip, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const API_BASE = "http://localhost:8002";

const RISK_COLORS = { HIGH: "#dc2626", MODERATE: "#f59e0b", LOW: "#16a34a" };

const zoneIcon = new L.DivIcon({
  className: "",
  html: `<div style="width:14px;height:14px;background:#1e3a8a;border:2px solid white;border-radius:50%;box-shadow:0 0 0 2px #1e3a8a55;"></div>`,
  iconSize: [14, 14], iconAnchor: [7, 7],
});

const RAINFALL_PRESETS = [
  { label: "Light monsoon (50mm)", value: 50 },
  { label: "Moderate (180mm)", value: 180 },
  { label: "Heavy (300mm)", value: 300 },
  { label: "Severe (420mm)", value: 420 },
  { label: "1993 flood record (540mm)", value: 540 },
];

function ClickCatcher({ onClick }) {
  useMapEvents({ click: onClick });
  return null;
}

export default function App() {
  const [rainfall, setRainfall] = useState(180);
  const [grid, setGrid] = useState(null);
  const [zones, setZones] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [satResult, setSatResult] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/flood-zones`).then((r) => r.json()).then(setZones)
      .catch((e) => setError("Backend not reachable: " + e.message));
  }, []);

  const loadGrid = useCallback((rf) => {
    setLoading(true);
    fetch(`${API_BASE}/api/risk-grid?rainfall_mm=${rf}&step=0.006`)
      .then((r) => r.json()).then((d) => { setGrid(d); setLoading(false); })
      .catch((e) => { setError("Backend not reachable: " + e.message); setLoading(false); });
  }, []);

  useEffect(() => { loadGrid(rainfall); }, []); // eslint-disable-line

  const handleMapClick = async (e) => {
    const { lat, lng } = e.latlng;
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon: lng, rainfall_mm: rainfall }),
      });
      setSelected(await res.json());
    } catch (err) { setError("Backend not reachable: " + err.message); }
  };

  const runSatelliteDemo = async (kind) => {
    const patch = kind === "water"
      ? Array.from({ length: 32 }, () => Array.from({ length: 32 }, () => 0.7 + (Math.random() - 0.5) * 0.05))
      : Array.from({ length: 32 }, () => Array.from({ length: 32 }, () => Math.random() * 0.6));
    const sar = kind === "water"
      ? Array.from({ length: 32 }, () => Array.from({ length: 32 }, () => 0.2 + (Math.random() - 0.5) * 0.05))
      : Array.from({ length: 32 }, () => Array.from({ length: 32 }, () => Math.random() * 0.6));
    try {
      const res = await fetch(`${API_BASE}/api/predict-satellite`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ndwi_patch: patch, sar_patch: sar }),
      });
      setSatResult(await res.json());
    } catch (err) { setError("Backend not reachable: " + err.message); }
  };

  const counts = grid ? grid.features.reduce((acc, f) => {
    acc[f.properties.risk_level] = (acc[f.properties.risk_level] || 0) + 1;
    return acc;
  }, {}) : {};

  return (
    <div className="h-full w-full flex flex-col bg-slate-950 text-slate-100">
      <header className="px-5 py-3 border-b border-slate-800 flex items-center justify-between shrink-0 flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">🌊 Kathmandu Valley Flood Risk — Live Prediction Map</h1>
          <p className="text-xs text-slate-400">XGBoost terrain+rainfall model + PyTorch satellite CNN · Bagmati river basin, Nepal</p>
        </div>
        {error && <span className="text-xs text-red-400 bg-red-950 px-3 py-1 rounded-full border border-red-800">{error}</span>}
      </header>

      <div className="flex flex-1 min-h-0 flex-col md:flex-row">
        <aside className="w-full md:w-80 shrink-0 border-r border-slate-800 p-4 overflow-y-auto space-y-5">
          <div>
            <label className="text-sm font-medium block mb-2">
              Rainfall intensity: <span className="text-blue-400">{rainfall} mm</span>
            </label>
            <input type="range" min="20" max="600" step="10" value={rainfall}
              onChange={(e) => setRainfall(Number(e.target.value))}
              onMouseUp={() => loadGrid(rainfall)} onTouchEnd={() => loadGrid(rainfall)}
              className="w-full accent-blue-500" />
            <div className="flex flex-wrap gap-1.5 mt-2">
              {RAINFALL_PRESETS.map((p) => (
                <button key={p.value} onClick={() => { setRainfall(p.value); loadGrid(p.value); }}
                  className="text-[11px] px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700">
                  {p.label}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-500 mt-2">540mm is the real recorded 24-hour rainfall from the deadly 1993 Bagmati floods (1,336 deaths).</p>
          </div>

          <div className="border-t border-slate-800 pt-4">
            <h2 className="text-sm font-medium mb-2">Risk distribution</h2>
            {loading ? <p className="text-xs text-slate-500">Recomputing grid…</p> : (
              <div className="space-y-1.5">
                {["HIGH", "MODERATE", "LOW"].map((lvl) => (
                  <div key={lvl} className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ background: RISK_COLORS[lvl] }} />{lvl}
                    </span>
                    <span className="text-slate-400">{counts[lvl] || 0} cells</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="border-t border-slate-800 pt-4">
            <h2 className="text-sm font-medium mb-2">Click map to query a point</h2>
            {selected ? (
              <div className="text-xs bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-1">
                <div className="font-semibold" style={{ color: RISK_COLORS[selected.risk_level] }}>
                  {selected.risk_level} RISK — {(selected.risk_score * 100).toFixed(1)}%
                </div>
                <div className="text-slate-400">{selected.lat.toFixed(4)}, {selected.lon.toFixed(4)}</div>
                <div className="text-slate-500 mt-1">elevation ≈ {selected.features.elevation_m}m · dist to nearest known flood zone: {selected.features.dist_to_known_flood_zone_km}km</div>
              </div>
            ) : <p className="text-xs text-slate-500">No point selected yet.</p>}
          </div>

          <div className="border-t border-slate-800 pt-4">
            <h2 className="text-sm font-medium mb-2">PyTorch satellite CNN demo</h2>
            <p className="text-[11px] text-slate-500 mb-2">Runs the Sentinel-1/2 flood-extent CNN on a synthetic patch (water-like vs land-like signature) — see README for wiring real imagery.</p>
            <div className="flex gap-2">
              <button onClick={() => runSatelliteDemo("water")} className="text-[11px] px-2 py-1 rounded-md bg-blue-900 hover:bg-blue-800 border border-blue-700">Test water-like patch</button>
              <button onClick={() => runSatelliteDemo("land")} className="text-[11px] px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700">Test land-like patch</button>
            </div>
            {satResult && (
              <div className="text-xs bg-slate-900 border border-slate-800 rounded-lg p-3 mt-2">
                <div className="font-semibold" style={{ color: satResult.is_flooded ? RISK_COLORS.HIGH : RISK_COLORS.LOW }}>
                  flood probability: {(satResult.flood_probability * 100).toFixed(1)}%
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-slate-800 pt-4">
            <h2 className="text-sm font-medium mb-2">Legend</h2>
            <div className="flex items-center gap-2 text-xs mb-1">
              <span className="w-3 h-3 rounded-full bg-blue-900 border-2 border-white inline-block" />
              Real named flood-susceptible zone (published research)
            </div>
          </div>
        </aside>

        <main className="flex-1 relative min-h-[300px]">
          <MapContainer center={[27.70, 85.32]} zoom={12} style={{ height: "100%", width: "100%" }}>
            <ClickCatcher onClick={handleMapClick} />
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
            {grid && grid.features.map((f, i) => {
              const [lon, lat] = f.geometry.coordinates;
              const p = f.properties;
              if (p.risk_level === "LOW") return null;
              return (
                <CircleMarker key={i} center={[lat, lon]} radius={p.risk_level === "HIGH" ? 9 : 6}
                  pathOptions={{ color: RISK_COLORS[p.risk_level], fillColor: RISK_COLORS[p.risk_level], fillOpacity: 0.35, weight: 0 }} />
              );
            })}
            {zones && zones.features.map((f, i) => {
              const [lon, lat] = f.geometry.coordinates;
              return (
                <Marker key={i} position={[lat, lon]} icon={zoneIcon}>
                  <Popup><b>{f.properties.name}</b><br />{f.properties.note}<br />
                    <span className="text-xs text-slate-500">Source: {f.properties.source}</span></Popup>
                  <Tooltip>{f.properties.name}</Tooltip>
                </Marker>
              );
            })}
          </MapContainer>
        </main>
      </div>
    </div>
  );
}
