import os
import math
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# PathSnap | Laptop 2 evaluator-facing frontend
# API contract: GET /health, GET /trips, POST /predict {"trip_id": "..."}
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="PathSnap | Route Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --bg:#F3EFE5; --panel:#FBF8F0; --panel2:#E9E2D4; --line:#D7CDBB;
      --muted:#806F60; --text:#35251E; --sage:#729458; --walnut:#4F3328;
      --amber:#C1681A; --cream:#F8F4EA;
    }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
    .stApp { background:radial-gradient(circle at 82% -12%, #E4E8D8 0, #F3EFE5 42%, #F3EFE5 100%); color:var(--text); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#E8E2D5; border-right:1px solid #D4C9B8; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
    .block-container { padding-top:1.6rem; padding-bottom:2.5rem; max-width:1600px; }
    h1,h2,h3 { color:#4F3328 !important; letter-spacing:-.035em; }
    p, label, .stMarkdown { color:#554438; }
    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
    .brand-mark { width:42px;height:42px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#729458,#8DA66F);color:#FFF9EF;font-size:22px;font-weight:900;box-shadow:0 8px 24px #72945835; }
    .brand-name { font-size:25px;font-weight:800;letter-spacing:-1.2px;color:#4F3328; }
    .brand-sub { color:#806F60;font-size:10px;letter-spacing:2.1px;text-transform:uppercase;font-weight:700; }
    .eyebrow { color:#729458;font-size:10px;letter-spacing:2px;text-transform:uppercase;font-weight:800;margin-bottom:7px; }
    .page-title { font-size:30px;font-weight:800;letter-spacing:-1.4px;color:#4F3328;line-height:1.15; }
    .page-sub { color:#806F60;font-size:13px;margin-top:7px; }
    .status-pill { display:inline-flex;align-items:center;gap:8px;border:1px solid #B7C7A5;background:#E5EBDD;color:#4F6D3A;border-radius:30px;padding:8px 13px;font-size:11px;font-weight:800;letter-spacing:.5px; }
    .status-dot { width:7px;height:7px;border-radius:50%;background:#729458;box-shadow:0 0 10px #72945880; }
    .status-off { border-color:#D9B39A;background:#F5E7DC;color:#9A4D24; }
    .metric-card { background:linear-gradient(145deg,#FFFCF5,#EEE8DC);border:1px solid #D7CDBB;border-radius:16px;padding:17px 18px;min-height:112px;box-shadow:0 10px 26px #4F33280D; }
    .metric-label { color:#806F60;font-size:10px;letter-spacing:1.35px;text-transform:uppercase;font-weight:800; }
    .metric-value { color:#4F3328;font-size:27px;font-weight:800;letter-spacing:-1px;margin-top:10px;line-height:1.05; }
    .metric-note { color:#978675;font-size:10px;margin-top:7px; }
    .panel { background:linear-gradient(145deg,#FBF8F0,#F0EADD);border:1px solid #D7CDBB;border-radius:18px;padding:18px 19px; }
    .panel-heading { font-size:14px;font-weight:800;color:#4F3328;letter-spacing:-.2px; }
    .panel-caption { font-size:11px;color:#806F60;margin-top:4px; }
    .map-shell { border:1px solid #D7CDBB;border-radius:18px;overflow:hidden;background:#E8E2D5; }
    .legend { display:flex;gap:18px;align-items:center;flex-wrap:wrap;color:#806F60;font-size:11px;margin-top:12px; }
    .legend-item { display:flex;align-items:center;gap:7px; }
    .legend-dot { width:9px;height:9px;border-radius:50%;display:inline-block; }
    .legend-line { width:18px;height:3px;border-radius:5px;display:inline-block; }
    div.stButton > button { border-radius:11px;border:1px solid #BDAE9B;background:#E9E2D5;color:#4F3328;font-weight:800;min-height:42px;transition:all .15s ease; }
    div.stButton > button:hover { border-color:#729458;color:#4F6D3A;background:#E3EAD9; }
    div.stButton > button[kind="primary"] { background:linear-gradient(135deg,#C1681A,#D47A27);border:0;color:#FFF9EF;box-shadow:0 8px 22px #C1681A35; }
    div.stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#D47A27,#B85D16);color:#FFF9EF; }
    .stSelectbox [data-baseweb="select"] > div, .stTextInput [data-baseweb="input"] { background:#FBF8F0;border-color:#CFC3B1;border-radius:10px;color:#4F3328; }
    [data-testid="stMetric"] { background:transparent; }
    [data-testid="stAlert"] { border-radius:12px; }
    .small-muted { color:#806F60;font-size:11px; }
    .summary-grid { display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px; }
    .summary-cell { background:#F5F0E6;border:1px solid #D7CDBB;border-radius:12px;padding:11px 12px; }
    .summary-k { color:#8A7867;font-size:9px;text-transform:uppercase;letter-spacing:1.2px;font-weight:800; }
    .summary-v { color:#4F3328;font-size:13px;font-weight:700;margin-top:5px;overflow-wrap:anywhere; }
    hr { border-color:#D7CDBB !important; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(base: str, path: str, timeout: int = 8) -> requests.Response:
    return requests.get(f"{base.rstrip('/')}{path}", timeout=timeout)


def parse_trip_ids(payload: Any) -> List[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("trip_ids"), list):
        raise ValueError("Malformed /trips response. Expected {\"trip_ids\": [...]}.")
    ids = []
    for item in payload["trip_ids"]:
        if item is None:
            continue
        # Keep IDs as strings so leading zeros are preserved.
        ids.append(str(item))
    return ids


def valid_pair(point: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        lat, lon = float(point[0]), float(point[1])
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return lat, lon
    except (TypeError, ValueError):
        return None


def normalize_prediction(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Prediction response must be a JSON object.")
    if "trip_id" not in data or "gps_points" not in data or "predicted_route" not in data or "metrics" not in data:
        raise ValueError("Prediction response is missing one or more required fields: trip_id, gps_points, predicted_route, metrics.")
    gps = [p for raw in (data.get("gps_points") or []) if (p := valid_pair(raw)) is not None]
    segments = []
    raw_routes = data.get("predicted_route") or []
    if not isinstance(raw_routes, list):
        raise ValueError("predicted_route must be a list of road-segment polylines.")
    for segment in raw_routes:
        if not isinstance(segment, list):
            continue
        clean = [p for raw in segment if (p := valid_pair(raw)) is not None]
        if clean:
            segments.append(clean)
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object.")
    return {**data, "gps_points": gps, "predicted_route": segments, "metrics": metrics}


def metric_value(value: Any, kind: str) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "N/A"
        if kind == "fraction":
            return f"{number * 100:.1f}%"
        if kind == "seconds":
            return f"{number:.3f} s"
        return str(value)
    except (TypeError, ValueError):
        return "N/A"


def make_map(result: Dict[str, Any], show_gps: bool, show_route: bool, offline: bool, segment_limit: Optional[int] = None) -> folium.Map:
    gps = result["gps_points"]
    segments = result["predicted_route"]
    all_coords = list(gps)
    for segment in segments:
        all_coords.extend(segment)
    center = [sum(p[0] for p in all_coords) / len(all_coords), sum(p[1] for p in all_coords) / len(all_coords)] if all_coords else [20.5937, 78.9629]
    fmap = folium.Map(location=center, zoom_start=14 if all_coords else 5, tiles=None if offline else "OpenStreetMap", control_scale=True)
    if offline:
        folium.Rectangle(bounds=[[-90, -180], [90, 180]], color="#D7CDBB", weight=0, fill=True, fill_color="#E8E2D5", fill_opacity=1, tooltip="Offline map mode · route overlays remain available").add_to(fmap)
    if show_gps and gps:
        gps_group = folium.FeatureGroup(name="GPS observations", show=True)
        for i, point in enumerate(gps):
            folium.CircleMarker(
                location=point, radius=4 if len(gps) > 100 else 5,
                color="#729458", weight=1, fill=True, fill_color="#729458", fill_opacity=.9,
                tooltip=f"GPS point {i + 1}: {point[0]:.6f}, {point[1]:.6f}",
            ).add_to(gps_group)
        gps_group.add_to(fmap)
    if show_route and segments:
        route_group = folium.FeatureGroup(name="Predicted road segments", show=True)
        visible_segments = segments if segment_limit is None else segments[:segment_limit]
        for idx, segment in enumerate(visible_segments):
            if len(segment) == 1:
                folium.CircleMarker(location=segment[0], radius=5, color="#C1681A", fill=True, fill_color="#C1681A", tooltip=f"Predicted segment {idx + 1}").add_to(route_group)
            else:
                # Each returned segment is drawn independently; never join disconnected segments.
                folium.PolyLine(segment, color="#C1681A", weight=5, opacity=.92, tooltip=f"Predicted road segment {idx + 1}").add_to(route_group)
        route_group.add_to(fmap)
    if all_coords:
        fmap.fit_bounds([[min(p[0] for p in all_coords), min(p[1] for p in all_coords)], [max(p[0] for p in all_coords), max(p[1] for p in all_coords)]], padding=(22, 22))
    folium.LayerControl(collapsed=True).add_to(fmap)
    Fullscreen(position="topright", title="Expand map", title_cancel="Exit fullscreen").add_to(fmap)
    return fmap


# Sidebar: connection and trip selection
with st.sidebar:
    st.markdown('<div class="brand-row"><div class="brand-mark">⌁</div><div><div class="brand-name">PathSnap</div><div class="brand-sub">Route intelligence</div></div></div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Connection</div>', unsafe_allow_html=True)
    api_url = st.text_input("Backend API URL", value=st.session_state.get("api_url", DEFAULT_API_URL), help="Laptop 1 FastAPI address, for example http://192.168.1.5:8000")
    st.session_state.api_url = api_url.rstrip("/")
    c1, c2 = st.columns(2)
    with c1:
        check_clicked = st.button("Check API", use_container_width=True)
    with c2:
        refresh_clicked = st.button("Refresh trips", use_container_width=True)

    health_ok = False
    health_message = "Not checked"
    try:
        health_response = api_get(st.session_state.api_url, "/health", timeout=4)
        health_response.raise_for_status()
        health_json = health_response.json()
        health_ok = isinstance(health_json, dict) and health_json.get("status") == "ok"
        health_message = "Connected" if health_ok else "Unexpected health response"
    except requests.exceptions.RequestException:
        health_message = "Backend unreachable"
    except ValueError:
        health_message = "Invalid health response"

    if check_clicked or refresh_clicked:
        if health_ok:
            st.success("API connection verified")
        else:
            st.error(f"{health_message}. Check the server, IP address and network.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if health_ok:
        st.markdown('<div class="status-pill"><span class="status-dot"></span> API CONNECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-off"><span class="status-dot" style="background:#ff987e;box-shadow:none"></span> API DISCONNECTED</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Evaluation trip</div>', unsafe_allow_html=True)
    if "trip_ids" not in st.session_state or refresh_clicked:
        try:
            trips_response = api_get(st.session_state.api_url, "/trips", timeout=8)
            trips_response.raise_for_status()
            st.session_state.trip_ids = parse_trip_ids(trips_response.json())
            st.session_state.trip_load_error = None
        except requests.exceptions.Timeout:
            st.session_state.trip_ids = []
            st.session_state.trip_load_error = "Loading trips timed out. Retry when the backend is ready."
        except requests.exceptions.RequestException as exc:
            st.session_state.trip_ids = []
            st.session_state.trip_load_error = f"Could not load trips: {exc}"
        except (ValueError, TypeError) as exc:
            st.session_state.trip_ids = []
            st.session_state.trip_load_error = str(exc)

    trip_ids = st.session_state.get("trip_ids", [])
    if trip_ids:
        selected_trip = st.selectbox("Select trip ID", trip_ids, key="selected_trip", label_visibility="collapsed")
    else:
        selected_trip = None
        st.caption("No evaluation trips loaded.")
        if st.session_state.get("trip_load_error"):
            st.error(st.session_state.trip_load_error)
    predict_clicked = st.button("↗  Predict route", type="primary", use_container_width=True, disabled=not bool(selected_trip))
    if st.session_state.pop("retry_now", False):
        predict_clicked = bool(selected_trip)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
    model_name = st.session_state.get("prediction", {}).get("model", "Awaiting result")
    st.markdown(f"<div style='font-size:15px;font-weight:800;color:#4F3328'>{model_name}</div><div class='small-muted' style='margin-top:5px'>Selected inference model</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div><div class='small-muted'>PATHSNAP · LAPTOP 2 FRONTEND</div>", unsafe_allow_html=True)

# Header
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown('<div class="eyebrow">Evaluation workspace / Route analysis</div><div class="page-title">Route intelligence</div><div class="page-sub">Inspect GPS observations, predicted road segments and live model performance.</div>', unsafe_allow_html=True)
with head_r:
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    if health_ok:
        st.markdown('<div style="text-align:right"><span class="status-pill"><span class="status-dot"></span> LIVE CONNECTION</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right"><span class="status-pill status-off"><span class="status-dot" style="background:#ff987e;box-shadow:none"></span> CONNECTION ISSUE</span></div>', unsafe_allow_html=True)

# Prediction action
if predict_clicked and selected_trip:
    with st.spinner(f"Running route inference for trip {selected_trip}…"):
        try:
            response = requests.post(
                f"{st.session_state.api_url}/predict",
                json={"trip_id": str(selected_trip)},
                timeout=120,
            )
            response.raise_for_status()
            prediction = normalize_prediction(response.json())
            st.session_state.prediction = prediction
            st.session_state.prediction_error = None
            st.session_state.active_trip = str(selected_trip)
            st.session_state.segment_replay = 0
        except requests.exceptions.Timeout:
            st.session_state.prediction_error = "The prediction request timed out after 120 seconds. Check the backend logs and retry."
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            detail = ""
            try:
                detail = exc.response.text[:300] if exc.response is not None else ""
            except Exception:
                pass
            st.session_state.prediction_error = f"Prediction failed (HTTP {status}). {detail or 'Check that the selected trip ID is supported.'}"
        except requests.exceptions.RequestException as exc:
            st.session_state.prediction_error = f"Cannot reach the prediction service: {exc}. Verify the backend is running and both laptops are on the same network."
        except (ValueError, TypeError) as exc:
            st.session_state.prediction_error = f"The backend returned a malformed response: {exc}"

if st.session_state.get("prediction_error"):
    st.error(st.session_state.prediction_error)
    if st.button("Retry prediction", key="retry_prediction") and selected_trip:
        st.session_state.retry_now = True
        st.rerun()

prediction = st.session_state.get("prediction")
if prediction:
    metrics = prediction["metrics"]
    st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
    metric_cols = st.columns(4, gap="medium")
    metric_specs = [
        ("TOP-1 ACCURACY", metric_value(metrics.get("top1_accuracy"), "fraction"), "Model prediction accuracy"),
        ("MACRO F1", metric_value(metrics.get("macro_f1"), "fraction"), "Balanced class performance"),
        ("ROUTE CONTINUITY", metric_value(metrics.get("route_continuity"), "fraction"), "Decoded route consistency"),
        ("INFERENCE TIME", metric_value(metrics.get("inference_seconds"), "seconds"), "End-to-end prediction time"),
    ]
    for col, (label, value, note) in zip(metric_cols, metric_specs):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)
    metric_cols = st.columns(4, gap="medium")
    for col, label, note in zip(metric_cols, ["TOP-1 ACCURACY", "MACRO F1", "ROUTE CONTINUITY", "INFERENCE TIME"], ["Model prediction accuracy", "Balanced class performance", "Decoded route consistency", "End-to-end prediction time"]):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='color:#71899e'>N/A</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
map_col, detail_col = st.columns([2.25, 1], gap="large")
with map_col:
    st.markdown('<div class="panel-heading">Predicted route map</div><div class="panel-caption">Geospatial view · GPS observations and matched road segments</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    control_cols = st.columns([1, 1, 1, 1.35])
    with control_cols[0]:
        show_gps = st.toggle("GPS points", value=True, key="show_gps")
    with control_cols[1]:
        show_route = st.toggle("Predicted route", value=True, key="show_route")
    with control_cols[2]:
        offline_mode = st.toggle("Offline map", value=False, key="offline_mode", help="Hides online map tiles; route overlays remain visible.")
    with control_cols[3]:
        if st.button("⌖  Fit to route", use_container_width=True, disabled=not bool(prediction)):
            # The map is fitted to all returned coordinates on every render.
            st.rerun()
    if prediction:
        replay_limit = None
        if prediction["predicted_route"]:
            max_segments = len(prediction["predicted_route"])
            if max_segments > 1:
                replay_index = st.slider("Route replay · visible segments", min_value=1, max_value=max_segments, value=max_segments, key="segment_replay")
                replay_limit = replay_index
        fmap = make_map(prediction, show_gps, show_route, offline_mode, replay_limit)
        st.markdown('<div class="map-shell">', unsafe_allow_html=True)
        st_folium(fmap, height=490, use_container_width=True, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="legend"><span class="legend-item"><span class="legend-dot" style="background:#729458"></span> GPS observations</span><span class="legend-item"><span class="legend-line" style="background:#C1681A"></span> Predicted road segments</span></div>', unsafe_allow_html=True)
        if not prediction["gps_points"] and not prediction["predicted_route"]:
            st.info("The API returned no valid coordinates for this trip. Metrics are shown where available.")
    else:
        st.markdown("<div class='map-shell' style='height:490px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0b1c2b,#0a1725)'><div style='text-align:center;max-width:380px;padding:25px'><div style='font-size:42px;margin-bottom:14px'>⌖</div><div style='font-size:17px;font-weight:800;color:#4F3328'>Your route will appear here</div><div style='font-size:12px;color:#806F60;margin-top:8px;line-height:1.7'>Select an evaluation trip and run Predict Route to visualize GPS observations and the model's predicted road segments.</div></div></div>", unsafe_allow_html=True)
        st.markdown('<div class="legend"><span class="legend-item"><span class="legend-dot" style="background:#729458"></span> GPS observations</span><span class="legend-item"><span class="legend-line" style="background:#C1681A"></span> Predicted road segments</span></div>', unsafe_allow_html=True)

with detail_col:
    st.markdown('<div class="panel"><div class="panel-heading">Trip overview</div><div class="panel-caption">Current evaluation result</div>', unsafe_allow_html=True)
    if prediction:
        gps_count = len(prediction["gps_points"])
        segment_count = len(prediction["predicted_route"])
        coordinate_count = sum(len(s) for s in prediction["predicted_route"])
        st.markdown(f"""
        <div class="summary-grid">
          <div class="summary-cell"><div class="summary-k">Trip ID</div><div class="summary-v">{prediction.get('trip_id', 'N/A')}</div></div>
          <div class="summary-cell"><div class="summary-k">Model</div><div class="summary-v">{prediction.get('model', 'N/A')}</div></div>
          <div class="summary-cell"><div class="summary-k">GPS points</div><div class="summary-v">{gps_count:,}</div></div>
          <div class="summary-cell"><div class="summary-k">Road segments</div><div class="summary-v">{segment_count:,}</div></div>
        </div>
        <div style="margin-top:15px;color:#806F60;font-size:11px;line-height:1.7">{coordinate_count:,} coordinates across returned road segments. Each segment is rendered independently to preserve gaps in the predicted route.</div>
        """, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Inference status</div>", unsafe_allow_html=True)
        st.markdown('<div style="color:#729458;font-weight:800;font-size:13px">● Prediction received</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted' style='margin-top:6px'>Result loaded for trip {prediction.get('trip_id', 'N/A')}.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding:28px 4px;color:#806F60;font-size:12px;line-height:1.8'>No trip has been evaluated yet. Once a prediction is received, this panel will summarize the selected trip, model and route geometry.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
st.markdown("<div style='border-top:1px solid #D7CDBB;padding-top:15px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#978675;font-size:10px;letter-spacing:.7px'><span>PATHSNAP · ROUTE INTELLIGENCE</span><span>LAPTOP 2 EVALUATOR INTERFACE · LIVE API DATA</span></div>", unsafe_allow_html=True)
