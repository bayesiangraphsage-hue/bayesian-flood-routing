import re
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import networkx as nx
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
from scipy.spatial import cKDTree

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Flood Route Planner",
    layout="wide",
    page_icon="🌊",
)

# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    h1, h2, h3 {
        color: #0f172a;
        font-weight: 650;
        letter-spacing: -0.02em;
    }

    .subtle {
        color: #64748b;
        font-size: 0.98rem;
        margin-top: -0.35rem;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .hint {
        background: #eff6ff;
        border: 1px solid #dbeafe;
        color: #1d4ed8;
        border-radius: 12px;
        padding: 10px 12px;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .badge {
        display: inline-block;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #0f172a;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        background: #ffffff;
        color: #0f172a;
        padding: 0.55rem 0.9rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #94a3b8;
        background: #f8fafc;
    }

    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TITLE
# =========================
st.title("Flood-Route Planner using Bayesian GraphSAGE-GRU and Dijkstra's Algorithm")
st.markdown(
    "<div class='subtle'>Dynamic flooding, risk-aware routing, and travel-time estimation on an interactive map.</div>",
    unsafe_allow_html=True,
)

# =========================
# CONSTANTS
# =========================
NODES_PATH = "./processed_nodes.gpkg"
EDGES_PATH = "./prediction_test_sequence_0.gpkg"
ASSUME_BIDIRECTIONAL_ROADS = True

# =========================
# HELPERS
# =========================
def to_native(x):
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return x
    return x


def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def flood_multiplier(time_step: int, total_steps: int) -> float:
    """
    Synthetic flood evolution used when the dataset does not contain time-indexed flood columns.
    Starts low, rises to a peak near the middle, then recedes.
    """
    if total_steps <= 1:
        return 1.0
    x = time_step / (total_steps - 1)
    # Low at ends, peak in the middle
    return float(0.25 + 0.95 * np.exp(-((x - 0.58) / 0.20) ** 2))


def get_time_column_value(row: pd.Series, base_name: str, time_step: int):
    """
    Tries to find time-indexed columns like:
    pred_flood_penalty_t0, pred_flood_penalty_0, pred_flood_penalty-t0, pred_flood_penaltyt0
    Returns (value, found_flag).
    """
    candidates = [
        f"{base_name}_t{time_step}",
        f"{base_name}_{time_step}",
        f"{base_name}-t{time_step}",
        f"{base_name}t{time_step}",
        f"{base_name}T{time_step}",
        f"{base_name}_T{time_step}",
    ]
    for c in candidates:
        if c in row.index:
            val = row.get(c, None)
            if pd.notna(val):
                return safe_float(val, 0.0), True
    return None, False


def best_edge_data(G, u, v, weight_key="planned_cost"):
    """
    Returns the best edge data between u and v for a MultiDiGraph.
    """
    data = G.get_edge_data(u, v)
    if not data:
        return None

    # MultiDiGraph => dict of keys -> attr dict
    if isinstance(data, dict):
        return min(
            data.values(),
            key=lambda d: safe_float(d.get(weight_key, np.inf), np.inf),
        )
    return data


def build_route_coords(route_nodes, node_xy_by_osmid):
    coords = []
    for n in route_nodes:
        if n in node_xy_by_osmid:
            lon, lat = node_xy_by_osmid[n]
            coords.append((lat, lon))
    return coords


# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    nodes = gpd.read_file(NODES_PATH)
    edges = gpd.read_file(EDGES_PATH)

    nodes_wgs = nodes.to_crs("EPSG:4326")
    edges_wgs = edges.to_crs("EPSG:4326")

    return nodes, edges, nodes_wgs, edges_wgs


try:
    nodes, edges, nodes_wgs, edges_wgs = load_data()
except Exception as e:
    st.error(f"Failed to load data files. Check paths and file contents.\n\nDetails: {e}")
    st.stop()

# =========================
# TIME SLIDER
# =========================
st.sidebar.header("Model Controls")
ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0, 0.1)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0, 0.1)
SHOW_FLOOD = st.sidebar.checkbox("Show Flood Hazard Layer", True)

TIME_STEPS = st.sidebar.slider(
    "Dynamic Flood Timeline Length",
    min_value=6,
    max_value=24,
    value=12,
    step=1,
    help="Used when your dataset does not include time-indexed flood columns.",
)

time_step = st.sidebar.slider(
    "Flood Time Slider",
    min_value=0,
    max_value=TIME_STEPS - 1,
    value=0,
    step=1,
)

phase_name = (
    "Dry"
    if time_step <= max(1, TIME_STEPS // 4)
    else "Rising"
    if time_step <= max(2, TIME_STEPS // 2)
    else "Peak"
    if time_step <= max(3, (3 * TIME_STEPS) // 4)
    else "Recession"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current phase:** {phase_name}")
st.sidebar.markdown(f"**Flood multiplier:** {flood_multiplier(time_step, TIME_STEPS):.2f}")

# =========================
# BUILD GRAPH
# =========================
@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_graph(edges_gdf, alpha, lam, time_step, total_steps):
    G = nx.MultiDiGraph()

    for _, row in edges_gdf.iterrows():
        try:
            u = to_native(row["u"])
            v = to_native(row["v"])

            # Base travel time
            speed = safe_float(row.get("speed_mps", 25 * 1000 / 3600), 25 * 1000 / 3600)
            length = safe_float(row.get("length", 0.0), 0.0)
            base_time = safe_float(row.get("travel_time", length / speed if speed > 0 else 0.0), 0.0)

            # Dynamic flood penalty
            penalty_t, found_penalty_t = get_time_column_value(row, "pred_flood_penalty", time_step)
            unc_t, found_unc_t = get_time_column_value(row, "uncertainty", time_step)

            if found_penalty_t:
                base_penalty = penalty_t
            else:
                base_penalty = safe_float(row.get("pred_flood_penalty", 0.0), 0.0) * flood_multiplier(time_step, total_steps)

            if found_unc_t:
                uncertainty = unc_t
            else:
                uncertainty = safe_float(row.get("uncertainty", 0.0), 0.0)

            risk = np.clip(base_penalty + lam * uncertainty, 0.0, 1.0)
            planned_cost = base_time * (1.0 + alpha * risk)

            edge_attrs = {
                "edge_id": to_native(row.get("edge_id", 0)),
                "length": length,
                "travel_time": base_time,
                "risk": float(risk),
                "planned_cost": float(planned_cost),
            }

            G.add_edge(u, v, **edge_attrs)
            if ASSUME_BIDIRECTIONAL_ROADS:
                G.add_edge(v, u, **edge_attrs)

        except Exception:
            continue

    return G


G = build_graph(edges, ROUTING_ALPHA, BAYES_LAMBDA, time_step, TIME_STEPS)

# =========================
# SPATIAL INDEX
# =========================
@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_tree(nodes_gdf):
    coords = np.array(list(zip(nodes_gdf.geometry.x, nodes_gdf.geometry.y)))
    tree = cKDTree(coords)
    osmids = nodes_gdf["osmid"].values
    node_xy_by_osmid = dict(zip(osmids.tolist(), coords.tolist()))
    return tree, osmids, coords, node_xy_by_osmid


tree, osmids, coords, node_xy_by_osmid = build_tree(nodes_wgs)

# =========================
# SESSION STATE
# =========================
if "origin" not in st.session_state:
    st.session_state.origin = None
if "destination" not in st.session_state:
    st.session_state.destination = None
if "origin_coords" not in st.session_state:
    st.session_state.origin_coords = None
if "destination_coords" not in st.session_state:
    st.session_state.destination_coords = None
if "active" not in st.session_state:
    st.session_state.active = "origin"
if "route_time" not in st.session_state:
    st.session_state.route_time = None
if "route_risk" not in st.session_state:
    st.session_state.route_risk = None

def reset_route():
    st.session_state.origin = None
    st.session_state.destination = None
    st.session_state.origin_coords = None
    st.session_state.destination_coords = None
    st.session_state.route_time = None
    st.session_state.route_risk = None
    st.session_state.active = "origin"

# =========================
# TOP STATUS BAR
# =========================
top1, top2, top3, top4 = st.columns([1.2, 1.2, 1.2, 1.0])
with top1:
    st.metric("Nodes", f"{len(nodes):,}")
with top2:
    st.metric("Edges", f"{len(edges):,}")
with top3:
    st.metric("Graph Density", f"{nx.density(G):.4f}")
with top4:
    st.metric("Time Step", f"{time_step}/{TIME_STEPS - 1}")

st.markdown("---")

# =========================
# LAYOUT
# =========================
left, right = st.columns([3.2, 1.0], gap="large")

# =========================
# MAP
# =========================
with left:
    st.markdown("### Interactive Map")
    st.markdown(
        f"<div class='hint'>Select <b>Origin</b> or <b>Destination</b>, then click on the map. "
        f"The flood hazard layer updates with the time slider.</div>",
        unsafe_allow_html=True,
    )

    center_lat = float(nodes_wgs.geometry.y.mean())
    center_lon = float(nodes_wgs.geometry.x.mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer("CartoDB positron", name="Light", control=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street", control=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    # =========================
    # FLOOD HAZARD LAYER
    # =========================
    if SHOW_FLOOD:
        for _, row in edges_wgs.iterrows():
            try:
                geom = row.geometry
                if geom is None:
                    continue

                # Use time-indexed hazard if available, otherwise synthesize one
                penalty_t, found_penalty_t = get_time_column_value(row, "pred_flood_penalty", time_step)
                unc_t, found_unc_t = get_time_column_value(row, "uncertainty", time_step)

                if found_penalty_t:
                    base_penalty = penalty_t
                else:
                    base_penalty = safe_float(row.get("pred_flood_penalty", 0.0), 0.0) * flood_multiplier(time_step, TIME_STEPS)

                if found_unc_t:
                    uncertainty = unc_t
                else:
                    uncertainty = safe_float(row.get("uncertainty", 0.0), 0.0)

                risk = float(np.clip(base_penalty + BAYES_LAMBDA * uncertainty, 0.0, 1.0))

                if risk < 0.33:
                    color = "#2563eb"  # blue
                elif risk < 0.66:
                    color = "#f59e0b"  # amber
                else:
                    color = "#dc2626"  # red

                folium.GeoJson(
                    geom,
                    name="Flood Hazard",
                    style_function=lambda x, col=color: {
                        "color": col,
                        "weight": 3,
                        "opacity": 0.65,
                    },
                ).add_to(m)

            except Exception:
                continue

    # =========================
    # MARKERS
    # =========================
    if st.session_state.origin_coords:
        folium.CircleMarker(
            st.session_state.origin_coords,
            radius=8,
            color="#16a34a",
            fill=True,
            fill_color="#16a34a",
            fill_opacity=1.0,
            tooltip="Origin",
        ).add_to(m)

    if st.session_state.destination_coords:
        folium.CircleMarker(
            st.session_state.destination_coords,
            radius=8,
            color="#dc2626",
            fill=True,
            fill_color="#dc2626",
            fill_opacity=1.0,
            tooltip="Destination",
        ).add_to(m)

    # =========================
    # ROUTE
    # =========================
    route_computed = False

    if st.session_state.origin is not None and st.session_state.destination is not None:
        try:
            route = nx.shortest_path(
                G,
                source=st.session_state.origin,
                target=st.session_state.destination,
                weight="planned_cost",
            )

            route_coords = []
            total_time = 0.0
            total_risk = 0.0
            edge_count = 0

            for i in range(len(route) - 1):
                u = route[i]
                v = route[i + 1]
                edge_data = best_edge_data(G, u, v, "planned_cost")

                if edge_data is None:
                    continue

                total_time += safe_float(edge_data.get("travel_time", 0.0), 0.0)
                total_risk += safe_float(edge_data.get("risk", 0.0), 0.0)
                edge_count += 1

            route_coords = build_route_coords(route, node_xy_by_osmid)

            if route_coords:
                folium.PolyLine(
                    route_coords,
                    color="#0f766e",
                    weight=6,
                    opacity=0.9,
                    tooltip="Optimized route",
                ).add_to(m)

            if edge_count > 0:
                st.session_state.route_time = total_time
                st.session_state.route_risk = total_risk / edge_count
            else:
                st.session_state.route_time = None
                st.session_state.route_risk = None

            route_computed = True

        except nx.NetworkXNoPath:
            st.session_state.route_time = None
            st.session_state.route_risk = None
            st.warning("No route found between the selected points.")
        except Exception as e:
            st.session_state.route_time = None
            st.session_state.route_risk = None
            st.error(f"Routing error: {e}")

    map_data = st_folium(
        m,
        use_container_width=True,
        height=650,
        returned_objects=["last_clicked"],
    )

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        _, idx = tree.query([lon, lat])
        nearest_node = to_native(osmids[idx])
        nearest_coord = (float(coords[idx][1]), float(coords[idx][0]))

        if st.session_state.active == "origin":
            st.session_state.origin = nearest_node
            st.session_state.origin_coords = nearest_coord
        else:
            st.session_state.destination = nearest_node
            st.session_state.destination_coords = nearest_coord

        st.rerun()

# =========================
# RIGHT PANEL
# =========================
with right:
    st.markdown("### Controls")

    st.radio(
        "Active Marker",
        ["origin", "destination"],
        key="active",
        format_func=lambda x: "Origin" if x == "origin" else "Destination",
        horizontal=False,
    )

    st.button("Reset Route", on_click=reset_route, use_container_width=True)

    st.markdown("---")
    st.markdown("### Route Summary")

    if st.session_state.route_time is not None and st.session_state.route_risk is not None:
        st.metric("Travel Time", f"{st.session_state.route_time / 60:.2f} min")
        st.metric("Average Flood Risk", f"{st.session_state.route_risk:.2f}")
    else:
        st.info("Select an origin and destination to compute a route.")

    st.markdown("---")
    st.markdown("### Current Selection")

    st.write("Origin:", st.session_state.origin if st.session_state.origin is not None else "Not set")
    st.write("Destination:", st.session_state.destination if st.session_state.destination is not None else "Not set")

    st.markdown("---")
    st.markdown("### Legend")
    st.markdown(
        """
        <div class='badge' style='color:#16a34a;'>Origin</div>
        <div class='badge' style='color:#dc2626;'>Destination</div>
        <div class='badge' style='color:#2563eb;'>Low flood risk</div>
        <div class='badge' style='color:#f59e0b;'>Moderate risk</div>
        <div class='badge' style='color:#dc2626;'>High risk</div>
        """,
        unsafe_allow_html=True,
    )
