import streamlit as st
import geopandas as gpd
import networkx as nx
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
from scipy.spatial import cKDTree
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Flood Routing", layout="wide")

# =========================
# CLEAN UI STYLING
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    color: #111827;
    font-weight: 600;
}

/* Subtitle */
.subtitle {
    color: #6b7280;
    font-size: 15px;
}

/* Card */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Buttons */
.stButton button {
    border-radius: 8px;
    background-color: #2563eb;
    color: white;
    border: none;
}

/* Info box */
.info-box {
    background: #eff6ff;
    padding: 10px;
    border-radius: 8px;
    color: #1e40af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1> Flood-Route Planner using Bayesian GraphSAGE-GRU and Dijkstra's Algorithm </h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart routing using flood prediction and uncertainty modeling</p>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Model Settings")

ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.caption("Higher α avoids flooded roads more aggressively")

# =========================
# DATA
# =========================
@st.cache_data
def load_data():
    nodes = gpd.read_file("./processed_nodes.gpkg")
    edges = gpd.read_file("./prediction_test_sequence_0.gpkg")
    return nodes, edges, nodes.to_crs("EPSG:4326")

@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_graph(edges, alpha, lam):
    G = nx.MultiDiGraph()
    for _, row in edges.iterrows():
        try:
            u, v = row["u"], row["v"]
            speed = float(row.get("speed_mps", 25 * 1000 / 3600))
            base = float(row.get("travel_time", row["length"] / speed))

            penalty = float(row.get("pred_flood_penalty", 0))
            uncertainty = float(row.get("uncertainty", 0))

            risk = np.clip(penalty + lam * uncertainty, 0, 1)
            cost = base * (1 + alpha * risk)

            G.add_edge(u, v, planned_cost=cost)
            G.add_edge(v, u, planned_cost=cost)
        except:
            continue
    return G

@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_tree(nodes):
    coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
    return cKDTree(coords), nodes["osmid"].values, coords

nodes, edges, nodes_wgs = load_data()
G = build_graph(edges, ROUTING_ALPHA, BAYES_LAMBDA)
tree, osmids, coords = build_tree(nodes_wgs)

# =========================
# SESSION STATE
# =========================
if "origin" not in st.session_state:
    st.session_state.origin = None
    st.session_state.destination = None
    st.session_state.origin_coords = None
    st.session_state.destination_coords = None
    st.session_state.active = "origin"

def reset():
    st.session_state.origin = None
    st.session_state.destination = None
    st.session_state.origin_coords = None
    st.session_state.destination_coords = None

# =========================
# LAYOUT
# =========================
left, right = st.columns([3, 1])

# =========================
# RIGHT PANEL
# =========================
with right:
    st.markdown("### 📍 Selection")

    st.radio(
        "Active Marker",
        ["origin", "destination"],
        key="active",
        format_func=lambda x: "Origin (Green)" if x == "origin" else "Destination (Red)"
    )

    st.markdown("<div class='info-box'>Click anywhere on the map to place or move the selected marker.</div>", unsafe_allow_html=True)

    st.markdown("### Current Nodes")
    st.write("Origin:", st.session_state.origin or "Not set")
    st.write("Destination:", st.session_state.destination or "Not set")

    st.button("Reset Route", on_click=reset)

# =========================
# MAP
# =========================
with left:
    center = [nodes_wgs.geometry.y.mean(), nodes_wgs.geometry.x.mean()]

    m = folium.Map(location=center, zoom_start=13, control_scale=True)

    # basemaps
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    folium.LayerControl().add_to(m)
    MiniMap().add_to(m)

    # markers
    if st.session_state.origin_coords:
        folium.CircleMarker(
            st.session_state.origin_coords,
            radius=8,
            color="#16a34a",
            fill=True,
            fill_color="#16a34a",
            tooltip="Origin"
        ).add_to(m)

    if st.session_state.destination_coords:
        folium.CircleMarker(
            st.session_state.destination_coords,
            radius=8,
            color="#dc2626",
            fill=True,
            fill_color="#dc2626",
            tooltip="Destination"
        ).add_to(m)

    # route
    if st.session_state.origin and st.session_state.destination:
        try:
            route = nx.shortest_path(
                G,
                st.session_state.origin,
                st.session_state.destination,
                weight="planned_cost"
            )

            coords_route = []
            for n in route:
                idx = np.where(osmids == n)[0][0]
                lon, lat = coords[idx]
                coords_route.append((lat, lon))

            folium.PolyLine(
                coords_route,
                color="#2563eb",
                weight=5,
                opacity=0.8
            ).add_to(m)

        except:
            st.warning("No route found")

    map_data = st_folium(m, width=900, height=600)

# =========================
# CLICK HANDLER
# =========================
if map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    _, idx = tree.query([lon, lat])
    node = osmids[idx]
    coord = (coords[idx][1], coords[idx][0])

    if st.session_state.active == "origin":
        st.session_state.origin = node
        st.session_state.origin_coords = coord
    else:
        st.session_state.destination = node
        st.session_state.destination_coords = coord

    st.rerun()
