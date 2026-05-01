import streamlit as st
import geopandas as gpd
import networkx as nx
import folium
from streamlit_folium import st_folium
from scipy.spatial import cKDTree
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bayesian Flood Routing",
    layout="wide"
)

# =========================
# CLEAN CSS (minimal, professional)
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px;
}

.small-text {
    color: #6b7280;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("Bayesian Flood Routing")
st.caption("Risk-aware route planning with flood prediction and uncertainty")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Model Settings")

ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0)
ASSUME_BIDIRECTIONAL_ROADS = st.sidebar.checkbox("Bidirectional Roads", True)

# =========================
# FILE PATHS
# =========================
NODES_PATH = "./processed_nodes.gpkg"
EDGES_PATH = "./prediction_test_sequence_0.gpkg"

# =========================
# DATA
# =========================
@st.cache_data
def load_data():
    nodes = gpd.read_file(NODES_PATH)
    edges = gpd.read_file(EDGES_PATH)
    return nodes, edges, nodes.to_crs("EPSG:4326"), edges.to_crs("EPSG:4326")

@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_graph(edges, alpha, lam, bidirectional):
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

            if bidirectional:
                G.add_edge(v, u, planned_cost=cost)

        except:
            continue
    return G

@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_tree(nodes):
    coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
    return cKDTree(coords), nodes["osmid"].values, coords

# =========================
# LOAD
# =========================
nodes, edges, nodes_wgs, edges_wgs = load_data()
G = build_graph(edges, ROUTING_ALPHA, BAYES_LAMBDA, ASSUME_BIDIRECTIONAL_ROADS)
tree, osmids, coords = build_tree(nodes_wgs)

# =========================
# SESSION STATE
# =========================
if "origin" not in st.session_state:
    st.session_state.origin = None
    st.session_state.destination = None
    st.session_state.origin_coords = None
    st.session_state.destination_coords = None
    st.session_state.mode = None   # NEW: interaction mode

def reset():
    for k in list(st.session_state.keys()):
        st.session_state[k] = None

# =========================
# LAYOUT
# =========================
left, right = st.columns([3, 1])

# =========================
# RIGHT PANEL (CONTROLS)
# =========================
with right:
    st.markdown("### Selection")

    if st.button("📍 Select Origin"):
        st.session_state.mode = "origin"

    if st.button("🏁 Select Destination"):
        st.session_state.mode = "destination"

    st.markdown("---")

    st.markdown("### Current State")

    st.write("Mode:", st.session_state.mode or "None")
    st.write("Origin:", st.session_state.origin)
    st.write("Destination:", st.session_state.destination)

    st.markdown("---")

    st.button("Reset", on_click=reset)

# =========================
# MAP
# =========================
with left:
    center = [nodes_wgs.geometry.y.mean(), nodes_wgs.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=13)

    # markers
    if st.session_state.origin_coords:
        folium.Marker(
            st.session_state.origin_coords,
            tooltip="Origin",
            icon=folium.Icon(color="green")
        ).add_to(m)

    if st.session_state.destination_coords:
        folium.Marker(
            st.session_state.destination_coords,
            tooltip="Destination",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # route
    route_found = False

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
                weight=5
            ).add_to(m)

            route_found = True

        except:
            st.warning("No route found")

    # map
    map_data = st_folium(m, width=900, height=600)

    # instruction
    if st.session_state.mode:
        st.info(f"Click on the map to set **{st.session_state.mode}**")

# =========================
# CLICK HANDLER (IMPROVED UX)
# =========================
if map_data.get("last_clicked") and st.session_state.mode:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    _, idx = tree.query([lon, lat])
    node = osmids[idx]
    coord = (coords[idx][1], coords[idx][0])

    if st.session_state.mode == "origin":
        st.session_state.origin = node
        st.session_state.origin_coords = coord

    elif st.session_state.mode == "destination":
        st.session_state.destination = node
        st.session_state.destination_coords = coord

    st.session_state.mode = None
    st.rerun()
