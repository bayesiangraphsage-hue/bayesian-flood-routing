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
    layout="wide",
    page_icon="🌊"
)

# =========================
# CUSTOM CSS (KEY UPGRADE)
# =========================
st.markdown("""
<style>
/* Main background */
.main {
    background-color: #0e1117;
}

/* Titles */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

/* Card container */
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Metric styling */
[data-testid="metric-container"] {
    background-color: #161b22;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO HEADER
# =========================
st.markdown("""
<div style='padding:20px 0px'>
<h1>🌊 Bayesian Flood Routing System</h1>
<p style='color:gray;font-size:16px'>
Real-time intelligent routing with flood risk and uncertainty modeling
</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR (CONTROL PANEL)
# =========================
st.sidebar.markdown("## ⚙️ Control Panel")

ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0)
ASSUME_BIDIRECTIONAL_ROADS = st.sidebar.toggle("Bidirectional Roads", True)

st.sidebar.markdown("---")
st.sidebar.caption("Adjust model behavior dynamically")

# =========================
# FILE PATHS
# =========================
NODES_PATH = "./processed_nodes.gpkg"
EDGES_PATH = "./prediction_test_sequence_0.gpkg"

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    nodes = gpd.read_file(NODES_PATH)
    edges = gpd.read_file(EDGES_PATH)
    return nodes, edges, nodes.to_crs("EPSG:4326"), edges.to_crs("EPSG:4326")

@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_graph(edges_gdf, alpha, lam, bidirectional):
    G = nx.MultiDiGraph()
    for _, row in edges_gdf.iterrows():
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
def build_kdtree(nodes):
    coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
    return cKDTree(coords), nodes["osmid"].values, coords

# =========================
# LOAD + BUILD
# =========================
with st.spinner("Loading data..."):
    nodes, edges, nodes_wgs, edges_wgs = load_data()

with st.spinner("Building routing engine..."):
    G = build_graph(edges, ROUTING_ALPHA, BAYES_LAMBDA, ASSUME_BIDIRECTIONAL_ROADS)
    tree, osmids, coords = build_kdtree(nodes_wgs)

# =========================
# DASHBOARD METRICS
# =========================
st.markdown("### 📊 System Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Nodes", f"{len(nodes):,}")
col2.metric("Edges", f"{len(edges):,}")
col3.metric("Density", f"{nx.density(G):.4f}")

st.markdown("---")

# =========================
# SESSION STATE
# =========================
for k in ["origin", "destination", "origin_coords", "destination_coords", "reset"]:
    if k not in st.session_state:
        st.session_state[k] = None

def reset():
    for k in st.session_state:
        st.session_state[k] = None

# =========================
# MAIN LAYOUT (2-COLUMN)
# =========================
left, right = st.columns([2.5, 1])

# =========================
# MAP PANEL
# =========================
with left:
    st.markdown("### 🗺️ Interactive Map")

    center = [nodes_wgs.geometry.y.mean(), nodes_wgs.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    if st.session_state.origin_coords:
        folium.Marker(st.session_state.origin_coords, icon=folium.Icon(color="green")).add_to(m)

    if st.session_state.destination_coords:
        folium.Marker(st.session_state.destination_coords, icon=folium.Icon(color="red")).add_to(m)

    route_found = False
    route_nodes = 0

    if st.session_state.origin and st.session_state.destination:
        try:
            route = nx.shortest_path(G,
                                    st.session_state.origin,
                                    st.session_state.destination,
                                    weight="planned_cost")

            coords_route = []
            for n in route:
                idx = np.where(osmids == n)[0][0]
                lon, lat = coords[idx]
                coords_route.append((lat, lon))

            folium.PolyLine(coords_route, color="#3b82f6", weight=6).add_to(m)

            route_found = True
            route_nodes = len(route)

        except:
            st.error("No route found.")

    map_data = st_folium(m, width=900, height=600)

# =========================
# SIDE PANEL (RESULTS)
# =========================
with right:
    st.markdown("### 📍 Route Info")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if route_found:
        st.success("Route computed")

        st.metric("Nodes in Route", route_nodes)
        st.metric("Risk Factor", f"{ROUTING_ALPHA:.2f}")

    else:
        st.info("Click on the map to select origin and destination.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🎮 Actions")
    st.button("🔄 Reset Route", on_click=reset)

# =========================
# CLICK HANDLER
# =========================
if map_data.get("last_clicked") and not st.session_state.reset:
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

    _, idx = tree.query([lon, lat])
    node = osmids[idx]
    coord = (coords[idx][1], coords[idx][0])

    if st.session_state.origin is None:
        st.session_state.origin = node
        st.session_state.origin_coords = coord
        st.rerun()

    elif st.session_state.destination is None:
        st.session_state.destination = node
        st.session_state.destination_coords = coord
        st.rerun()

    else:
        reset()
        st.session_state.origin = node
        st.session_state.origin_coords = coord
        st.rerun()

st.session_state.reset = False

# =========================
# FOOTER
# =========================
st.markdown("""
---
<p style='text-align:center;color:gray'>
Bayesian Flood Routing System • Research Prototype
</p>
""", unsafe_allow_html=True)
