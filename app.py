import streamlit as st
import geopandas as gpd
import networkx as nx
import folium
from streamlit_folium import st_folium
from scipy.spatial import cKDTree
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Bayesian Flood Routing",
    layout="wide",
    page_icon="🌊"
)

# --- HEADER ---
st.markdown("""
# 🌊 Bayesian Flood Routing System
### Intelligent Route Planning with Flood Risk Awareness
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Model Controls")

ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0)
ASSUME_BIDIRECTIONAL_ROADS = st.sidebar.checkbox("Bidirectional Roads", True)

st.sidebar.markdown("---")
st.sidebar.info("""
Adjust parameters to see how flood risk affects routing decisions.
""")

# --- FILE PATHS ---
NODES_PATH = "./processed_nodes.gpkg"
EDGES_PATH = "./prediction_test_sequence_0.gpkg"

# --- LOAD DATA ---
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
            key = row.get("key", 0)

            speed = float(row.get("speed_mps", 25 * 1000 / 3600))
            base_time = float(row.get("travel_time", row["length"] / speed))

            penalty = float(row.get("pred_flood_penalty", 0))
            uncertainty = float(row.get("uncertainty", 0))

            planned_penalty = np.clip(penalty + lam * uncertainty, 0, 1)
            cost = base_time * (1 + alpha * planned_penalty)

            attrs = {"planned_cost": cost, "length": row["length"]}

            G.add_edge(u, v, key=key, **attrs)

            if bidirectional:
                G.add_edge(v, u, key=f"{key}_rev", **attrs)

        except:
            continue

    return G


@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_kdtree(nodes):
    coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
    return cKDTree(coords), nodes["osmid"].values, coords


# --- SESSION STATE ---
for key in ["origin", "destination", "origin_coords", "destination_coords", "reset"]:
    if key not in st.session_state:
        st.session_state[key] = None


def reset():
    for key in st.session_state.keys():
        st.session_state[key] = None


# --- LOAD ---
with st.spinner("Loading GIS data..."):
    nodes, edges, nodes_wgs, edges_wgs = load_data()

# --- BUILD ---
with st.spinner("Building routing graph..."):
    G = build_graph(edges, ROUTING_ALPHA, BAYES_LAMBDA, ASSUME_BIDIRECTIONAL_ROADS)
    tree, osmids, coords = build_kdtree(nodes_wgs)

# --- METRICS ---
col1, col2, col3 = st.columns(3)

col1.metric("Nodes", len(nodes))
col2.metric("Edges", len(edges))
col3.metric("Graph Density", f"{nx.density(G):.4f}")

st.markdown("---")

# --- MAP ---
center = [nodes_wgs.geometry.y.mean(), nodes_wgs.geometry.x.mean()]

m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

# --- MARKERS ---
if st.session_state.origin_coords:
    folium.Marker(st.session_state.origin_coords, icon=folium.Icon(color="green")).add_to(m)

if st.session_state.destination_coords:
    folium.Marker(st.session_state.destination_coords, icon=folium.Icon(color="red")).add_to(m)

# --- ROUTE ---
route_found = False
route_length = 0
route_cost = 0

if st.session_state.origin and st.session_state.destination:
    try:
        route = nx.shortest_path(G,
                                st.session_state.origin,
                                st.session_state.destination,
                                weight="planned_cost")

        route_coords = []
        for n in route:
            idx = np.where(osmids == n)[0][0]
            lon, lat = coords[idx]
            route_coords.append((lat, lon))

        folium.PolyLine(route_coords, color="blue", weight=5).add_to(m)

        route_found = True
        route_length = len(route)

    except:
        st.error("No route found.")

# --- MAP DISPLAY ---
map_data = st_folium(m, width=1000, height=600)

# --- CLICK HANDLER ---
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

# --- ROUTE PANEL ---
if route_found:
    st.success("✅ Optimal route computed")

    c1, c2 = st.columns(2)
    c1.metric("Route Nodes", route_length)
    c2.metric("Risk-Aware Cost", f"{route_length * ROUTING_ALPHA:.2f}")

# --- BUTTONS ---
st.button("🔄 Reset", on_click=reset)

# --- FOOTER ---
st.markdown("""
---
💡 **Tip:** Increase α to avoid flooded roads more aggressively.
""")
