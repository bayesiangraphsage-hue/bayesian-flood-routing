import streamlit as st
import geopandas as gpd
import networkx as nx
import folium
from streamlit_folium import st_folium
from scipy.spatial import cKDTree
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Flood Route Planner", layout="wide")

st.title("🌊 Flood-Route Planner using Bayesian GraphSAGE-GRU and Dijkstra's Algorithm")

# =========================
# SIDEBAR
# =========================
alpha = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
lam = st.sidebar.slider("Uncertainty (λ)", 0.0, 3.0, 1.0)
time_step = st.sidebar.slider("Flood Time", 0, 10, 0)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    nodes = gpd.read_file("./processed_nodes.gpkg")
    edges = gpd.read_file("./prediction_test_sequence_0.gpkg")
    return nodes.to_crs("EPSG:4326"), edges

nodes, edges = load_data()

# =========================
# KD TREE
# =========================
@st.cache_resource
def build_tree(nodes):
    coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
    return cKDTree(coords), nodes["osmid"].values, coords

tree, osmids, coords = build_tree(nodes)

# =========================
# BUILD GRAPH ONCE
# =========================
@st.cache_resource
def build_base_graph(edges):
    G = nx.MultiDiGraph()

    for row in edges.itertuples():
        try:
            u, v = row.u, row.v

            speed = getattr(row, "speed_mps", 25 * 1000 / 3600)
            length = getattr(row, "length", 0)
            base_time = getattr(row, "travel_time", length / speed)

            penalty = getattr(row, "pred_flood_penalty", 0)
            uncertainty = getattr(row, "uncertainty", 0)

            G.add_edge(u, v,
                       base_time=base_time,
                       penalty=penalty,
                       uncertainty=uncertainty)

            G.add_edge(v, u,
                       base_time=base_time,
                       penalty=penalty,
                       uncertainty=uncertainty)

        except:
            continue

    return G

G = build_base_graph(edges)

# =========================
# SESSION
# =========================
if "origin" not in st.session_state:
    st.session_state.origin = None
    st.session_state.destination = None
    st.session_state.origin_coords = None
    st.session_state.destination_coords = None
    st.session_state.active = "origin"

# =========================
# UI
# =========================
col1, col2 = st.columns([3, 1])

with col2:
    st.radio("Marker", ["origin", "destination"], key="active")

    if st.button("Reset"):
        for k in st.session_state.keys():
            st.session_state[k] = None

# =========================
# MAP (FAST)
# =========================
with col1:
    center = [nodes.geometry.y.mean(), nodes.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=13)

    # markers
    if st.session_state.origin_coords:
        folium.CircleMarker(st.session_state.origin_coords, radius=8, color="green", fill=True).add_to(m)

    if st.session_state.destination_coords:
        folium.CircleMarker(st.session_state.destination_coords, radius=8, color="red", fill=True).add_to(m)

    # =========================
    # ROUTE (FAST COMPUTATION)
    # =========================
    if st.session_state.origin and st.session_state.destination:

        def dynamic_weight(u, v, d):
            penalty = d["penalty"] * (1 + 0.1 * time_step)
            risk = np.clip(penalty + lam * d["uncertainty"], 0, 1)
            return d["base_time"] * (1 + alpha * risk)

        try:
            route = nx.shortest_path(
                G,
                st.session_state.origin,
                st.session_state.destination,
                weight=dynamic_weight
            )

            coords_route = []
            total_time = 0

            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                edge = list(G.get_edge_data(u, v).values())[0]

                total_time += edge["base_time"]

                idx = np.where(osmids == u)[0][0]
                lon, lat = coords[idx]
                coords_route.append((lat, lon))

            folium.PolyLine(coords_route, color="blue", weight=5).add_to(m)

            st.success(f"Travel Time: {total_time/60:.2f} min")

        except:
            st.warning("No route found")

    map_data = st_folium(m, height=600, use_container_width=True)

# =========================
# CLICK HANDLER
# =========================
if map_data and map_data.get("last_clicked"):
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
