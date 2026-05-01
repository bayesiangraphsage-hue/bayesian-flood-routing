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
st.set_page_config(page_title="Flood Route Planner", layout="wide")

st.title("🌊 Flood-Route Planner using Bayesian GraphSAGE-GRU and Dijkstra's Algorithm")
st.caption("Flood-aware routing with hazard visualization, travel time, and uncertainty")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Model Controls")

ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0)
SHOW_FLOOD = st.sidebar.checkbox("Show Flood Hazard", True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    nodes = gpd.read_file("./processed_nodes.gpkg")
    edges = gpd.read_file("./prediction_test_sequence_0.gpkg")
    return nodes, edges, nodes.to_crs("EPSG:4326"), edges.to_crs("EPSG:4326")

nodes, edges, nodes_wgs, edges_wgs = load_data()

# =========================
# BUILD GRAPH
# =========================
@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_graph(edges, alpha, lam):
    G = nx.MultiDiGraph()

    for _, row in edges.iterrows():
        try:
            u, v = row["u"], row["v"]

            speed = float(row.get("speed_mps", 25 * 1000 / 3600))
            base_time = float(row.get("travel_time", row["length"] / speed))

            penalty = float(row.get("pred_flood_penalty", 0))
            uncertainty = float(row.get("uncertainty", 0))

            risk = np.clip(penalty + lam * uncertainty, 0, 1)
            cost = base_time * (1 + alpha * risk)

            attrs = {
                "planned_cost": cost,
                "travel_time": base_time,
                "risk": risk
            }

            G.add_edge(u, v, **attrs)
            G.add_edge(v, u, **attrs)

        except:
            continue

    return G

G = build_graph(edges, ROUTING_ALPHA, BAYES_LAMBDA)

# =========================
# KD TREE
# =========================
@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_tree(nodes):
    coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
    return cKDTree(coords), nodes["osmid"].values, coords

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
    st.session_state.route_time = None
    st.session_state.route_risk = None

def reset():
    st.session_state.origin = None
    st.session_state.destination = None
    st.session_state.origin_coords = None
    st.session_state.destination_coords = None
    st.session_state.route_time = None
    st.session_state.route_risk = None

# =========================
# LAYOUT
# =========================
left, right = st.columns([3, 1])

# =========================
# RIGHT PANEL
# =========================
with right:
    st.subheader("Selection")

    st.radio(
        "Active Marker",
        ["origin", "destination"],
        key="active",
        format_func=lambda x: "Origin" if x == "origin" else "Destination"
    )

    st.markdown("---")

    st.subheader("Route Info")

    if st.session_state.route_time:
        st.metric("Travel Time (min)", f"{st.session_state.route_time / 60:.2f}")
        st.metric("Avg Flood Risk", f"{st.session_state.route_risk:.2f}")
    else:
        st.info("Select origin and destination")

    st.button("Reset Route", on_click=reset)

# =========================
# MAP
# =========================
with left:
    center = [nodes_wgs.geometry.y.mean(), nodes_wgs.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=13, control_scale=True)

    folium.TileLayer("CartoDB positron").add_to(m)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.LayerControl().add_to(m)
    MiniMap().add_to(m)

    # =========================
    # FLOOD LAYER
    # =========================
    if SHOW_FLOOD:
        for _, row in edges_wgs.iterrows():
            try:
                geom = row.geometry
                risk = float(row.get("pred_flood_penalty", 0))

                if risk < 0.3:
                    color = "blue"
                elif risk < 0.6:
                    color = "orange"
                else:
                    color = "red"

                folium.GeoJson(
                    geom,
                    style_function=lambda x, col=color: {
                        "color": col,
                        "weight": 2,
                        "opacity": 0.5
                    }
                ).add_to(m)
            except:
                continue

    # =========================
    # MARKERS
    # =========================
    if st.session_state.origin_coords:
        folium.CircleMarker(st.session_state.origin_coords, radius=8, color="green", fill=True).add_to(m)

    if st.session_state.destination_coords:
        folium.CircleMarker(st.session_state.destination_coords, radius=8, color="red", fill=True).add_to(m)

    # =========================
    # ROUTE
    # =========================
    if st.session_state.origin and st.session_state.destination:
        try:
            route = nx.shortest_path(
                G,
                st.session_state.origin,
                st.session_state.destination,
                weight="planned_cost"
            )

            route_coords = []
            total_time = 0
            total_risk = 0

            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]

                edge_data = list(G.get_edge_data(u, v).values())[0]

                total_time += edge_data["travel_time"]
                total_risk += edge_data["risk"]

                idx = np.where(osmids == u)[0][0]
                lon, lat = coords[idx]
                route_coords.append((lat, lon))

            # last node
            idx = np.where(osmids == route[-1])[0][0]
            lon, lat = coords[idx]
            route_coords.append((lat, lon))

            st.session_state.route_time = total_time
            st.session_state.route_risk = total_risk / (len(route) - 1)

            folium.PolyLine(route_coords, color="#2563eb", weight=5).add_to(m)

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
