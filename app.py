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

# =========================
# TITLE
# =========================
st.title("Flood-Route Planner using Bayesian GraphSAGE-GRU and Dijkstra's Algorithm")
st.caption("Flood-aware routing with uncertainty modeling and optimal path computation")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Model Controls")

ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0)

show_flood = st.sidebar.checkbox("Show Flood Hazard Layer", True)

# =========================
# DATA
# =========================
@st.cache_data
def load_data():
    nodes = gpd.read_file("./processed_nodes.gpkg")
    edges = gpd.read_file("./prediction_test_sequence_0.gpkg")
    return nodes, edges, nodes.to_crs("EPSG:4326"), edges.to_crs("EPSG:4326")

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

            G.add_edge(u, v,
                       planned_cost=cost,
                       travel_time=base_time,
                       risk=risk)

            G.add_edge(v, u,
                       planned_cost=cost,
                       travel_time=base_time,
                       risk=risk)
        except:
            continue

    return G

@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_tree(nodes):
    coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
    return cKDTree(coords), nodes["osmid"].values, coords

nodes, edges, nodes_wgs, edges_wgs = load_data()
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

    st.markdown("### 📊 Route Info")

    route_time = 0
    route_risk = 0

    if "route_time" in st.session_state:
        route_time = st.session_state.route_time
        route_risk = st.session_state.route_risk

        st.metric("Travel Time (min)", f"{route_time/60:.2f}")
        st.metric("Avg Flood Risk", f"{route_risk:.2f}")

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
    # FLOOD HAZARD LAYER
    # =========================
    if show_flood:
        for _, row in edges_wgs.iterrows():
            try:
                geom = row.geometry
                risk = float(row.get("pred_flood_penalty", 0))

                if risk < 0.3:
                    color = "#3b82f6"  # blue
                elif risk < 0.6:
                    color = "#facc15"  # yellow
                else:
                    color = "#dc2626"  # red

                folium.GeoJson(
                    geom,
                    style_function=lambda x, col=color: {
                        "color": col,
                        "weight": 2,
                        "opacity": 0.6
                    }
                ).add_to(m)

            except:
                continue

    # markers
    if st.session_state.origin_coords:
        folium.CircleMarker(st.session_state.origin_coords, radius=8, color="green", fill=True).add_to(m)

    if st.session_state.destination_coords:
        folium.CircleMarker(st.session_state.destination_coords, radius=8, color="red", fill=True).add_to(m)

    # =========================
    # ROUTE COMPUTATION
    # =========================
    if st.session_state.origin and st.session_state.destination:
        try:
            route = nx.shortest_path(
                G,
                st.session_state.origin,
                st.session_state.destination,
                weight="planned_cost"
            )

            coords_route = []
            total_time = 0
            total_risk = 0

            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                edge_data = G.get_edge_data(u, v)[0]

                total_time += edge_data["travel_time"]
                total_risk += edge_data["risk"]

                idx = np.where(osmids == u)[0][0]
                lon, lat = coords[idx]
                coords_route.append((lat, lon))

            # add last node
            idx = np.where(osmids == route[-1])[0][0]
            lon, lat = coords[idx]
            coords_route.append((lat, lon))

            avg_risk = total_risk / (len(route)-1)

            st.session_state.route_time = total_time
            st.session_state.route_risk = avg_risk

            folium.PolyLine(coords_route, color="#2563eb", weight=5).add_to(m)

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
