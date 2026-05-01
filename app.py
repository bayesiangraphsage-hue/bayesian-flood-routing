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
st.set_page_config(page_title="Bayesian Flood Routing", layout="wide")

# =========================
# HEADER
# =========================
st.title("Bayesian Flood Routing")
st.caption("Interactive route planning with flood-aware optimization")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Model Controls")

ROUTING_ALPHA = st.sidebar.slider("Risk Sensitivity (α)", 0.0, 5.0, 2.0)
BAYES_LAMBDA = st.sidebar.slider("Uncertainty Weight (λ)", 0.0, 3.0, 1.0)

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
    st.session_state.active_marker = "origin"

def reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]

# =========================
# LAYOUT
# =========================
left, right = st.columns([3, 1])

# =========================
# RIGHT PANEL
# =========================
with right:
    st.markdown("### 🎯 Controls")

    st.radio(
        "Active Marker",
        ["origin", "destination"],
        key="active_marker",
        format_func=lambda x: "Origin" if x == "origin" else "Destination"
    )

    st.markdown("### 📍 Current Positions")
    st.write("Origin:", st.session_state.origin)
    st.write("Destination:", st.session_state.destination)

    st.button("Reset", on_click=reset)

    st.info("Click on the map to reposition the selected marker.")

# =========================
# MAP
# =========================
with left:
    center = [nodes_wgs.geometry.y.mean(), nodes_wgs.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # --- markers ---
    if st.session_state.origin_coords:
        folium.Marker(
            st.session_state.origin_coords,
            tooltip="Origin",
            icon=folium.Icon(color="green" if st.session_state.active_marker == "origin" else "darkgreen")
        ).add_to(m)

    if st.session_state.destination_coords:
        folium.Marker(
            st.session_state.destination_coords,
            tooltip="Destination",
            icon=folium.Icon(color="red" if st.session_state.active_marker == "destination" else "darkred")
        ).add_to(m)

    # --- route ---
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

            folium.PolyLine(coords_route, color="#2563eb", weight=5).add_to(m)

        except:
            st.warning("No route found")

    # --- render map ---
    map_data = st_folium(m, width=900, height=600)

# =========================
# CLICK HANDLER (DRAG-LIKE UX)
# =========================
if map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    _, idx = tree.query([lon, lat])
    node = osmids[idx]
    coord = (coords[idx][1], coords[idx][0])

    if st.session_state.active_marker == "origin":
        st.session_state.origin = node
        st.session_state.origin_coords = coord

    else:
        st.session_state.destination = node
        st.session_state.destination_coords = coord

    st.rerun()
