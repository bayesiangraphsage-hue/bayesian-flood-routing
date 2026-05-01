import streamlit as st
import geopandas as gpd
import networkx as nx
import folium
from streamlit_folium import st_folium
from scipy.spatial import cKDTree
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Bayesian Flood Routing", layout="wide")
st.title("Interactive Bayesian Routing Software")

# --- Constants ---
ROUTING_ALPHA = 2.0
BAYES_LAMBDA = 1.0
ASSUME_BIDIRECTIONAL_ROADS = True

# --- File Paths ---
NODES_PATH = "./processed_nodes.gpkg"
EDGES_PATH = "./prediction_test_sequence_0.gpkg"

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        nodes_gdf = gpd.read_file(NODES_PATH)
        edges_gdf = gpd.read_file(EDGES_PATH)

        # Convert to WGS84 for mapping
        nodes_wgs84 = nodes_gdf.to_crs("EPSG:4326")
        edges_wgs84 = edges_gdf.to_crs("EPSG:4326")

        return nodes_gdf, edges_gdf, nodes_wgs84, edges_wgs84
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


# --- FIXED: Graph Builder (handles unhashable GeoDataFrame) ---
@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_graph(edges_gdf):
    edges_gdf = edges_gdf.copy()  # avoid mutation issues
    R = nx.MultiDiGraph()

    for _, row in edges_gdf.iterrows():
        try:
            u = row["u"]
            v = row["v"]
            key = row.get("key", 0)

            # Base travel time
            speed = float(row.get("speed_mps", 25 * 1000 / 3600))
            base_time = float(row.get("travel_time", row["length"] / speed))

            # Flood model inputs
            penalty = float(row.get("pred_flood_penalty", 0.0))
            uncertainty = float(row.get("uncertainty", 0.0))

            planned_penalty = np.clip(penalty + BAYES_LAMBDA * uncertainty, 0, 1)
            planned_cost = base_time * (1 + ROUTING_ALPHA * planned_penalty)

            edge_attrs = {
                "edge_id": row.get("edge_id", 0),
                "length": float(row.get("length", 0)),
                "travel_time": base_time,
                "planned_cost": planned_cost,
            }

            R.add_edge(u, v, key=key, **edge_attrs)

            if ASSUME_BIDIRECTIONAL_ROADS:
                R.add_edge(v, u, key=f"{key}_rev", **edge_attrs)

        except Exception as e:
            # Skip problematic rows instead of crashing
            continue

    return R


# --- Spatial Index ---
@st.cache_resource(hash_funcs={gpd.GeoDataFrame: lambda _: None})
def build_spatial_index(nodes_wgs84):
    coords = np.array(list(zip(nodes_wgs84.geometry.x, nodes_wgs84.geometry.y)))
    tree = cKDTree(coords)
    osmids = nodes_wgs84["osmid"].values
    return tree, osmids, coords


# --- Session State ---
if "origin" not in st.session_state:
    st.session_state.origin = None
    st.session_state.origin_coords = None

if "destination" not in st.session_state:
    st.session_state.destination = None
    st.session_state.destination_coords = None

if "reset" not in st.session_state:
    st.session_state.reset = False


def reset_app():
    st.session_state.origin = None
    st.session_state.origin_coords = None
    st.session_state.destination = None
    st.session_state.destination_coords = None
    st.session_state.reset = True


# --- Main App ---
nodes_gdf, edges_gdf, nodes_wgs84, edges_wgs84 = load_data()

if nodes_gdf is not None and edges_gdf is not None:

    graph = build_graph(edges_gdf)
    spatial_tree, node_osmids, node_coords = build_spatial_index(nodes_wgs84)

    st.markdown("### Instructions:")
    st.markdown("1. Click map → set Origin (Green)")
    st.markdown("2. Click again → set Destination (Red)")
    st.markdown("3. Bayesian optimal route will appear")
    st.button("Reset Selection", on_click=reset_app)

    # Map center
    center_lat = nodes_wgs84.geometry.y.mean()
    center_lon = nodes_wgs84.geometry.x.mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron"
    )

    route_found = False

    # Draw markers
    if st.session_state.origin_coords:
        folium.Marker(
            st.session_state.origin_coords,
            popup="Origin",
            icon=folium.Icon(color="green")
        ).add_to(m)

    if st.session_state.destination_coords:
        folium.Marker(
            st.session_state.destination_coords,
            popup="Destination",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # --- Routing ---
    if st.session_state.origin and st.session_state.destination:
        try:
            route = nx.shortest_path(
                graph,
                source=st.session_state.origin,
                target=st.session_state.destination,
                weight="planned_cost"
            )

            route_coords = []
            for n in route:
                idx = np.where(node_osmids == n)[0][0]
                lon, lat = node_coords[idx]
                route_coords.append((lat, lon))

            folium.PolyLine(
                route_coords,
                color="blue",
                weight=4,
                opacity=0.8
            ).add_to(m)

            st.success(f"Route found! Nodes: {len(route)}")
            route_found = True

        except nx.NetworkXNoPath:
            st.error("No path found.")
        except Exception as e:
            st.error(f"Routing error: {e}")

    # --- Render Map ---
    map_data = st_folium(m, width=1000, height=600)

    # --- Handle Clicks ---
    if map_data.get("last_clicked") and not st.session_state.reset:
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]

        dist, idx = spatial_tree.query([click_lon, click_lat])
        nearest_node = node_osmids[idx]
        nearest_latlon = (node_coords[idx][1], node_coords[idx][0])

        if st.session_state.origin is None:
            st.session_state.origin = nearest_node
            st.session_state.origin_coords = nearest_latlon
            st.rerun()

        elif st.session_state.destination is None:
            st.session_state.destination = nearest_node
            st.session_state.destination_coords = nearest_latlon
            st.rerun()

        elif route_found:
            # Reset for new route
            st.session_state.origin = nearest_node
            st.session_state.origin_coords = nearest_latlon
            st.session_state.destination = None
            st.session_state.destination_coords = None
            st.rerun()

    st.session_state.reset = False

else:
    st.warning("Waiting for data to load...")
