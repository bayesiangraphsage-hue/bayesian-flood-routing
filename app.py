import streamlit as st
import pandas as pd
import numpy as np

# Friendly import check for PyTorch to provide an actionable Streamlit error
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    st.error(
        "Missing dependency: `torch`.\n\nInstall it locally with `pip install torch` or add it to `requirements.txt`."
    )
    st.stop()
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bayesian GraphSAGE-GRU OD Predictor",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# 2. GRAPH MODEL: BAYESIAN GRAPHSAGE-GRU WITH MC DROPOUT
# ============================================================

class MeanGraphSAGELayer(nn.Module):
    """
    Simple GraphSAGE-style mean aggregation layer.

    For each node:
    - Take its own features.
    - Take the mean features of its neighbors.
    - Concatenate them.
    - Pass through Linear + ReLU + Dropout.
    """

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        """
        x shape:        [num_nodes, in_dim]
        adj_norm shape: [num_nodes, num_nodes]
        """

        neighbor_mean = adj_norm @ x
        combined = torch.cat([x, neighbor_mean], dim=-1)

        h = self.linear(combined)
        h = torch.relu(h)
        h = self.dropout(h)

        return h


class BayesianGraphSAGEGRU(nn.Module):
    """
    Bayesian GraphSAGE-GRU model with dropout.

    Input:
    - x_seq: historical node features
      shape [time_steps, num_nodes, node_feature_dim]

    Output:
    - one scalar prediction for selected OD pair
    """

    def __init__(
        self,
        node_feature_dim,
        sage_hidden_dim=32,
        gru_hidden_dim=64,
        dropout=0.25
    ):
        super().__init__()

        self.sage1 = MeanGraphSAGELayer(
            in_dim=node_feature_dim,
            out_dim=sage_hidden_dim,
            dropout=dropout
        )

        self.sage2 = MeanGraphSAGELayer(
            in_dim=sage_hidden_dim,
            out_dim=sage_hidden_dim,
            dropout=dropout
        )

        # For each timestep, we combine:
        # origin embedding, destination embedding, absolute difference
        od_feature_dim = sage_hidden_dim * 3

        self.gru = nn.GRU(
            input_size=od_feature_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, sage_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sage_hidden_dim, 1)
        )

    def forward(self, x_seq, adj_norm, origin_idx, destination_idx):
        od_sequence = []

        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]

            h = self.sage1(x_t, adj_norm)
            h = self.sage2(h, adj_norm)

            origin_emb = h[origin_idx]
            dest_emb = h[destination_idx]
            diff_emb = torch.abs(origin_emb - dest_emb)

            od_emb = torch.cat([origin_emb, dest_emb, diff_emb], dim=-1)
            od_sequence.append(od_emb)

        od_sequence = torch.stack(od_sequence, dim=0)
        od_sequence = od_sequence.unsqueeze(0)

        _, hidden = self.gru(od_sequence)

        final_hidden = hidden[-1]
        prediction = self.output_head(final_hidden)

        return prediction.squeeze()


def enable_mc_dropout(model):
    """
    MC Dropout idea:
    - Keep the model mostly in eval mode.
    - Turn Dropout layers back on.
    """

    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model,
    x_seq,
    adj_norm,
    origin_idx,
    destination_idx,
    n_samples=100
):
    """
    Run many stochastic forward passes.

    Returns:
    - mean prediction
    - standard deviation
    - lower 95% interval
    - upper 95% interval
    - raw samples
    """

    enable_mc_dropout(model)

    samples = []

    for _ in range(n_samples):
        y = model(
            x_seq=x_seq,
            adj_norm=adj_norm,
            origin_idx=origin_idx,
            destination_idx=destination_idx
        )
        samples.append(float(y.cpu().item()))

    samples = np.array(samples)

    mean_pred = float(np.mean(samples))
    std_pred = float(np.std(samples))
    lower = float(np.percentile(samples, 2.5))
    upper = float(np.percentile(samples, 97.5))

    return mean_pred, std_pred, lower, upper, samples


# ============================================================
# 3. DATA LOADING
# ============================================================

DATA_DIR = Path("data")
NODES_PATH = DATA_DIR / "nodes.csv"
EDGES_PATH = DATA_DIR / "edges.csv"
FEATURES_PATH = DATA_DIR / "features.npy"
MODEL_PATH = Path("model_weights.pt")


def load_default_nodes_edges():
    if NODES_PATH.exists() and EDGES_PATH.exists():
        nodes = pd.read_csv(NODES_PATH)
        edges = pd.read_csv(EDGES_PATH)
    else:
        nodes = pd.DataFrame({
            "node_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "x": [0, 1, 2, 3, 4, 2, 4, 5],
            "y": [0, 1, 0, 1, 0, 2, 2, 1],
        })

        edges = pd.DataFrame({
            "source": ["A", "B", "C", "D", "B", "F", "G", "E", "C", "D"],
            "target": ["B", "C", "D", "E", "F", "G", "H", "H", "F", "G"],
        })

    nodes["node_id"] = nodes["node_id"].astype(str)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    return nodes, edges


def load_features(num_nodes, node_feature_dim=4, time_steps=12):
    """
    If data/features.npy exists, load it.
    Otherwise, create demo features.

    Expected real shape:
    [time_steps, num_nodes, node_feature_dim]
    """

    if FEATURES_PATH.exists():
        features = np.load(FEATURES_PATH).astype(np.float32)
    else:
        rng = np.random.default_rng(seed=42)

        # Demo-only features.
        # Replace with your real traffic/OD/node features.
        features = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(time_steps, num_nodes, node_feature_dim)
        ).astype(np.float32)

    return features


def build_adjacency_matrix(nodes, edges, undirected=True):
    node_ids = nodes["node_id"].tolist()
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    num_nodes = len(node_ids)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for _, row in edges.iterrows():
        source = str(row["source"])
        target = str(row["target"])

        if source not in id_to_idx or target not in id_to_idx:
            continue

        i = id_to_idx[source]
        j = id_to_idx[target]

        adj[i, j] = 1.0

        if undirected:
            adj[j, i] = 1.0

    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0

    adj_norm = adj / row_sum

    return torch.tensor(adj_norm, dtype=torch.float32), id_to_idx


# ============================================================
# 4. MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(
    node_feature_dim,
    sage_hidden_dim,
    gru_hidden_dim,
    dropout
):
    model = BayesianGraphSAGEGRU(
        node_feature_dim=node_feature_dim,
        sage_hidden_dim=sage_hidden_dim,
        gru_hidden_dim=gru_hidden_dim,
        dropout=dropout
    )

    checkpoint_loaded = False
    checkpoint_message = "Demo mode: no trained model_weights.pt found."

    if MODEL_PATH.exists():
        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            checkpoint_loaded = True
            checkpoint_message = "Loaded trained model_weights.pt."
        except Exception as e:
            checkpoint_message = f"Could not load model_weights.pt: {e}"

    model.eval()

    return model, checkpoint_loaded, checkpoint_message


# ============================================================
# 5. PLOTLY GRAPH DRAWING
# ============================================================

def create_graph_figure(nodes, edges, origin_id=None, destination_id=None):
    position = {
        str(row["node_id"]): (float(row["x"]), float(row["y"]))
        for _, row in nodes.iterrows()
    }

    edge_x = []
    edge_y = []

    for _, row in edges.iterrows():
        source = str(row["source"])
        target = str(row["target"])

        if source not in position or target not in position:
            continue

        x0, y0 = position[source]
        x1, y1 = position[target]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        showlegend=False
    )

    node_x = nodes["x"].astype(float).tolist()
    node_y = nodes["y"].astype(float).tolist()
    node_ids = nodes["node_id"].astype(str).tolist()

    marker_colors = []
    marker_sizes = []

    for node_id in node_ids:
        if node_id == origin_id:
            marker_colors.append("green")
            marker_sizes.append(24)
        elif node_id == destination_id:
            marker_colors.append("red")
            marker_sizes.append(24)
        else:
            marker_colors.append("lightblue")
            marker_sizes.append(18)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_ids,
        textposition="top center",
        customdata=node_ids,
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(width=2, color="black")
        ),
        hovertemplate="Node: %{customdata}<extra></extra>",
        showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Click/select a node: first Origin, second Destination",
        clickmode="event+select",
        dragmode="select",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )

    return fig


def extract_selected_node(event, nodes):
    """
    Streamlit Plotly selection result contains selected points.
    The node trace is curve_number = 1 because:
    trace 0 = edges
    trace 1 = nodes
    """

    if event is None:
        return None

    try:
        points = event["selection"]["points"]
    except Exception:
        return None

    if not points:
        return None

    node_points = [
        p for p in points
        if p.get("curve_number") == 1
    ]

    if not node_points:
        return None

    last_point = node_points[-1]

    if "customdata" in last_point:
        return str(last_point["customdata"])

    point_number = last_point.get("point_number")

    if point_number is None:
        return None

    return str(nodes.iloc[point_number]["node_id"])


# ============================================================
# 6. STREAMLIT USER INTERFACE
# ============================================================

st.title("🧠 Bayesian GraphSAGE-GRU OD Predictor with MC Dropout")

st.write(
    """
    This app lets you choose an Origin-Destination pair by selecting nodes on the graph.
    The model then produces a prediction and MC Dropout uncertainty estimate.
    """
)

st.warning(
    """
    Important: if you have not added your trained `model_weights.pt`, this app runs in demo mode.
    Demo predictions are not meaningful for real decisions.
    """
)

# -----------------------------
# Sidebar settings
# -----------------------------

st.sidebar.header("Settings")

output_name = st.sidebar.text_input(
    "Output name",
    value="Predicted OD value"
)

output_unit = st.sidebar.text_input(
    "Output unit",
    value="units"
)

sage_hidden_dim = st.sidebar.slider(
    "GraphSAGE hidden dimension",
    min_value=8,
    max_value=128,
    value=32,
    step=8
)

gru_hidden_dim = st.sidebar.slider(
    "GRU hidden dimension",
    min_value=16,
    max_value=256,
    value=64,
    step=16
)

dropout = st.sidebar.slider(
    "Dropout probability",
    min_value=0.05,
    max_value=0.70,
    value=0.25,
    step=0.05
)

mc_samples = st.sidebar.slider(
    "MC Dropout samples",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

st.sidebar.divider()

# -----------------------------
# Load data
# -----------------------------

nodes, edges = load_default_nodes_edges()
num_nodes = len(nodes)

features_np = load_features(num_nodes=num_nodes)
time_steps, feature_nodes, node_feature_dim = features_np.shape

if feature_nodes != num_nodes:
    st.error(
        f"""
        Feature mismatch:
        features.npy has {feature_nodes} nodes,
        but nodes.csv has {num_nodes} nodes.
        """
    )
    st.stop()

x_seq = torch.tensor(features_np, dtype=torch.float32)
adj_norm, id_to_idx = build_adjacency_matrix(nodes, edges)

node_ids = nodes["node_id"].astype(str).tolist()

# -----------------------------
# Session state
# -----------------------------

if "origin_id" not in st.session_state:
    st.session_state.origin_id = None

if "destination_id" not in st.session_state:
    st.session_state.destination_id = None

if "last_processed_node" not in st.session_state:
    st.session_state.last_processed_node = None


# -----------------------------
# Manual fallback controls
# -----------------------------

st.sidebar.subheader("Manual OD Selection")

manual_origin = st.sidebar.selectbox(
    "Manual Origin",
    options=["None"] + node_ids
)

manual_destination = st.sidebar.selectbox(
    "Manual Destination",
    options=["None"] + node_ids
)

if st.sidebar.button("Use manual OD pair"):
    st.session_state.origin_id = None if manual_origin == "None" else manual_origin
    st.session_state.destination_id = None if manual_destination == "None" else manual_destination
    st.session_state.last_processed_node = None

if st.sidebar.button("Reset OD pair"):
    st.session_state.origin_id = None
    st.session_state.destination_id = None
    st.session_state.last_processed_node = None


# -----------------------------
# Main layout
# -----------------------------

left_col, right_col = st.columns([2, 1])

with left_col:
    fig = create_graph_figure(
        nodes=nodes,
        edges=edges,
        origin_id=st.session_state.origin_id,
        destination_id=st.session_state.destination_id
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="network_graph"
    )

    selected_node = extract_selected_node(event, nodes)

    if (
        selected_node is not None
        and selected_node != st.session_state.last_processed_node
    ):
        if st.session_state.origin_id is None:
            st.session_state.origin_id = selected_node

        elif (
            st.session_state.destination_id is None
            and selected_node != st.session_state.origin_id
        ):
            st.session_state.destination_id = selected_node

        elif (
            st.session_state.origin_id is not None
            and st.session_state.destination_id is not None
        ):
            # Start a new OD pair if both are already selected
            st.session_state.origin_id = selected_node
            st.session_state.destination_id = None

        st.session_state.last_processed_node = selected_node
        st.rerun()


with right_col:
    st.subheader("Selected OD Pair")

    st.write(f"**Origin:** {st.session_state.origin_id}")
    st.write(f"**Destination:** {st.session_state.destination_id}")

    st.divider()

    st.subheader("Data Summary")
    st.write(f"Nodes: **{num_nodes}**")
    st.write(f"Edges: **{len(edges)}**")
    st.write(f"Time steps: **{time_steps}**")
    st.write(f"Node feature dimension: **{node_feature_dim}**")

    st.divider()

    model, checkpoint_loaded, checkpoint_message = load_model(
        node_feature_dim=node_feature_dim,
        sage_hidden_dim=sage_hidden_dim,
        gru_hidden_dim=gru_hidden_dim,
        dropout=dropout
    )

    if checkpoint_loaded:
        st.success(checkpoint_message)
    else:
        st.info(checkpoint_message)


# -----------------------------
# Prediction section
# -----------------------------

st.divider()
st.header("Prediction Output")

if st.session_state.origin_id is None or st.session_state.destination_id is None:
    st.info("Select two different nodes: first Origin, then Destination.")

elif st.session_state.origin_id == st.session_state.destination_id:
    st.error("Origin and Destination must be different nodes.")

else:
    origin_idx = id_to_idx[st.session_state.origin_id]
    destination_idx = id_to_idx[st.session_state.destination_id]

    mean_pred, std_pred, lower, upper, samples = mc_dropout_predict(
        model=model,
        x_seq=x_seq,
        adj_norm=adj_norm,
        origin_idx=origin_idx,
        destination_idx=destination_idx,
        n_samples=mc_samples
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label=f"Mean {output_name}",
            value=f"{mean_pred:.4f} {output_unit}"
        )

    with c2:
        st.metric(
            label="Uncertainty Std.",
            value=f"{std_pred:.4f}"
        )

    with c3:
        st.metric(
            label="95% MC Interval",
            value=f"{lower:.4f} to {upper:.4f}"
        )

    st.subheader("MC Dropout Samples")

    sample_df = pd.DataFrame({
        "sample_number": np.arange(1, len(samples) + 1),
        "prediction": samples
    })

    st.line_chart(
        sample_df,
        x="sample_number",
        y="prediction"
    )

    st.download_button(
        label="Download MC prediction samples as CSV",
        data=sample_df.to_csv(index=False),
        file_name="mc_dropout_predictions.csv",
        mime="text/csv"
    )
