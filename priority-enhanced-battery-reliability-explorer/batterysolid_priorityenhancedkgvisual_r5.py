import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms import community
import re

# -----------------------
# 1. Data Loading
# -----------------------
DB_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

@st.cache_data
def load_data():
    edges_path = os.path.join(DB_DIR, 'knowledge_graph_edges.csv')
    nodes_path = os.path.join(DB_DIR, 'knowledge_graph_nodes.csv')
    
    if not os.path.exists(edges_path) or not os.path.exists(nodes_path):
        st.error("❌ One or both CSV files are missing. Please upload 'knowledge_graph_edges.csv' and 'knowledge_graph_nodes.csv'.")
        st.stop()
    
    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(nodes_path)
    return edges_df, nodes_df

edges_df, nodes_df = load_data()

# -----------------------
# 2. Normalize Terms
# -----------------------
def normalize_term(term: str) -> str:
    term = term.lower().strip()
    replacements = {
        "batteries": "battery",
        "materials": "material",
        "mater": "material",
        "lithium ions": "lithium ion",
        "lithium ion(s)": "lithium ion",
        "fatigue": "fatigue",
    }
    return replacements.get(term, term)

nodes_df["node"] = nodes_df["node"].apply(normalize_term)
edges_df["source"] = edges_df["source"].apply(normalize_term)
edges_df["target"] = edges_df["target"].apply(normalize_term)

# -----------------------
# 3. Build Graph
# -----------------------
G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(row["node"], 
               type=row["type"], 
               category=row["category"], 
               frequency=row["frequency"],
               size=row["frequency"]/10 if row["frequency"] > 0 else 5)

for _, row in edges_df.iterrows():
    G.add_edge(row["source"], row["target"], 
               weight=row["weight"], 
               type=row["type"], 
               label=row["label"])

# -----------------------
# 4. Sidebar Controls
# -----------------------
st.title("🔋 Battery Research Knowledge Graph Explorer")

st.markdown("""
Explore key concepts in **battery mechanical degradation research**.  
- **Nodes** = Terms (colored by cluster).  
- **Size** = Term frequency.  
- **Edges** = Relationships (thicker = stronger).  
Use the sidebar to filter and explore deeper insights.
""")

# Edge weight filter
min_weight = st.sidebar.slider(
    "Minimum edge weight (filter weak links)", 
    min_value=int(edges_df["weight"].min()), 
    max_value=int(edges_df["weight"].max()), 
    value=10, step=1
)

# Node frequency filter
min_node_freq = st.sidebar.slider(
    "Minimum node frequency", 
    min_value=int(nodes_df["frequency"].min()), 
    max_value=int(nodes_df["frequency"].max()), 
    value=5, step=1
)

# Category filter
categories = sorted(nodes_df["category"].dropna().unique())
selected_categories = st.sidebar.multiselect("Filter categories", categories, default=categories)

# Type filter
types = sorted(nodes_df["type"].dropna().unique())
selected_types = st.sidebar.multiselect("Filter types", types, default=types)

# Node exclusion
st.sidebar.subheader("🚫 Exclude Unwanted Nodes")
exclude_input = st.sidebar.text_area(
    "Enter nodes or substrings to exclude (comma-separated):",
    placeholder="e.g., smith, john, author, research"
)
exclude_terms = [t.strip().lower() for t in exclude_input.split(",") if t.strip()]

# Default exclusion
default_exclude = ["author", "study", "research", "data"]
if st.sidebar.checkbox("Auto-exclude generic terms", value=True):
    exclude_terms.extend(default_exclude)

def should_exclude_node(node_name: str, exclude_terms: list) -> bool:
    node_lower = str(node_name).lower()
    return any(term in node_lower for term in exclude_terms)

# -----------------------
# 5. Graph Filtering
# -----------------------
def filter_graph(G, min_weight, min_freq, selected_categories, selected_types, exclude_terms):
    G_filtered = nx.Graph()
    
    for n, d in G.nodes(data=True):
        if (d.get("frequency", 0) >= min_freq and 
            d.get("category", "") in selected_categories and
            d.get("type", "") in selected_types and
            not should_exclude_node(n, exclude_terms)):
            G_filtered.add_node(n, **d)
    
    for u, v, d in G.edges(data=True):
        if (u in G_filtered.nodes and v in G_filtered.nodes and 
            d.get("weight", 0) >= min_weight):
            G_filtered.add_edge(u, v, **d)
    
    return G_filtered

G_filtered = filter_graph(G, min_weight, min_node_freq, selected_categories, selected_types, exclude_terms)

# -----------------------
# 6. Community Detection
# -----------------------
communities = community.greedy_modularity_communities(G_filtered)
comm_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        comm_map[node] = i
colors = [f"hsl({comm_map[node]*50},70%,50%)" for node in G_filtered.nodes()]

# -----------------------
# 7. Layout Options
# -----------------------
layout_choice = st.sidebar.selectbox(
    "Graph layout",
    ["Spring", "Kamada-Kawai", "Circular", "Spectral"]
)
if layout_choice == "Kamada-Kawai":
    pos = nx.kamada_kawai_layout(G_filtered)
elif layout_choice == "Circular":
    pos = nx.circular_layout(G_filtered)
elif layout_choice == "Spectral":
    pos = nx.spectral_layout(G_filtered)
else:
    pos = nx.spring_layout(G_filtered, k=1, iterations=100, seed=42)

# -----------------------
# 8. Visualization
# -----------------------
fig = go.Figure()

# Add edges
for edge in G_filtered.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = G_filtered.edges[edge]["weight"]
    fig.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        line=dict(width=max(1, weight/20), color="lightgrey"),
        mode="lines", hoverinfo="none", opacity=0.4
    ))

# Add nodes
node_x, node_y, node_size, node_text, text_sizes = [], [], [], [], []
for node, data in G_filtered.nodes(data=True):
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_size.append(data.get("size", 10))
    node_text.append(f"{node}<br>Freq: {data.get('frequency', 0)}")
    text_sizes.append(10 + (data.get("size", 10) / 2))

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=[n for n in G_filtered.nodes()],
    textposition="top center",
    textfont=dict(size=text_sizes),
    hoverinfo="text",
    marker=dict(
        size=node_size,
        color=colors,
        line=dict(width=1, color="black")
    )
))

fig.update_layout(
    showlegend=False,
    hovermode="closest",
    margin=dict(b=0, l=0, r=0, t=0),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=800,
    dragmode="pan",
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 9. Inference Tools
# -----------------------
# Hubs
centrality = nx.degree_centrality(G_filtered)
hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
st.sidebar.subheader("🔑 Top Hub Nodes")
for node, score in hubs:
    st.sidebar.write(f"**{node}** ({score:.2f})")

# Betweenness centrality
bet_centrality = nx.betweenness_centrality(G_filtered)
bridges = sorted(bet_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
st.sidebar.subheader("🌐 Key Bridge Nodes")
for node, score in bridges:
    st.sidebar.write(f"{node} ({score:.3f})")

# Communities
st.sidebar.subheader("🧩 Communities")
for i, comm in enumerate(communities):
    st.sidebar.write(f"Cluster {i+1}: {len(comm)} nodes")

# Path finding
st.sidebar.subheader("🔗 Find Path Between Nodes")
node_a = st.sidebar.selectbox("Source node", sorted(G_filtered.nodes()), key="path_source")
node_b = st.sidebar.selectbox("Target node", sorted(G_filtered.nodes()), key="path_target")
if st.sidebar.button("Find Path"):
    try:
        path = nx.shortest_path(G_filtered, source=node_a, target=node_b)
        st.sidebar.success(" → ".join(path))
    except nx.NetworkXNoPath:
        st.sidebar.warning("No path found.")

# -----------------------
# 10. Node Details
# -----------------------
st.sidebar.subheader("🔍 Explore Node Details")
if len(G_filtered.nodes) > 0:
    selected_node = st.sidebar.selectbox("Choose a node", sorted(G_filtered.nodes()))
    if selected_node:
        node_data = G_filtered.nodes[selected_node]
        st.sidebar.markdown(f"### {selected_node.title()}")
        st.sidebar.write(f"**Category:** {node_data.get('category','N/A')}")
        st.sidebar.write(f"**Type:** {node_data.get('type','N/A')}")
        st.sidebar.write(f"**Frequency:** {node_data.get('frequency','N/A')}")

        neighbors = list(G_filtered.neighbors(selected_node))
        if neighbors:
            st.sidebar.write("**Connected Terms:**")
            for n in neighbors:
                w = G_filtered.edges[selected_node, n].get("weight", 1)
                st.sidebar.write(f"- {n} (weight {w})")
        else:
            st.sidebar.write("No connected terms above current filter threshold.")
