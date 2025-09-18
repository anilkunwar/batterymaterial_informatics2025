import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms import community

# -----------------------
# 1. Data Loading
# -----------------------
@st.cache_data
def load_data():
    edges_df = pd.read_csv("knowledge_graph_edges.csv")
    nodes_df = pd.read_csv("knowledge_graph_nodes.csv")
    return edges_df, nodes_df

edges_df, nodes_df = load_data()

# -----------------------
# 2. Normalize Terms
# -----------------------
import re
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
Click a node in the sidebar to explore its details.
""")

min_weight = st.sidebar.slider(
    "Minimum edge weight (filter weak links)", 
    min_value=int(edges_df["weight"].min()), 
    max_value=int(edges_df["weight"].max()), 
    value=10, step=1
)

# Filter weak edges
G_filtered = nx.Graph(((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= min_weight))
for n, d in G.nodes(data=True):
    if n in G_filtered.nodes:
        G_filtered.nodes[n].update(d)

# -----------------------
# 5. Community Detection
# -----------------------
communities = community.greedy_modularity_communities(G_filtered)
comm_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        comm_map[node] = i
colors = [f"hsl({comm_map[node]*50},70%,50%)" for node in G_filtered.nodes()]

# -----------------------
# 6. Node Positions & Visualization
# -----------------------
pos = nx.spring_layout(G_filtered, k=1, iterations=100, seed=42)

fig = go.Figure()

# Add edges
for edge in G_filtered.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = G_filtered.edges[edge]["weight"]
    fig.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        line=dict(width=weight/50, color="lightgrey"),
        mode="lines", hoverinfo="none", opacity=0.4
    ))

# Add nodes
node_x, node_y, node_size, node_text = [], [], [], []
for node, data in G_filtered.nodes(data=True):
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_size.append(data.get("size", 10))
    node_text.append(f"{node}<br>Freq: {int(data.get('size', 1)*10)}")

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=[n for n in G_filtered.nodes()],
    textposition="middle center",
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
    height=700,
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 7. Hub Nodes
# -----------------------
centrality = nx.degree_centrality(G_filtered)
hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
st.sidebar.subheader("🔑 Top Hub Nodes")
for node, score in hubs:
    st.sidebar.write(f"**{node}** ({score:.2f})")

# -----------------------
# 8. Node Details Panel
# -----------------------
st.sidebar.subheader("🔍 Explore Node Details")
selected_node = st.sidebar.selectbox("Choose a node", sorted(G_filtered.nodes()))

if selected_node:
    node_data = G_filtered.nodes[selected_node]
    st.sidebar.markdown(f"### {selected_node.title()}")
    st.sidebar.write(f"**Category:** {node_data.get('category','N/A')}")
    st.sidebar.write(f"**Type:** {node_data.get('type','N/A')}")
    st.sidebar.write(f"**Frequency:** {int(node_data.get('size',1)*10)}")

    neighbors = list(G_filtered.neighbors(selected_node))
    if neighbors:
        st.sidebar.write("**Connected Terms:**")
        for n in neighbors:
            w = G_filtered.edges[selected_node, n].get("weight", 1)
            st.sidebar.write(f"- {n} (weight {w})")
    else:
        st.sidebar.write("No connected terms above current filter threshold.")

