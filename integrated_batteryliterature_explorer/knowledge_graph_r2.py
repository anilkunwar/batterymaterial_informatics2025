import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# Load data
nodes_df = pd.read_csv("knowledge_graph_nodes.csv")
edges_df = pd.read_csv("knowledge_graph_edges.csv")

# Build graph
G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(row["id"], label=row["label"], category=row["category"])
for _, row in edges_df.iterrows():
    G.add_edge(row["source"], row["target"])

# Sidebar filters
st.sidebar.header("Filters")
categories = nodes_df["category"].unique()
selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)

# Font size control
label_font_size = st.sidebar.slider("Label Font Size", min_value=8, max_value=24, value=12)

# Filter graph
filtered_nodes = [n for n, attr in G.nodes(data=True) if attr["category"] in selected_categories]
G_filtered = G.subgraph(filtered_nodes)

# Layout
pos = nx.spring_layout(G_filtered, k=0.5, seed=42)

# Node coloring based on category
categories = sorted(set(nx.get_node_attributes(G_filtered, 'category').values()))
category_color_map = {cat: color for cat, color in zip(categories, px.colors.qualitative.Set3)}

node_colors = [
    category_color_map.get(G_filtered.nodes[node].get('category', ''), "lightgray")
    for node in G_filtered.nodes()
]

# Edge traces
edge_x = []
edge_y = []
for edge in G_filtered.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines"
)

# Node traces
node_x = []
node_y = []
node_text = []
for node in G_filtered.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(G_filtered.nodes[node]["label"])

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    textfont=dict(size=label_font_size),
    hoverinfo="text",
    marker=dict(
        color=node_colors,
        size=10,
        line_width=2
    )
)

# Legend
legend_items = []
for cat, color in category_color_map.items():
    legend_items.append(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            legendgroup=cat,
            showlegend=True,
            name=cat
        )
    )

# Figure
fig = go.Figure(data=[edge_trace, node_trace] + legend_items,
                layout=go.Layout(
                    showlegend=True,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

st.plotly_chart(fig, use_container_width=True)
