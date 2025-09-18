import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components

# Define the directory for data files
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

# Load data
edges_df, nodes_df = load_data()

# Create a directed graph for hierarchy and an undirected graph for the main viz
G_directed = nx.DiGraph()
G_undirected = nx.Graph()

# Add nodes with attributes
for _, row in nodes_df.iterrows():
    node_attrs = {
        'type': row['type'],
        'category': row['category'],
        'size': row['frequency'] / 10 if row['frequency'] > 0 else 5  # Scale node size
    }
    G_directed.add_node(row['node'], **node_attrs)
    G_undirected.add_node(row['node'], **node_attrs)

# Add edges with attributes
for _, row in edges_df.iterrows():
    edge_attrs = {
        'weight': row['weight'],
        'type': row['type'],
        'label': row['label']
    }
    # Add to both graphs for different layout purposes
    G_directed.add_edge(row['source'], row['target'], **edge_attrs)
    if row['type'] == 'term-term':  # Only add term-term edges to the undirected graph
        G_undirected.add_edge(row['source'], row['target'], **edge_attrs)

# Streamlit App
st.title("🔋 Battery Research Knowledge Graph Explorer")
st.markdown("""
This interactive visualization explores the relationships between key concepts in battery mechanical degradation research.
*   **Nodes:** Circles are **Terms**, Hexagons are **Categories**.
*   **Edges:** Grey lines show `belongs_to`, Colored lines show `related_to`.
*   **Size:** Node size is proportional to term frequency.
""")

# Sidebar Controls
st.sidebar.header("Visualization Controls")
layout_option = st.sidebar.selectbox(
    "Choose Layout",
    ("Hierarchical (Clarity)", "Force-Directed (Connectivity)")
)

# Color mapping
category_list = nodes_df['category'].dropna().unique()
color_map = {cat: f'hsl({i*(360/len(category_list))}, 70%, 50%)' for i, cat in enumerate(category_list)}
term_color_map = {row['node']: color_map.get(row['category'], 'grey') for _, row in nodes_df.iterrows()}

# Generate positions based on selection
if layout_option == "Hierarchical (Clarity)":
    pos = nx.spring_layout(G_directed, k=2, iterations=50, weight='weight', seed=42)
    # Push category nodes to the top
    for node in G_directed.nodes:
        if G_directed.nodes[node]['type'] == 'category':
            pos[node][1] += 2  # Adjust Y-coordinate
else:  # Force-Directed
    pos = nx.spring_layout(G_undirected, k=1, iterations=100, weight='weight', seed=42)

# Create Plotly Figure
fig = go.Figure()

# Add edges to the figure
edge_traces = []
for edge in G_undirected.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_weight = G_undirected.edges[edge].get('weight', 1)
    edge_type = G_undirected.edges[edge].get('type', '')

    # Customize edges based on type
    if edge_type == 'category-term':
        line_color = 'lightgrey'
        line_width = 1
        opacity = 0.6
    else: # term-term
        line_color = 'cornflowerblue'
        line_width = edge_weight / 50  # Scale line width
        opacity = 0.3 + (edge_weight / edges_df['weight'].max()) * 0.7 # Scale opacity

    edge_trace = go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        line=dict(width=line_width, color=line_color),
        hoverinfo='none',
        mode='lines',
        opacity=opacity
    )
    edge_traces.append(edge_trace)

for trace in edge_traces:
    fig.add_trace(trace)

# Add nodes to the figure
node_x = []; node_y = []; node_text = []; node_color = []; node_size = []
for node in G_undirected.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>Frequency: {G_undirected.nodes[node].get('size', 0)*10}")
    node_color.append(term_color_map.get(node, 'grey'))
    node_size.append(G_undirected.nodes[node].get('size', 10)) # Use scaled size

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=[node for node in G_undirected.nodes()],
    textposition="middle center",
    marker=dict(
        showscale=False,
        color=node_color,
        size=node_size,
        line_width=1,
        line_color='black'
    )
)
fig.add_trace(node_trace)

# Layout settings
fig.update_layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=0, l=0, r=0, t=0),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    height=700,
    title_text=f"Battery Knowledge Graph: {layout_option}", title_x=0.5
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Add a download button for the processed graph data
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

processed_edges = edges_df[['source', 'target', 'weight', 'type', 'label']]
csv = convert_df_to_csv(processed_edges)

st.sidebar.download_button(
    label="📥 Download Processed Edge Data",
    data=csv,
    file_name='battery_kg_processed_edges.csv',
    mime='text/csv',
)

st.sidebar.markdown("---")
st.sidebar.info("**Instructions:** Use the selector to change the layout. Hover over nodes for details. The download provides the cleaned edge list for further analysis.")
