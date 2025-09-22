import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from networkx.algorithms import community
import community as community_louvain
from collections import Counter
import numpy as np
import traceback
from itertools import combinations
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# -----------------------
# 1. Setup and Data Loading with Caching
# -----------------------
DB_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

KEY_TERMS = [
    "electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress",
    "micro-cracking", "electrolyte degradation", "capacity fade", "lithium plating", "thermal runaway",
    "mechanical degradation", "cycle life", "lithium", "electrode", "crack", "fracture", "battery",
    "particles", "cathode", "mechanical", "cycles", "electrolyte", "degradation", "surface", "capacity",
    "cycling", "stress", "diffusion", "solid electrolyte interphase", "impendence", "degrades the battery capacity",
    "cycling degradation", "calendar degradation", "complex cycling damage", "chemo-mechanical degradation mechanisms",
    "microcrack formation", "active particles", "differential degradation mechanisms", "SOL swing", "lithiation",
    "electrochemical performance", "mechanical integrity", "battery safety", "Coupled mechanical-chemical degradation",
    "physics-based models", "predict degradation mechanisms", "Electrode Side Reactions", "Capacity Loss",
    "Mechanical Degradation", "Particle Versus SEI Cracking", "degradation models", "predict degradation"
]

# Load SciBERT model for semantic similarity
@st.cache_resource
def load_scibert():
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"Failed to load SciBERT: {str(e)}. Semantic similarity will be disabled.")
        return None, None

scibert_tokenizer, scibert_model = load_scibert()

@st.cache_data
def get_scibert_embedding(texts):
    if scibert_tokenizer is None or scibert_model is None:
        return [None] * len(texts) if isinstance(texts, list) else None
    try:
        if isinstance(texts, str):
            texts = [texts]
        if not texts or all(not t.strip() for t in texts):
            return [None] * len(texts)
        inputs = scibert_tokenizer(texts, return_tensors="pt", truncation=True, max_length=64, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1].mean(dim=1).numpy()
        embeddings = []
        for emb in last_hidden_states:
            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm != 0 else None)
        return embeddings if len(texts) > 1 else embeddings[0]
    except Exception as e:
        st.warning(f"SciBERT embedding failed: {str(e)}")
        return [None] * len(texts) if isinstance(texts, list) else None

KEY_TERMS_EMBEDDINGS = get_scibert_embedding(KEY_TERMS)
KEY_TERMS_EMBEDDINGS = [emb for emb in KEY_TERMS_EMBEDDINGS if emb is not None]

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

# -----------------------
# 2. Priority Score Calculation
# -----------------------
def calculate_priority_scores(G, nodes_df):
    max_freq = nodes_df['frequency'].max() if nodes_df['frequency'].max() > 0 else 1
    nodes_df['norm_frequency'] = nodes_df['frequency'] / max_freq
    
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_centrality = {node: 0 for node in G.nodes()}
    
    node_terms = nodes_df['node'].tolist()
    term_embeddings = get_scibert_embedding(node_terms)
    term_embeddings_dict = dict(zip(node_terms, term_embeddings))
    
    semantic_scores = {}
    for node in G.nodes():
        emb = term_embeddings_dict.get(node)
        if emb is None:
            semantic_scores[node] = 0
        else:
            similarities = [cosine_similarity([emb], [kt_emb])[0][0] for kt_emb in KEY_TERMS_EMBEDDINGS]
            semantic_scores[node] = max(similarities, default=0)
    
    priority_scores = {}
    for node in G.nodes():
        priority_scores[node] = (
            0.4 * nodes_df[nodes_df['node'] == node]['norm_frequency'].iloc[0] +
            0.3 * degree_centrality.get(node, 0) +
            0.2 * betweenness_centrality.get(node, 0) +
            0.1 * semantic_scores.get(node, 0)
        )
    
    return priority_scores

# -----------------------
# 3. Failure Analysis Functions
# -----------------------
def analyze_failure_centrality(G_filtered, focus_terms=None):
    if focus_terms is None:
        focus_terms = ["crack", "fracture", "degradation", "fatigue", "damage", "failure", "mechanical", "cycling", "capacity fade", "SEI"]
    
    degree_centrality = nx.degree_centrality(G_filtered)
    betweenness_centrality = nx.betweenness_centrality(G_filtered)
    closeness_centrality = nx.closeness_centrality(G_filtered)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_filtered, max_iter=1000)
    except:
        eigenvector_centrality = {node: 0 for node in G_filtered.nodes()}
    
    centrality_results = []
    for node in G_filtered.nodes():
        if any(term in node.lower() for term in focus_terms):
            centrality_results.append({
                'node': node,
                'degree': degree_centrality.get(node, 0),
                'betweenness': betweenness_centrality.get(node, 0),
                'closeness': closeness_centrality.get(node, 0),
                'eigenvector': eigenvector_centrality.get(node, 0),
                'category': G_filtered.nodes[node].get('category', ''),
                'type': G_filtered.nodes[node].get('type', '')
            })
    
    return pd.DataFrame(centrality_results)

def detect_failure_communities(G_filtered):
    try:
        partition = community_louvain.best_partition(G_filtered, weight='weight', resolution=1.2)
    except:
        partition = {node: 0 for node in G_filtered.nodes()}
    
    community_analysis = {}
    for node, community_id in partition.items():
        if community_id not in community_analysis:
            community_analysis[community_id] = {'nodes': [], 'categories': Counter(), 'failure_keywords': Counter()}
        community_analysis[community_id]['nodes'].append(node)
        category = G_filtered.nodes[node].get('category', '')
        if category:
            community_analysis[community_id]['categories'][category] += 1
        failure_keywords = ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure']
        for keyword in failure_keywords:
            if keyword in node.lower():
                community_analysis[community_id]['failure_keywords'][keyword] += 1
    
    return community_analysis, partition

def analyze_ego_networks(G_filtered, central_nodes=None):
    if central_nodes is None:
        central_nodes = ["electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress", "capacity fade", "lithium plating"]
    
    ego_results = {}
    for central_node in central_nodes:
        if central_node in G_filtered.nodes():
            try:
                ego_net = nx.ego_graph(G_filtered, central_node, radius=2)
                ego_results[central_node] = {
                    'node_count': ego_net.number_of_nodes(),
                    'edge_count': ego_net.number_of_edges(),
                    'density': nx.density(ego_net),
                    'average_degree': sum(dict(ego_net.degree()).values()) / ego_net.number_of_nodes(),
                    'centrality': nx.degree_centrality(ego_net).get(central_node, 0),
                    'neighbors': list(ego_net.neighbors(central_node)),
                    'subgraph_categories': Counter([ego_net.nodes[n].get('category', '') for n in ego_net.nodes()])
                }
            except:
                ego_results[central_node] = {
                    'node_count': 0, 'edge_count': 0, 'density': 0, 'average_degree': 0, 'centrality': 0,
                    'neighbors': [], 'subgraph_categories': Counter()
                }
    
    return ego_results

def find_failure_pathways(G_filtered, source_terms, target_terms):
    pathways = {}
    for source in source_terms:
        for target in target_terms:
            if source in G_filtered.nodes() and target in G_filtered.nodes():
                try:
                    path = nx.shortest_path(G_filtered, source=source, target=target, weight='weight')
                    pathways[f"{source} -> {target}"] = {'path': path, 'length': len(path)-1, 'nodes': path}
                except nx.NetworkXNoPath:
                    pathways[f"{source} -> {target}"] = {'path': None, 'length': float('inf'), 'nodes': []}
    return pathways

# -----------------------
# 4. Graph Filtering Function with Close Matches
# -----------------------
def filter_graph(
    G, min_weight=0.1, min_freq=1, selected_categories=None, selected_types=None, 
    selected_nodes=None, excluded_terms=None, min_priority_score=0, suppress_low_priority=False,
    enable_close_match=False, close_match_threshold=0.8
):
    if selected_categories is None:
        selected_categories = []
    if selected_types is None:
        selected_types = []
    if excluded_terms is None:
        excluded_terms = []
    
    G_filtered = nx.Graph()
    
    # Determine nodes to include
    valid_nodes = set()
    if selected_nodes:
        for node in selected_nodes:
            if node in G.nodes and G.nodes[node].get('priority_score', 0) >= min_priority_score:
                valid_nodes.add(node)
                valid_nodes.update(G.neighbors(node))
    else:
        for n, d in G.nodes(data=True):
            if ((not selected_categories or d.get("category", "") in selected_categories) and
                (not selected_types or d.get("type", "") in selected_types) and
                d.get("frequency", 0) >= min_freq and
                (not suppress_low_priority or d.get("priority_score", 0) >= min_priority_score)):
                valid_nodes.add(n)
    
    # Remove excluded terms
    valid_nodes = {n for n in valid_nodes if not any(ex.lower() in n.lower() for ex in excluded_terms)}
    
    # Add nodes
    for n in valid_nodes:
        G_filtered.add_node(n, **G.nodes[n])
    
    # Add edges
    for u, v, d in G.edges(data=True):
        if u in G_filtered.nodes and v in G_filtered.nodes and d.get("weight", 0) >= min_weight:
            G_filtered.add_edge(u, v, **d)
    
    # Add close-match edges
    if enable_close_match:
        nodes_list = list(G_filtered.nodes())
        for node in nodes_list:
            matches = difflib.get_close_matches(node, nodes_list, n=5, cutoff=close_match_threshold)
            for match in matches:
                if node != match and not G_filtered.has_edge(node, match):
                    G_filtered.add_edge(
                        node, match,
                        weight=0.5, type="close_match", label="close match", relationship="textual_similarity"
                    )
    
    return G_filtered

# -----------------------
# Streamlit App UI
# -----------------------
st.set_page_config(layout="wide", page_title="Battery Knowledge Graph Explorer")

st.title("🔋 Battery Knowledge Graph Explorer")

# Load data
edges_df, nodes_df = load_data()

# Create full graph
G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(row['node'], category=row.get('category', ''), type=row.get('type', ''), frequency=row.get('frequency', 0))

for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row.get('weight', 1), type=row.get('type', ''), label=row.get('label', ''))

# Calculate priority scores
priority_scores = calculate_priority_scores(G, nodes_df)
nx.set_node_attributes(G, priority_scores, 'priority_score')

# Sidebar filters
st.sidebar.header("Filter Graph")
min_weight = st.sidebar.slider("Minimum edge weight", 0.0, 1.0, 0.1, 0.05)
min_node_freq = st.sidebar.slider("Minimum node frequency", 1, int(nodes_df['frequency'].max()), 1)
selected_categories = st.sidebar.multiselect("Categories", options=nodes_df['category'].unique())
selected_types = st.sidebar.multiselect("Types", options=nodes_df['type'].unique())
selected_nodes = st.sidebar.multiselect("Select nodes", options=nodes_df['node'].unique())
excluded_terms = st.sidebar.text_area("Exclude terms (comma-separated)").split(",")
min_priority_score = st.sidebar.slider("Minimum priority score", 0.0, 1.0, 0.0, 0.01)
suppress_low_priority = st.sidebar.checkbox("Suppress low-priority nodes", value=False)

st.sidebar.subheader("🔗 Textual Similarity (Close Matches)")
enable_close_match = st.sidebar.checkbox("Enable close-match edges", value=True)
close_match_threshold = st.sidebar.slider("Close-match similarity threshold", 0.6, 1.0, 0.8, 0.05)

# Filter graph
G_filtered = filter_graph(
    G, min_weight, min_node_freq, selected_categories, selected_types, 
    selected_nodes, excluded_terms, min_priority_score, suppress_low_priority,
    enable_close_match, close_match_threshold
)

st.write(f"Filtered graph has {G_filtered.number_of_nodes()} nodes and {G_filtered.number_of_edges()} edges.")

# Show close matches for a selected node
selected_node = st.selectbox("Select a node to see close matches", options=list(G_filtered.nodes()))
if selected_node and enable_close_match:
    close_terms = difflib.get_close_matches(selected_node, list(G_filtered.nodes()), cutoff=close_match_threshold)
    close_terms = [t for t in close_terms if t != selected_node]
    if close_terms:
        st.sidebar.write("**Close Matches:**")
        for t in close_terms:
            st.sidebar.write(f"- {t}")


# -----------------------
# 5. Plotly Network Visualization
# -----------------------
def plot_network(G_plot):
    pos = nx.spring_layout(G_plot, seed=42)  # consistent layout

    edge_x = []
    edge_y = []
    edge_color = []
    edge_dash = []

    for u, v, d in G_plot.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        if d.get("type") == "close_match":
            edge_color.append("red")
            edge_dash.append("dash")
        else:
            edge_color.append("gray")
            edge_dash.append("solid")

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []

    categories = list(set(nx.get_node_attributes(G_plot, 'category').values()))
    category_colors = px.colors.qualitative.Safe
    cat_color_map = {cat: category_colors[i % len(category_colors)] for i, cat in enumerate(categories)}

    for n, d in G_plot.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_color.append(cat_color_map.get(d.get('category', ''), 'blue'))
        node_size.append(10 + 30 * d.get('priority_score', 0))
        node_text.append(f"{n}<br>Category: {d.get('category','')}"
                         f"<br>Type: {d.get('type','')}"
                         f"<br>Freq: {d.get('frequency',0)}"
                         f"<br>Priority: {d.get('priority_score',0):.2f}")

    # Plot edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Overwrite with proper colors/dash per edge
    edge_traces = []
    for u, v, d in G_plot.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(color='red' if d.get("type")=="close_match" else 'gray',
                      width=1.5,
                      dash='dash' if d.get("type")=="close_match" else 'solid'),
            hoverinfo='text',
            text=f"{u} - {v} ({d.get('label','')})",
            mode='lines'
        ))

    # Plot nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=node_size,
            line_width=2
        ),
        text=[n for n in G_plot.nodes()]
    )
    node_trace.text = node_text

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700
                    ))
    st.plotly_chart(fig, use_container_width=True)

# Call the visualization
st.subheader("🔗 Knowledge Graph Visualization")
plot_network(G_filtered)
