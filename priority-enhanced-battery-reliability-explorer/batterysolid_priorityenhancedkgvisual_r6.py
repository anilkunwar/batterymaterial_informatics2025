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

# Predefined key terms for battery reliability
KEY_TERMS = [
    "electrode cracking", "sei formation", "cyclic mechanical damage", "diffusion-induced stress",
    "micro-cracking", "electrolyte degradation", "capacity fade", "lithium plating", "thermal runaway",
    "mechanical degradation", "cycle life", "lithium", "electrode", "crack", "fracture", "battery",
    "particles", "cathode", "mechanical", "cycles", "electrolyte", "degradation", "surface", "capacity",
    "cycling", "stress", "diffusion", "solid electrolyte interphase", "impendence", "degrades the battery capacity",
    "cycling degradation", "calendar degradation", "complex cycling damage", "chemo-mechanical degradation mechanisms",
    "microcrack formation", "active particles", "differential degradation mechanisms", "sol swing", "lithiation",
    "electrochemical performance", "mechanical integrity", "battery safety", "coupled mechanical-chemical degradation",
    "physics-based models", "predict degradation mechanisms", "electrode side reactions", "capacity loss",
    "mechanical degradation", "particle versus sei cracking", "degradation models", "predict degradation"
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
        st.warning(f"Failed to load SciBERT: {str(e)}. Semantic similarity will use exact matches.")
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

# Precompute embeddings for key terms
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
    
    # Normalize node names before checking for missing terms
    nodes_df['node'] = nodes_df['node'].apply(normalize_term)
    edges_df['source'] = edges_df['source'].apply(normalize_term)
    edges_df['target'] = edges_df['target'].apply(normalize_term)
    
    # Check for missing key terms
    existing_nodes = set(nodes_df['node'].str.lower())
    missing_terms = [t for t in KEY_TERMS if t.lower() not in existing_nodes]
    if missing_terms:
        st.warning(f"Adding {len(missing_terms)} missing key terms to nodes data: {', '.join(missing_terms[:5])}{', ...' if len(missing_terms) > 5 else ''}")
        new_nodes = []
        for term in missing_terms:
            new_nodes.append({
                'node': term,
                'type': 'Unknown',
                'category': 'Failure Mechanism',
                'frequency': 1,
                'unit': 'None'
            })
        nodes_df = pd.concat([nodes_df, pd.DataFrame(new_nodes)], ignore_index=True)
    
    # Debug: Log nodes and missing terms
    st.write("Debug: Nodes in graph after loading:", sorted(nodes_df['node'].tolist()[:10]))
    st.write("Debug: Missing KEY_TERMS after normalization:", [t for t in KEY_TERMS if t.lower() not in nodes_df['node'].str.lower().tolist()])
    
    return edges_df, nodes_df

# -----------------------
# 2. Priority Score Calculation
# -----------------------
@st.cache_data
def calculate_priority_scores(_G, nodes_df):
    max_freq = nodes_df['frequency'].max() if nodes_df['frequency'].max() > 0 else 1
    nodes_df['norm_frequency'] = nodes_df['frequency'] / max_freq
    
    degree_centrality = nx.degree_centrality(_G)
    betweenness_centrality = nx.betweenness_centrality(_G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(_G, max_iter=1000)
    except:
        eigenvector_centrality = {node: 0 for node in _G.nodes()}
    
    node_terms = nodes_df['node'].tolist()
    term_embeddings = get_scibert_embedding(node_terms)
    term_embeddings_dict = dict(zip(node_terms, term_embeddings))
    
    semantic_scores = {}
    for node in _G.nodes():
        emb = term_embeddings_dict.get(node)
        if emb is None:
            semantic_scores[node] = 0
        else:
            similarities = [cosine_similarity([emb], [kt_emb])[0][0] for kt_emb in KEY_TERMS_EMBEDDINGS]
            semantic_scores[node] = max(similarities, default=0)
    
    priority_scores = {}
    for node in _G.nodes():
        priority_scores[node] = (
            0.4 * nodes_df[nodes_df['node'] == node]['norm_frequency'].iloc[0] +
            0.3 * degree_centrality.get(node, 0) +
            0.2 * betweenness_centrality.get(node, 0) +
            0.1 * semantic_scores.get(node, 0)
        )
    
    return priority_scores

# -----------------------
# 3. Semantic Similarity Grouping
# -----------------------
def find_similar_terms(node_terms, selected_nodes, similarity_threshold=0.6):
    if not selected_nodes:
        return set(), {}
    
    similar_terms = set(selected_nodes)
    similarity_scores = {}
    
    if scibert_tokenizer is None or scibert_model is None:
        st.warning("SciBERT not available. Using exact matches for key terms.")
        for term in node_terms:
            for selected in selected_nodes:
                if selected in term.lower() or term.lower() in selected:
                    similar_terms.add(term)
                    similarity_scores[term] = (selected, 1.0)
        return similar_terms, similarity_scores
    
    selected_embeddings = get_scibert_embedding(selected_nodes)
    if not selected_embeddings or all(emb is None for emb in selected_embeddings):
        st.warning("No valid embeddings for selected nodes. Including selected nodes only.")
        return set(selected_nodes), {}
    
    selected_embeddings_dict = dict(zip(selected_nodes, selected_embeddings))
    term_embeddings = get_scibert_embedding(node_terms)
    term_embeddings_dict = dict(zip(node_terms, term_embeddings))
    
    for term in node_terms:
        if term in similar_terms:
            continue
        term_emb = term_embeddings_dict.get(term)
        if term_emb is not None:
            max_sim = 0
            matched_term = None
            for selected_term, selected_emb in selected_embeddings_dict.items():
                if selected_emb is not None:
                    try:
                        sim = cosine_similarity([term_emb], [selected_emb])[0][0]
                        if sim > max_sim:
                            max_sim = sim
                            matched_term = selected_term
                        if sim > similarity_threshold:
                            similar_terms.add(term)
                            similarity_scores[term] = (matched_term, sim)
                            break
                    except Exception as e:
                        st.warning(f"Error computing similarity for {term}: {str(e)}")
                        continue
    
    return similar_terms, similarity_scores

# -----------------------
# 4. Failure Analysis Functions
# -----------------------
@st.cache_data
def analyze_failure_centrality(_G_filtered, focus_terms=None):
    if focus_terms is None:
        focus_terms = ["crack", "fracture", "degradation", "fatigue", "damage", "failure", "mechanical", "cycling", "capacity fade", "sei"]
    
    degree_centrality = nx.degree_centrality(_G_filtered)
    betweenness_centrality = nx.betweenness_centrality(_G_filtered)
    closeness_centrality = nx.closeness_centrality(_G_filtered)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(_G_filtered, max_iter=1000)
    except:
        eigenvector_centrality = {node: 0 for node in _G_filtered.nodes()}
    
    centrality_results = []
    for node in _G_filtered.nodes():
        if any(term in node.lower() for term in focus_terms):
            centrality_results.append({
                'node': node,
                'degree': degree_centrality.get(node, 0),
                'betweenness': betweenness_centrality.get(node, 0),
                'closeness': closeness_centrality.get(node, 0),
                'eigenvector': eigenvector_centrality.get(node, 0),
                'category': _G_filtered.nodes[node].get('category', ''),
                'type': _G_filtered.nodes[node].get('type', '')
            })
    
    return pd.DataFrame(centrality_results)

@st.cache_data
def detect_failure_communities(_G_filtered, resolution=1.2):
    try:
        partition = community_louvain.best_partition(_G_filtered, weight='weight', resolution=resolution)
    except:
        partition = {node: 0 for node in _G_filtered.nodes()}
    
    community_analysis = {}
    for node, community_id in partition.items():
        if community_id not in community_analysis:
            community_analysis[community_id] = {
                'nodes': [],
                'categories': Counter(),
                'failure_keywords': Counter()
            }
        
        community_analysis[community_id]['nodes'].append(node)
        category = _G_filtered.nodes[node].get('category', '')
        if category:
            community_analysis[community_id]['categories'][category] += 1
        
        failure_keywords = ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure']
        for keyword in failure_keywords:
            if keyword in node.lower():
                community_analysis[community_id]['failure_keywords'][keyword] += 1
    
    return community_analysis, partition

@st.cache_data
def analyze_ego_networks(_G_filtered, central_nodes=None):
    if central_nodes is None:
        central_nodes = ["electrode cracking", "sei formation", "cyclic mechanical damage", "diffusion-induced stress", "capacity fade", "lithium plating"]
    
    ego_results = {}
    for central_node in central_nodes:
        if central_node in _G_filtered.nodes():
            try:
                ego_net = nx.ego_graph(_G_filtered, central_node, radius=2)
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
                    'node_count': 0,
                    'edge_count': 0,
                    'density': 0,
                    'average_degree': 0,
                    'centrality': 0,
                    'neighbors': [],
                    'subgraph_categories': Counter()
                }
    
    return ego_results

@st.cache_data
def find_failure_pathways(_G_filtered, source_terms, target_terms):
    pathways = {}
    for source in source_terms:
        for target in target_terms:
            if source in _G_filtered.nodes() and target in _G_filtered.nodes():
                try:
                    path = nx.shortest_path(_G_filtered, source=source, target=target, weight='weight')
                    weight = sum(_G_filtered[u][v].get('weight', 1) for u, v in zip(path[:-1], path[1:]))
                    pathways[f"{source} -> {target}"] = {
                        'path': path,
                        'length': len(path) - 1,
                        'weight': weight,
                        'nodes': path
                    }
                except nx.NetworkXNoPath:
                    pathways[f"{source} -> {target}"] = {
                        'path': None,
                        'length': float('inf'),
                        'weight': 0,
                        'nodes': []
                    }
    
    return pathways

@st.cache_data
def analyze_temporal_patterns(nodes_df, edges_df, time_column='year'):
    if time_column in nodes_df.columns:
        time_periods = sorted(nodes_df[time_column].dropna().unique())
        temporal_analysis = {}
        for period in time_periods:
            period_nodes = nodes_df[nodes_df[time_column] == period]
            temporal_analysis[period] = {
                'total_concepts': len(period_nodes),
                'failure_concepts': len([n for n in period_nodes['node'] if any(kw in n.lower() for kw in ['crack', 'fracture', 'degrad', 'fatigue', 'damage'])]),
                'top_concepts': period_nodes.nlargest(5, 'frequency')['node'].tolist()
            }
        return temporal_analysis
    else:
        return {"error": "Time column not found in data"}

@st.cache_data
def analyze_failure_correlations(_G_filtered):
    failure_terms = [n for n in _G_filtered.nodes() if any(kw in n.lower() for kw in ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure'])]
    corr_matrix = np.zeros((len(failure_terms), len(failure_terms)))
    for i, term1 in enumerate(failure_terms):
        for j, term2 in enumerate(failure_terms):
            if _G_filtered.has_edge(term1, term2):
                corr_matrix[i, j] = _G_filtered[term1][term2].get('weight', 0)
            else:
                corr_matrix[i, j] = 0
    return corr_matrix, failure_terms

@st.cache_data
def analyze_clusters(_G_filtered):
    clusters = list(nx.algorithms.community.girvan_newman(_G_filtered))
    cluster_summary = {}
    for i, cluster in enumerate(clusters[0]):
        cluster_summary[i] = {
            'nodes': list(cluster),
            'categories': Counter([_G_filtered.nodes[n].get('category', '') for n in cluster]),
            'failure_keywords': Counter([kw for n in cluster for kw in ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure'] if kw in n.lower()])
        }
    return cluster_summary

# -----------------------
# 5. Helper Functions
# -----------------------
def normalize_term(term: str) -> str:
    if not isinstance(term, str):
        return ""
    term = term.lower().strip()
    replacements = {
        "batteries": "battery",
        "materials": "material",
        "mater": "material",
        "lithium ions": "lithium ion",
        "lithium ion(s)": "lithium ion",
        "fatigue": "fatigue",
        "cracking": "crack",
        "microcracking": "micro-cracking",
        "sei": "sei formation",
        "electrode crack": "electrode cracking",
        "solid electrolyte interphase": "sei formation",
        "capacity loss": "capacity fade",
        "mechanical degradation": "mechanical degradation"
    }
    return replacements.get(term, term)

def fig_to_base64(fig, format='png', dpi=300):
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def create_static_visualization(_G_filtered, pos, node_colors, node_sizes, font_size=8):
    plt.figure(figsize=(16, 12))
    edge_weights = [_G_filtered[u][v].get('weight', 1) for u, v in _G_filtered.edges()]
    max_weight = max(edge_weights, default=1)
    edge_widths = [0.5 + 2.5 * (w / max_weight) for w in edge_weights]
    
    nx.draw_networkx_edges(_G_filtered, pos, alpha=0.3, width=edge_widths)
    nx.draw_networkx_nodes(_G_filtered, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_labels(_G_filtered, pos, font_size=font_size, font_family='sans-serif')
    
    plt.title("Battery Research Knowledge Graph", fontsize=16)
    plt.axis('off')
    return plt

# -----------------------
# 6. Main Application
# -----------------------
try:
    # Load data
    edges_df, nodes_df = load_data()

    # Build Initial Graph
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["node"], type=row["type"], category=row["category"], frequency=row["frequency"], unit=row.get("unit", "None"), similarity_score=row.get("similarity_score", 0))
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"], type=row["type"], label=row["label"], relationship=row.get("relationship", ""), strength=row.get("strength", 0))

    # Calculate Priority Scores and Add to Graph
    priority_scores = calculate_priority_scores(G, nodes_df)
    for node in G.nodes():
        G.nodes[node]['priority_score'] = priority_scores.get(node, 0)
    nodes_df['priority_score'] = nodes_df['node'].apply(lambda x: priority_scores.get(x, 0))

    # Sidebar Controls
    st.title("🔋 Battery Research Knowledge Graph Explorer")
    st.markdown("""
    Explore key concepts in **battery mechanical degradation research**.  
    - **Nodes** = Terms (colored by cluster).  
    - **Size** = Priority score.  
    - **Edges** = Relationships (thicker = stronger, dashed = close match).  
    Click a node in the sidebar to explore its details.
    """)

    # Reset Filters Button
    st.sidebar.subheader("🔄 Reset Filters")
    if st.sidebar.button("Reset All Filters"):
        st.session_state.clear()

    # Filters
    with st.sidebar.expander("🔧 Filter Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.slider("Min edge weight", min_value=int(edges_df["weight"].min()), max_value=int(edges_df["weight"].max()), value=5, step=1, help="Filter edges by minimum weight")
        with col2:
            min_node_freq = st.slider("Min node frequency", min_value=int(nodes_df["frequency"].min()), max_value=int(nodes_df["frequency"].max()), value=2, step=1, help="Filter nodes by minimum frequency")
        
        categories = sorted(nodes_df["category"].dropna().unique())
        selected_categories = st.multiselect("Filter by category", categories, default=categories, help="Select categories to include")
        node_types = sorted(nodes_df["type"].dropna().unique())
        selected_types = st.multiselect("Filter by node type", node_types, default=node_types, help="Select node types to include")
        min_priority_score = st.slider("Min priority score", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Filter nodes by minimum priority score")
        semantic_similarity_threshold = st.slider("Semantic similarity threshold", min_value=0.5, max_value=1.0, value=0.6, step=0.05, help="Threshold for grouping related terms")

    # Node Inclusion/Exclusion
    with st.sidebar.expander("🔍 Node Inclusion/Exclusion", expanded=True):
        selected_nodes = st.multiselect("Include specific nodes (optional)", options=sorted(G.nodes()), default=["crack", "fracture", "electrode cracking"] if any(n in G.nodes() for n in ["crack", "fracture", "electrode cracking"]) else [], help="Select nodes to include in the graph")
        excluded_terms = st.text_input("Exclude terms (comma-separated)", value="", help="Enter terms to exclude (e.g., battery,material)").split(',')
        excluded_terms = [t.strip().lower() for t in excluded_terms if t.strip()]
        related_terms_input = st.text_input("Add related terms (comma-separated, e.g., crack,fracture)", value="crack,fracture,micro-cracking", help="Group related terms for semantic similarity")
        related_terms = [t.strip().lower() for t in related_terms_input.split(',') if t.strip()]

    # Textual Similarity (Close Matches)
    with st.sidebar.expander("🔗 Textual Similarity (Close Matches)", expanded=True):
        enable_close_match = st.checkbox("Enable close-match edges", value=True, help="Add edges for textually similar terms")
        close_match_threshold = st.slider("Close-match similarity threshold", min_value=0.6, max_value=1.0, value=0.8, step=0.05, help="Threshold for textual similarity using difflib")

    # Highlighting and Visualization Settings
    with st.sidebar.expander("🎯 Visualization Settings", expanded=True):
        highlight_priority = st.checkbox("Highlight high-priority nodes", value=True, help="Use star symbols for high-priority nodes")
        priority_threshold = st.slider("Priority highlight threshold", 0.5, 1.0, 0.7, step=0.05, help="Threshold for highlighting nodes")
        suppress_low_priority = st.checkbox("Suppress low-priority nodes", value=False, help="Hide nodes below min priority score")
        show_labels = st.checkbox("Show Node Labels", value=True, help="Display node labels on the graph")
        label_font_size = st.slider("Label Font Size", 10, 24, 14, help="Font size for node labels")
        label_max_chars = st.slider("Max Characters per Label", 10, 30, 15, help="Maximum characters for node labels")
        edge_width_factor = st.slider("Edge Width Factor", 0.1, 2.0, 0.5, help="Scale edge widths")
        show_edge_labels = st.checkbox("Show Edge Labels (High-Weight)", value=False, help="Display labels for edges with high weights")

    # Filter the graph
    def filter_graph(_G, min_weight, min_node_freq, selected_categories, selected_types, selected_nodes, excluded_terms, min_priority_score, suppress_low_priority, related_terms, semantic_similarity_threshold, enable_close_match, close_match_threshold):
        G_filtered = nx.Graph()
        valid_nodes = set()
        
        # Normalize selected_nodes and related_terms with difflib
        cleaned_selected_nodes = []
        term_mappings = []
        for term in selected_nodes:
            if term in _G.nodes():
                cleaned_selected_nodes.append(term)
            else:
                close = difflib.get_close_matches(term, _G.nodes(), n=1, cutoff=0.7)
                if close:
                    cleaned_selected_nodes.append(close[0])
                    term_mappings.append(f"'{term}' → '{close[0]}'")
        
        cleaned_related_terms = []
        for term in related_terms:
            if term in _G.nodes():
                cleaned_related_terms.append(term)
            else:
                close = difflib.get_close_matches(term, _G.nodes(), n=1, cutoff=0.7)
                if close:
                    cleaned_related_terms.append(close[0])
                    term_mappings.append(f"'{term}' → '{close[0]}'")
        
        # Display term mappings
        if term_mappings:
            st.sidebar.subheader("🔗 Term Mappings")
            for mapping in term_mappings:
                st.sidebar.write(f"- {mapping}")
        
        selected_nodes = list(set(cleaned_selected_nodes))
        related_terms = list(set(cleaned_related_terms))
        
        # Find semantically similar terms
        similar_terms, similarity_scores = find_similar_terms(list(_G.nodes()), selected_nodes + related_terms, semantic_similarity_threshold)
        
        # Check for missing nodes
        all_selected = selected_nodes + related_terms
        missing_nodes = [n for n in all_selected if n not in _G.nodes()]
        if missing_nodes:
            st.sidebar.warning(f"Warning: The following selected nodes are not in the graph: {', '.join(missing_nodes)}")
            st.sidebar.info(f"Suggestions: Try terms like {', '.join([t for t in KEY_TERMS if t in _G.nodes()][:5])}")
        
        # Include selected nodes, related terms, and their neighbors
        if all_selected:
            for node in all_selected:
                if node in _G.nodes():
                    valid_nodes.add(node)
                    valid_nodes.update(_G.neighbors(node))
            valid_nodes.update(similar_terms)
        else:
            for n, d in _G.nodes(data=True):
                if (d.get("frequency", 0) >= min_node_freq and 
                    d.get("category", "") in selected_categories and
                    d.get("type", "") in selected_types and
                    (not suppress_low_priority or d.get("priority_score", 0) >= min_priority_score)):
                    valid_nodes.add(n)
        
        valid_nodes = {n for n in valid_nodes if not any(ex in n.lower() for ex in excluded_terms)}
        for n in valid_nodes:
            G_filtered.add_node(n, **_G.nodes[n])
        for u, v, d in _G.edges(data=True):
            if u in G_filtered.nodes and v in G_filtered.nodes and d.get("weight", 0) >= min_weight:
                G_filtered.add_edge(u, v, **d)
        
        # Add close-match edges if enabled
        if enable_close_match:
            nodes_list = list(G_filtered.nodes())
            for node in nodes_list:
                close_matches = difflib.get_close_matches(node, nodes_list, n=5, cutoff=close_match_threshold)
                for match in close_matches:
                    if node != match and not G_filtered.has_edge(node, match):
                        G_filtered.add_edge(
                            node,
                            match,
                            weight=0.5,
                            type="close_match",
                            label="close match",
                            relationship="textual_similarity",
                            strength=0.5
                        )
        
        # Display similar terms
        if similar_terms:
            st.sidebar.subheader("🔗 Semantically Similar Terms")
            for term in sorted(similar_terms):
                if term in similarity_scores:
                    matched_term, sim_score = similarity_scores[term]
                    st.sidebar.write(f"- {term} (similar to '{matched_term}', score: {sim_score:.2f})")
                else:
                    st.sidebar.write(f"- {term}")
        
        # Display close matches for selected node
        if selected_nodes and enable_close_match:
            for selected_node in selected_nodes:
                close_terms = difflib.get_close_matches(selected_node, nodes_list, n=5, cutoff=close_match_threshold)
                close_terms = [t for t in close_terms if t != selected_node]
                if close_terms:
                    st.sidebar.subheader(f"Close Matches for {selected_node}")
                    for t in close_terms:
                        st.sidebar.write(f"- {t}")
        
        return G_filtered

    G_filtered = filter_graph(G, min_weight, min_node_freq, selected_categories, selected_types, selected_nodes, excluded_terms, min_priority_score, suppress_low_priority, related_terms, semantic_similarity_threshold, enable_close_match, close_match_threshold)
    st.sidebar.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # Community Detection
    if G_filtered.number_of_nodes() > 0 and G_filtered.number_of_edges() > 0:
        G_weighted = nx.Graph()
        for n, d in G_filtered.nodes(data=True):
            G_weighted.add_node(n, **d)
        for u, v, d in G_filtered.edges(data=True):
            G_weighted.add_edge(u, v, weight=d.get('weight', 1))
        
        partition = community_louvain.best_partition(G_weighted, weight='weight', resolution=1.2)
        comm_map = partition
        n_communities = len(set(partition.values()))
        colors = px.colors.qualitative.Set3[:n_communities]
        node_colors = [colors[comm_map[node] % len(colors)] for node in G_filtered.nodes()]
    else:
        node_colors = ["lightblue"] * G_filtered.number_of_nodes()

    # Visualization
    if G_filtered.number_of_nodes() > 0:
        pos = nx.spring_layout(G_filtered, k=1.5, iterations=100, seed=42, weight='weight')
        
        priority_scores = [G_filtered.nodes[node].get('priority_score', 0) for node in G_filtered.nodes()]
        min_size, max_size = 20, 80
        if max(priority_scores) > min(priority_scores):
            node_sizes = [min_size + (max_size - min_size) * (score - min(priority_scores)) / (max(priority_scores) - min(priority_scores)) for score in priority_scores]
        else:
            node_sizes = [40] * len(priority_scores)
        
        # Separate regular and close-match edges
        regular_edge_x, regular_edge_y, regular_edge_weights, regular_edge_labels = [], [], [], []
        close_edge_x, close_edge_y, close_edge_weights, close_edge_labels = [], [], [], []
        for edge in G_filtered.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G_filtered.edges[edge].get('weight', 1)
            edge_type = G_filtered.edges[edge].get('type', '')
            edge_label = G_filtered.edges[edge].get('label', '') if weight > min_weight * 1.5 else ''
            if edge_type == "close_match":
                close_edge_x.extend([x0, x1, None])
                close_edge_y.extend([y0, y1, None])
                close_edge_weights.append(weight)
                close_edge_labels.append(edge_label)
            else:
                regular_edge_x.extend([x0, x1, None])
                regular_edge_y.extend([y0, y1, None])
                regular_edge_weights.append(weight)
                regular_edge_labels.append(edge_label)
        
        # Scale edge widths
        all_edge_weights = regular_edge_weights + close_edge_weights
        if all_edge_weights:
            max_weight = max(all_edge_weights, default=1)
            min_weight = min(all_edge_weights, default=0)
            regular_edge_widths = [0.5 + 4.5 * edge_width_factor * (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 2 * edge_width_factor for w in regular_edge_weights]
            close_edge_widths = [0.5 + 4.5 * edge_width_factor * (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 2 * edge_width_factor for w in close_edge_weights]
        else:
            regular_edge_widths = [2 * edge_width_factor] * len(regular_edge_weights)
            close_edge_widths = [2 * edge_width_factor] * len(close_edge_weights)
        
        # Create edge traces
        regular_edge_trace = go.Scatter(
            x=regular_edge_x, y=regular_edge_y,
            line=dict(width=regular_edge_widths, color='#888'),
            hoverinfo='text' if show_edge_labels else 'none',
            hovertext=regular_edge_labels if show_edge_labels else None,
            mode='lines'
        )
        
        close_edge_trace = go.Scatter(
            x=close_edge_x, y=close_edge_y,
            line=dict(width=close_edge_widths, color='#888', dash='dash'),
            hoverinfo='text' if show_edge_labels else 'none',
            hovertext=close_edge_labels if show_edge_labels else None,
            mode='lines'
        )
        
        node_x, node_y, node_text, node_labels, node_symbols = [], [], [], [], []
        for node in G_filtered.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = G_filtered.nodes[node]
            node_text.append(
                f"{node}<br>Category: {node_data.get('category', 'N/A')}<br>Type: {node_data.get('type', 'N/A')}<br>Frequency: {node_data.get('frequency', 'N/A')}<br>Unit: {node_data.get('unit', 'N/A')}<br>Priority Score: {node_data.get('priority_score', 0):.3f}"
            )
            node_labels.append(node[:label_max_chars] + "..." if len(node) > label_max_chars else node)
            node_symbols.append('star' if highlight_priority and node_data.get('priority_score', 0) > priority_threshold else 'circle')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            text=node_labels if show_labels else None,
            textfont=dict(size=label_font_size, color='black'),
            textposition='middle center',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(showscale=False, color=node_colors, size=node_sizes, symbol=node_symbols, line_width=2)
        )
        
        fig = go.Figure(data=[regular_edge_trace, close_edge_trace, node_trace])
        fig.update_layout(
            title='Battery Research Knowledge Graph',
            title_font_size=20,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            annotations=[dict(
                text="Interactive graph. Hover for details, click on nodes to explore connections. Dashed lines indicate close-match edges.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                font=dict(size=10)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Export Static Visualization
        with st.sidebar.expander("📤 Export for Publication", expanded=False):
            export_dpi = st.slider("Export DPI", 100, 600, 300, step=50, help="Resolution for exported images")
            if st.button("Generate Static Visualization"):
                static_fig = create_static_visualization(G_filtered, pos, node_colors, node_sizes, font_size=label_font_size)
                img_str = fig_to_base64(static_fig, dpi=export_dpi)
                href = f'<a href="data:image/png;base64,{img_str}" download="knowledge_graph.png">Download PNG</a>'
                st.markdown(href, unsafe_allow_html=True)
                img_str_svg = fig_to_base64(static_fig, format='svg', dpi=export_dpi)
                href_svg = f'<a href="data:image/svg+xml;base64,{img_str_svg}" download="knowledge_graph.svg">Download SVG</a>'
                st.markdown(href_svg, unsafe_allow_html=True)
                st.success("Static visualization generated. Click the links above to download.")
    else:
        st.warning("No nodes match the current filter criteria. Try lowering the filters, similarity threshold, or adding related terms.")

    # Hub Nodes and Analysis
    if G_filtered.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G_filtered)
        betweenness_centrality = nx.betweenness_centrality(G_filtered)
        closeness_centrality = nx.closeness_centrality(G_filtered)
        
        centrality_df = pd.DataFrame({
            'node': list(degree_centrality.keys()),
            'degree': list(degree_centrality.values()),
            'betweenness': list(betweenness_centrality.values()),
            'closeness': list(closeness_centrality.values())
        })
        centrality_df['category'] = centrality_df['node'].apply(lambda x: G_filtered.nodes[x].get('category', 'N/A'))
        centrality_df['type'] = centrality_df['node'].apply(lambda x: G_filtered.nodes[x].get('type', 'N/A'))
        centrality_df['frequency'] = centrality_df['node'].apply(lambda x: G_filtered.nodes[x].get('frequency', 0))
        
        st.sidebar.subheader("🔑 Top Hub Nodes")
        top_hubs = centrality_df.nlargest(10, 'degree')[['node', 'degree']]
        for _, row in top_hubs.iterrows():
            st.sidebar.write(f"**{row['node']}** ({row['degree']:.3f})")
        
        st.sidebar.subheader("🔍 Explore Node Details")
        selected_node = st.sidebar.selectbox("Choose a node", sorted(G_filtered.nodes()))
        if selected_node:
            node_data = G_filtered.nodes[selected_node]
            st.sidebar.markdown(f"### {selected_node.title()}")
            st.sidebar.write(f"**Category:** {node_data.get('category', 'N/A')}")
            st.sidebar.write(f"**Type:** {node_data.get('type', 'N/A')}")
            st.sidebar.write(f"**Frequency:** {node_data.get('frequency', 'N/A')}")
            st.sidebar.write(f"**Unit:** {node_data.get('unit', 'N/A')}")
            st.sidebar.write(f"**Similarity Score:** {node_data.get('similarity_score', 'N/A')}")
            st.sidebar.write(f"**Priority Score:** {node_data.get('priority_score', 0):.3f}")
            
            node_centrality = centrality_df[centrality_df['node'] == selected_node]
            if not node_centrality.empty:
                st.sidebar.write(f"**Degree Centrality:** {node_centrality['degree'].values[0]:.3f}")
                st.sidebar.write(f"**Betweenness Centrality:** {node_centrality['betweenness'].values[0]:.3f}")
                st.sidebar.write(f"**Closeness Centrality:** {node_centrality['closeness'].values[0]:.3f}")
            
            neighbors = list(G_filtered.neighbors(selected_node))
            if neighbors:
                st.sidebar.write("**Connected Terms:**")
                for n in neighbors:
                    w = G_filtered.edges[selected_node, n].get("weight", 1)
                    rel_type = G_filtered.edges[selected_node, n].get("type", "")
                    st.sidebar.write(f"- {n} ({rel_type}, weight: {w})")
            else:
                st.sidebar.write("No connected terms above current filter threshold.")
        
        st.subheader("📊 Graph Analytics")
        col1, col2 = st.columns(2)
        with col1:
            category_counts = nodes_df['category'].value_counts()
            fig_cat = px.pie(values=category_counts.values, names=category_counts.index, title="Node Distribution by Category")
            st.plotly_chart(fig_cat, use_container_width=True)
        with col2:
            top_nodes = nodes_df.nlargest(10, 'priority_score')[['node', 'priority_score']]
            fig_nodes = px.bar(top_nodes, x='priority_score', y='node', orientation='h', title="Top Nodes by Priority Score")
            st.plotly_chart(fig_nodes, use_container_width=True)
        
        edge_type_counts = edges_df['type'].value_counts()
        fig_edge = px.bar(x=edge_type_counts.index, y=edge_type_counts.values, title="Edge Type Distribution", labels={'x': 'Edge Type', 'y': 'Count'})
        st.plotly_chart(fig_edge, use_container_width=True)
        
        if G_filtered.number_of_nodes() > 0:
            st.subheader("👥 Community Analysis")
            community_summary = {}
            for node, comm_id in comm_map.items():
                if comm_id not in community_summary:
                    community_summary[comm_id] = []
                community_summary[comm_id].append(node)
            for comm_id, nodes in community_summary.items():
                with st.expander(f"Community {comm_id} ({len(nodes)} nodes)"):
                    st.write("**Nodes:** " + ", ".join(nodes[:10]))
                    if len(nodes) > 10:
                        st.write(f"*... and {len(nodes) - 10} more*")
                    categories_in_comm = [G_filtered.nodes[node].get('category', '') for node in nodes]
                    category_counter = Counter(categories_in_comm)
                    st.write("**Top categories:**")
                    for cat, count in category_counter.most_common(3):
                        st.write(f"- {cat}: {count} nodes")

    # Data Export
    with st.sidebar.expander("💾 Export Data", expanded=False):
        if st.button("Export Filtered Graph as CSV"):
            filtered_nodes = []
            for node, data in G_filtered.nodes(data=True):
                filtered_nodes.append({
                    'node': node,
                    'type': data.get('type', ''),
                    'category': data.get('category', ''),
                    'frequency': data.get('frequency', 0),
                    'unit': data.get('unit', ''),
                    'similarity_score': data.get('similarity_score', 0),
                    'priority_score': data.get('priority_score', 0)
                })
            filtered_edges = []
            for u, v, data in G_filtered.edges(data=True):
                filtered_edges.append({
                    'source': u,
                    'target': v,
                    'weight': data.get('weight', 0),
                    'type': data.get('type', ''),
                    'label': data.get('label', ''),
                    'relationship': data.get('relationship', ''),
                    'strength': data.get('strength', 0)
                })
            nodes_export = pd.DataFrame(filtered_nodes)
            edges_export = pd.DataFrame(filtered_edges)
            st.download_button(label="Download Nodes CSV", data=nodes_export.to_csv(index=False), file_name="filtered_nodes.csv", mime="text/csv")
            st.download_button(label="Download Edges CSV", data=edges_export.to_csv(index=False), file_name="filtered_edges.csv", mime="text/csv")

    # Failure Analysis Dashboard
    st.header("🔍 Advanced Failure Mechanism Analysis")
    with st.expander("📖 Post-Processing Guide: How to Study Battery Failure Mechanisms"):
        st.markdown("""
        ## Post-Processing Techniques for Battery Failure Analysis
        ### 1. Centrality Analysis
        - **Purpose**: Identify the most important failure mechanisms in your knowledge graph
        - **How to use**: Select "Centrality Analysis" to identify nodes with high degree, betweenness, or eigenvector centrality. Focus on terms like "crack", "fracture", "degradation".
        ### 2. Community Detection
        - **Purpose**: Discover groups of related failure concepts
        - **How to use**: Select "Community Detection" to examine communities for dominant failure types.
        ### 3. Ego Network Analysis
        - **Purpose**: Study specific failure mechanisms in detail
        - **How to use**: Select "Ego Network Analysis" and choose key failure terms like "electrode cracking" or "sei formation".
        ### 4. Pathway Analysis
        - **Purpose**: Find connections between different failure mechanisms
        - **How to use**: Select "Pathway Analysis" to analyze shortest paths between source and target mechanisms.
        ### 5. Temporal Analysis
        - **Purpose**: Analyze how failure concepts evolve over time (if 'year' data is available)
        - **How to use**: Select "Temporal Analysis" to examine trends in failure concepts.
        ### 6. Correlation Analysis
        - **Purpose**: Identify relationships between failure mechanisms
        - **How to use**: Select "Correlation Analysis" to examine the heatmap for strong correlations.
        ### 7. Cluster Analysis
        - **Purpose**: Identify tightly connected groups of failure mechanisms
        - **How to use**: Select "Cluster Analysis" to explore clusters and their properties.
        ### Research Questions to Explore:
        - How are different cracking mechanisms (electrode, sei, particle) related?
        - What connects mechanical degradation to capacity fade?
        - Which failure mechanisms act as bridges between different degradation modes?
        - How do failure communities correspond to different battery components?
        """)

    analysis_type = st.selectbox("Select Analysis Type", [
        "Centrality Analysis", "Community Detection", "Ego Network Analysis",
        "Pathway Analysis", "Temporal Analysis", "Correlation Analysis", "Cluster Analysis"
    ])

    if analysis_type == "Centrality Analysis":
        st.subheader("Centrality of Failure-Related Terms")
        failure_df = analyze_failure_centrality(G_filtered)
        if not failure_df.empty:
            fig = px.scatter(failure_df, x='degree', y='betweenness', color='category', size='eigenvector', hover_data=['node', 'closeness'], title="Centrality of Failure Mechanisms")
            st.plotly_chart(fig)
            st.subheader("Top Failure Mechanisms")
            for centrality_measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
                top_nodes = failure_df.nlargest(5, centrality_measure)
                st.write(f"**By {centrality_measure.title()}:**")
                for _, row in top_nodes.iterrows():
                    st.write(f"- {row['node']} ({row[centrality_measure]:.3f})")

    elif analysis_type == "Community Detection":
        st.subheader("Failure Mechanism Communities")
        communities, partition = detect_failure_communities(G_filtered)
        for comm_id, data in communities.items():
            with st.expander(f"Community {comm_id} ({len(data['nodes'])} nodes)"):
                st.write("**Top Categories:**")
                for category, count in data['categories'].most_common(3):
                    st.write(f"- {category}: {count} nodes")
                st.write("**Failure Keywords:**")
                for keyword, count in data['failure_keywords'].most_common():
                    st.write(f"- {keyword}: {count} occurrences")
                st.write("**Representative Nodes:**")
                st.write(", ".join(data['nodes'][:10]))

    elif analysis_type == "Ego Network Analysis":
        st.subheader("Ego Network Analysis")
        central_nodes = st.multiselect("Select central nodes for ego network analysis", options=sorted(G_filtered.nodes()), default=["electrode cracking", "sei formation", "capacity fade"] if any(n in G_filtered.nodes() for n in ["electrode cracking", "sei formation", "capacity fade"]) else [])
        if central_nodes:
            ego_results = analyze_ego_networks(G_filtered, central_nodes)
            for node, data in ego_results.items():
                with st.expander(f"Ego Network: {node}"):
                    st.write(f"**Network Size:** {data['node_count']} nodes, {data['edge_count']} edges")
                    st.write(f"**Density:** {data['density']:.3f}")
                    st.write(f"**Average Degree:** {data['average_degree']:.2f}")
                    st.write("**Immediate Neighbors:**")
                    st.write(", ".join(data['neighbors']))

    elif analysis_type == "Pathway Analysis":
        st.subheader("Pathways Between Failure Mechanisms")
        col1, col2 = st.columns(2)
        with col1:
            source_terms = st.multiselect("Source mechanisms", options=sorted(G_filtered.nodes()), default=["electrode cracking", "micro-cracking"] if any(n in G_filtered.nodes() for n in ["electrode cracking", "micro-cracking"]) else [])
        with col2:
            target_terms = st.multiselect("Target mechanisms", options=sorted(G_filtered.nodes()), default=["capacity fade", "sei formation"] if any(n in G_filtered.nodes() for n in ["capacity fade", "sei formation"]) else [])
        if source_terms and target_terms:
            pathways = find_failure_pathways(G_filtered, source_terms, target_terms)
            for pathway_name, data in pathways.items():
                if data['path']:
                    st.write(f"**{pathway_name}** (Length: {data['length']}, Weight: {data['weight']:.2f})")
                    st.write(" → ".join(data['nodes']))
                else:
                    st.write(f"**{pathway_name}**: No path found")
            # Visualize pathways
            if pathways:
                path_nodes = set()
                for data in pathways.values():
                    if data['nodes']:
                        path_nodes.update(data['nodes'])
                G_path = G_filtered.subgraph(path_nodes)
                pos_path = nx.spring_layout(G_path, k=1.5, iterations=100, seed=42)
                path_edge_x, path_edge_y = [], []
                for edge in G_path.edges():
                    x0, y0 = pos_path[edge[0]]
                    x1, y1 = pos_path[edge[1]]
                    path_edge_x.extend([x0, x1, None])
                    path_edge_y.extend([y0, y1, None])
                path_edge_trace = go.Scatter(x=path_edge_x, y=path_edge_y, line=dict(width=2, color='red'), hoverinfo='none', mode='lines')
                path_node_x, path_node_y, path_node_text = [], [], []
                for node in G_path.nodes():
                    x, y = pos_path[node]
                    path_node_x.append(x)
                    path_node_y.append(y)
                    path_node_text.append(node)
                path_node_trace = go.Scatter(x=path_node_x, y=path_node_y, mode='markers+text', text=path_node_text, textfont=dict(size=label_font_size), hoverinfo='text', marker=dict(size=20, color='red'))
                fig_path = go.Figure(data=[path_edge_trace, path_node_trace])
                fig_path.update_layout(title="Pathway Visualization", showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                st.plotly_chart(fig_path)

    elif analysis_type == "Temporal Analysis" and 'year' in nodes_df.columns:
        st.subheader("Temporal Evolution of Failure Concepts")
        temporal_data = analyze_temporal_patterns(nodes_df, edges_df)
        years = list(temporal_data.keys())
        failure_counts = [data['failure_concepts'] for data in temporal_data.values()]
        total_counts = [data['total_concepts'] for data in temporal_data.values()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=failure_counts, name='Failure Concepts', line=dict(width=4)))
        fig.add_trace(go.Scatter(x=years, y=total_counts, name='All Concepts', line=dict(width=2, dash='dash')))
        fig.update_layout(title="Evolution of Failure Concepts Over Time", xaxis_title="Year", yaxis_title="Number of Concepts")
        st.plotly_chart(fig)

    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Between Failure Mechanisms")
        corr_matrix, failure_terms = analyze_failure_correlations(G_filtered)
        fig = px.imshow(corr_matrix, x=failure_terms, y=failure_terms, title="Correlation Between Failure Mechanisms", color_continuous_scale='Reds')
        fig.update_layout(xaxis_title="Failure Mechanism", yaxis_title="Failure Mechanism")
        st.plotly_chart(fig)
        st.subheader("Strongest Correlations")
        correlations = []
        for i, term1 in enumerate(failure_terms):
            for j, term2 in enumerate(failure_terms):
                if i < j and corr_matrix[i, j] > 0:
                    correlations.append((term1, term2, corr_matrix[i, j]))
        correlations.sort(key=lambda x: x[2], reverse=True)
        for term1, term2, strength in correlations[:10]:
            st.write(f"**{term1}** ↔ **{term2}** (strength: {strength:.2f})")
        st.download_button(label="Download Correlation Matrix", data=pd.DataFrame(corr_matrix, index=failure_terms, columns=failure_terms).to_csv(), file_name="correlation_matrix.csv", mime="text/csv")

    elif analysis_type == "Cluster Analysis":
        st.subheader("Cluster Analysis")
        clusters = analyze_clusters(G_filtered)
        for cluster_id, data in clusters.items():
            with st.expander(f"Cluster {cluster_id} ({len(data['nodes'])} nodes)"):
                st.write("**Nodes:**")
                st.write(", ".join(data['nodes'][:10]))
                if len(data['nodes']) > 10:
                    st.write(f"*... and {len(data['nodes']) - 10} more*")
                st.write("**Top Categories:**")
                for category, count in data['categories'].most_common(3):
                    st.write(f"- {category}: {count} nodes")
                st.write("**Failure Keywords:**")
                for keyword, count in data['failure_keywords'].most_common():
                    st.write(f"- {keyword}: {count} occurrences")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.text("Detailed error information:")
    st.text(traceback.format_exc())
