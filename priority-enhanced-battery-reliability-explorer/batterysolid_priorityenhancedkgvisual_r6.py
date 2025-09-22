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

# -----------------------
# 1. Setup and Data Loading with Caching
# -----------------------
DB_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Predefined key terms for battery reliability
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
        st.warning(f"Failed to load SciBERT: {str(e)}. Semantic similarity will use exact matches only.")
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
# 3. Semantic Similarity Grouping
# -----------------------
def find_similar_terms(node_terms, selected_nodes, similarity_threshold=0.7):
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
    
    # FIXED: Check if any embeddings are valid (not None)
    if selected_embeddings is None or all(emb is None for emb in selected_embeddings):
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
            community_analysis[community_id] = {
                'nodes': [],
                'categories': Counter(),
                'failure_keywords': Counter()
            }
        
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
                    'node_count': 0,
                    'edge_count': 0,
                    'density': 0,
                    'average_degree': 0,
                    'centrality': 0,
                    'neighbors': [],
                    'subgraph_categories': Counter()
                }
    
    return ego_results

def find_failure_pathways(G_filtered, source_terms, target_terms):
    pathways = {}
    for source in source_terms:
        for target in target_terms:
            if source in G_filtered.nodes() and target in G_filtered.nodes():
                try:
                    path = nx.shortest_path(G_filtered, source=source, target=target, weight='weight')
                    pathways[f"{source} -> {target}"] = {
                        'path': path,
                        'length': len(path) - 1,
                        'nodes': path
                    }
                except nx.NetworkXNoPath:
                    pathways[f"{source} -> {target}"] = {
                        'path': None,
                        'length': float('inf'),
                        'nodes': []
                    }
    
    return pathways

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

def analyze_failure_correlations(G_filtered):
    failure_terms = [n for n in G_filtered.nodes() if any(kw in n.lower() for kw in ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure'])]
    corr_matrix = np.zeros((len(failure_terms), len(failure_terms)))
    for i, term1 in enumerate(failure_terms):
        for j, term2 in enumerate(failure_terms):
            if G_filtered.has_edge(term1, term2):
                corr_matrix[i, j] = G_filtered[term1][term2].get('weight', 0)
            else:
                corr_matrix[i, j] = 0
    return corr_matrix, failure_terms

# -----------------------
# 5. Helper Functions for Export
# -----------------------
def fig_to_base64(fig, format='png'):
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def create_static_visualization(G_filtered, pos, node_colors, node_sizes):
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(G_filtered, pos, alpha=0.3, width=1)
    nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_family='sans-serif')
    plt.title("Battery Research Knowledge Graph", fontsize=16)
    plt.axis('off')
    return plt

# -----------------------
# 6. Main Application
# -----------------------
try:
    # Load data
    edges_df, nodes_df = load_data()

    # Normalize Terms
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
        }
        return replacements.get(term, term)

    nodes_df["node"] = nodes_df["node"].apply(normalize_term)
    edges_df["source"] = edges_df["source"].apply(normalize_term)
    edges_df["target"] = edges_df["target"].apply(normalize_term)

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
    - **Edges** = Relationships (thicker = stronger).  
    Click a node in the sidebar to explore its details.
    """)

    # Filters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_weight = st.slider("Min edge weight", min_value=int(edges_df["weight"].min()), max_value=int(edges_df["weight"].max()), value=5, step=1)
    with col2:
        min_node_freq = st.slider("Min node frequency", min_value=int(nodes_df["frequency"].min()), max_value=int(nodes_df["frequency"].max()), value=2, step=1)

    categories = sorted(nodes_df["category"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Filter by category", categories, default=categories)
    node_types = sorted(nodes_df["type"].dropna().unique())
    selected_types = st.sidebar.multiselect("Filter by node type", node_types, default=node_types)
    min_priority_score = st.sidebar.slider("Min priority score", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    semantic_similarity_threshold = st.sidebar.slider("Semantic similarity threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.05)

    # Node Inclusion/Exclusion
    st.sidebar.subheader("🔍 Node Inclusion/Exclusion")
    selected_nodes = st.sidebar.multiselect("Include specific nodes (optional)", options=sorted(G.nodes()), default=["crack", "fracture", "electrode cracking"] if any(n in G.nodes() for n in ["crack", "fracture", "electrode cracking"]) else [])
    excluded_terms = st.sidebar.text_input("Exclude terms (comma-separated)", value="").split(',')
    excluded_terms = [t.strip().lower() for t in excluded_terms if t.strip()]

    # Custom Related Terms
    st.sidebar.subheader("🔗 Custom Related Terms")
    related_terms_input = st.sidebar.text_input("Add related terms (comma-separated, e.g., crack,fracture)", value="crack,fracture,micro-cracking")
    related_terms = [t.strip().lower() for t in related_terms_input.split(',') if t.strip()]

    # Highlighting and Suppression
    st.sidebar.subheader("🎯 Priority Highlighting")
    highlight_priority = st.sidebar.checkbox("Highlight high-priority nodes", value=True)
    priority_threshold = st.sidebar.slider("Priority highlight threshold", 0.5, 1.0, 0.7, step=0.05)
    suppress_low_priority = st.sidebar.checkbox("Suppress low-priority nodes", value=True)

    # Label Settings
    st.sidebar.subheader("📝 Label Settings for Publications")
    show_labels = st.sidebar.checkbox("Show Node Labels", value=True)
    label_font_size = st.sidebar.slider("Label Font Size", 10, 24, 14)
    label_max_chars = st.sidebar.slider("Max Characters per Label", 10, 30, 15)
    edge_width_factor = st.sidebar.slider("Edge Width Factor", 0.1, 2.0, 0.5)

    # Filter the graph
    def filter_graph(G, min_weight, min_freq, selected_categories, selected_types, selected_nodes, excluded_terms, min_priority_score, semantic_similarity_threshold, suppress_low_priority, related_terms):
        G_filtered = nx.Graph()
        valid_nodes = set()
        
        # Find semantically similar terms
        similar_terms, similarity_scores = find_similar_terms(list(G.nodes()), selected_nodes + related_terms, semantic_similarity_threshold)
        
        # Check for missing nodes
        all_selected = selected_nodes + related_terms
        missing_nodes = [n for n in all_selected if n not in G.nodes()]
        if missing_nodes:
            st.sidebar.warning(f"Warning: The following selected nodes are not in the graph: {', '.join(missing_nodes)}")
            st.sidebar.info(f"Suggestions: Try terms like {', '.join([t for t in KEY_TERMS if t in G.nodes()][:5])}")
        
        # Include selected nodes, related terms, and their neighbors
        if all_selected:
            for node in all_selected:
                if node in G.nodes():
                    valid_nodes.add(node)
                    valid_nodes.update(G.neighbors(node))
            valid_nodes.update(similar_terms)
        else:
            for n, d in G.nodes(data=True):
                if (d.get("frequency", 0) >= min_freq and 
                    d.get("category", "") in selected_categories and
                    d.get("type", "") in selected_types and
                    (not suppress_low_priority or d.get("priority_score", 0) >= min_priority_score)):
                    valid_nodes.add(n)
        
        valid_nodes = {n for n in valid_nodes if not any(ex in n.lower() for ex in excluded_terms)}
        for n in valid_nodes:
            G_filtered.add_node(n, **G.nodes[n])
        for u, v, d in G.edges(data=True):
            if u in G_filtered.nodes and v in G_filtered.nodes and d.get("weight", 0) >= min_weight:
                G_filtered.add_edge(u, v, **d)
        
        # Display similar terms
        if similar_terms:
            st.sidebar.subheader("🔗 Semantically Similar Terms")
            for term in sorted(similar_terms):
                if term in similarity_scores:
                    matched_term, sim_score = similarity_scores[term]
                    st.sidebar.write(f"- {term} (similar to '{matched_term}', score: {sim_score:.2f})")
                else:
                    st.sidebar.write(f"- {term}")
        else:
            st.sidebar.info("No semantically similar terms found. Try lowering the similarity threshold or adding related terms.")
        
        return G_filtered

    G_filtered = filter_graph(G, min_weight, min_node_freq, selected_categories, selected_types, selected_nodes, excluded_terms, min_priority_score, semantic_similarity_threshold, suppress_low_priority, related_terms)
    st.sidebar.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # Community Detection
    if G_filtered.number_of_nodes() > 0 and G_filtered.number_of_edges() > 0:
        G_weighted = nx.Graph()
        for n, d in G_filtered.nodes(data=True):
            G_weighted.add_node(n, **d)
        for u, v, d in G_filtered.edges(data=True):
            G_weighted.add_edge(u, v, weight=d.get("weight", 1))
        partition = community_louvain.best_partition(G_weighted, weight='weight')
        comm_map = partition
        n_communities = len(set(partition.values()))
        colors = px.colors.qualitative.Set3[:n_communities]
        node_colors = [colors[comm_map[node] % len(colors)] for node in G_filtered.nodes()]
    else:
        node_colors = ["lightblue"] * G_filtered.number_of_nodes()

    # Visualization
    if G_filtered.number_of_nodes() > 0:
        pos = nx.spring_layout(G_filtered, k=1, iterations=100, seed=42, weight='weight')
        priority_scores = [G_filtered.nodes[node].get('priority_score', 0) for node in G_filtered.nodes()]
        min_size, max_size = 15, 60
        if max(priority_scores) > min(priority_scores):
            node_sizes = [min_size + (max_size - min_size) * (score - min(priority_scores)) / (max(priority_scores) - min(priority_scores)) for score in priority_scores]
        else:
            node_sizes = [30] * len(priority_scores)
        
        edge_x, edge_y, edge_weights = [], [], []
        for edge in G_filtered.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G_filtered.edges[edge].get('weight', 1))
        
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            edge_widths = [0.5 + 4.5 * edge_width_factor * (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 2 * edge_width_factor for w in edge_weights]
        else:
            edge_widths = [2 * edge_width_factor] * len(edge_weights)
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
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
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Battery Research Knowledge Graph',
            title_font_size=20,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            annotations=[dict(text="Interactive graph. Hover for details, click on nodes to explore connections.", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002, font=dict(size=10))],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Export Static Visualization
        st.sidebar.subheader("📤 Export for Publication")
        if st.sidebar.button("Generate Static Visualization"):
            static_fig = create_static_visualization(G_filtered, pos, node_colors, node_sizes)
            img_str = fig_to_base64(static_fig)
            href = f'<a href="data:image/png;base64,{img_str}" download="knowledge_graph.png">Download PNG</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
            img_str_svg = fig_to_base64(static_fig, format='svg')
            href_svg = f'<a href="data:image/svg+xml;base64,{img_str_svg}" download="knowledge_graph.svg">Download SVG</a>'
            st.sidebar.markdown(href_svg, unsafe_allow_html=True)
            st.sidebar.success("Static visualization generated. Click the links above to download.")
    else:
        st.warning("No nodes match the current filter criteria. Try adjusting the filters, lowering the similarity threshold, or adding related terms.")

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
                    w = G_filtered.edges[selected_node, n].get('weight', 1)
                    rel_type = G_filtered.edges[selected_node, n].get('type', "")
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
    st.sidebar.subheader("💾 Export Data")
    if st.sidebar.button("Export Filtered Graph as CSV"):
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
        st.sidebar.download_button(label="Download Nodes CSV", data=nodes_export.to_csv(index=False), file_name="filtered_nodes.csv", mime="text/csv")
        st.sidebar.download_button(label="Download Edges CSV", data=edges_export.to_csv(index=False), file_name="filtered_edges.csv", mime="text/csv")

    # Failure Analysis Dashboard
    st.header("🔍 Advanced Failure Mechanism Analysis")
    with st.expander("📖 Post-Processing Guide: How to Study Battery Failure Mechanisms"):
        st.markdown("""
        ## Post-Processing Techniques for Battery Failure Analysis
        ### 1. Centrality Analysis
        - **Purpose**: Identify the most important failure mechanisms in your knowledge graph
        - **How to use**: Select "Centrality Analysis" from the dropdown. Look for nodes with high degree, betweenness, or eigenvector centrality. Focus on terms like "cracking", "degradation", "fatigue", "failure".
        ### 2. Community Detection
        - **Purpose**: Discover groups of related failure concepts
        - **How to use**: Select "Community Detection" from the dropdown. Examine each community for dominant failure types. Look for communities focused on specific failure modes.
        ### 3. Ego Network Analysis
        - **Purpose**: Study specific failure mechanisms in detail
        - **How to use**: Select "Ego Network Analysis" from the dropdown. Choose key failure terms like "electrode cracking" or "SEI formation". Examine their immediate connections and network properties.
        ### 4. Pathway Analysis
        - **Purpose**: Find connections between different failure mechanisms
        - **How to use**: Select "Pathway Analysis" from the dropdown. Choose source and target failure mechanisms. Analyze the shortest paths between them.
        ### 5. Temporal Analysis
        - **Purpose**: Analyze how failure concepts evolve over time (if 'year' data is available)
        - **How to use**: Select "Temporal Analysis" from the dropdown. Examine trends in failure concepts.
        ### 6. Correlation Analysis
        - **Purpose**: Identify relationships between failure mechanisms
        - **How to use**: Select "Correlation Analysis" from the dropdown. Examine the heatmap for strong correlations. Focus on relationships between mechanical and chemical degradation.
        ### Research Questions to Explore:
        - How are different cracking mechanisms (electrode, SEI, particle) related?
        - What connects mechanical degradation to capacity fade?
        - Which failure mechanisms act as bridges between different degradation modes?
        - How do failure communities correspond to different battery components?
        """)

    analysis_type = st.selectbox("Select Analysis Type", ["Centrality Analysis", "Community Detection", "Ego Network Analysis", "Pathway Analysis", "Temporal Analysis", "Correlation Analysis"])
    
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
        central_nodes = st.multiselect("Select central nodes for ego network analysis", options=sorted(G_filtered.nodes()), default=["electrode cracking", "SEI formation", "capacity fade"] if any(n in G_filtered.nodes() for n in ["electrode cracking", "SEI formation", "capacity fade"]) else [])
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
            target_terms = st.multiselect("Target mechanisms", options=sorted(G_filtered.nodes()), default=["capacity fade", "SEI formation"] if any(n in G_filtered.nodes() for n in ["capacity fade", "SEI formation"]) else [])
        if source_terms and target_terms:
            pathways = find_failure_pathways(G_filtered, source_terms, target_terms)
            for pathway_name, data in pathways.items():
                if data['path']:
                    st.write(f"**{pathway_name}** (Length: {data['length']})")
                    st.write(" → ".join(data['nodes']))
                else:
                    st.write(f"**{pathway_name}**: No path found")
    
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

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.text("Detailed error information:")
    st.text(traceback.format_exc())
