#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT BATTERY DEGRADATION KNOWLEDGE EXPLORER
===================================================
HOLISTIC ARCHITECTURE WITH PHYSICS-GROUNDED LLM REASONING

Core Innovations:
1. Natural language query → FULL sidebar auto-update (CoreShellGPT pattern)
2. Physics-augmented priority scoring with battery equations
3. LLM insights constrained by real graph metrics + physics formulas
4. Perfect harmony between parsed intent and executed math
5. All original visualizations preserved and enhanced

Author: Advanced Energy Informatics Platform
"""

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
import json
import re
import hashlib
from datetime import datetime
from collections import OrderedDict
import warnings
import time

# NEW IMPORTS FOR GPT/QWEN INTEGRATION
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("transformers not installed. GPT/Qwen features will be disabled.")

warnings.filterwarnings('ignore')

# -----------------------
# 1. GLOBAL CONFIGURATION & CONSTANTS
# -----------------------
DB_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Battery degradation physics equations (for LLM grounding)
PHYSICS_EQUATIONS = {
    "diffusion_stress": r"σ = \frac{E \Omega \Delta c}{1 - \nu}  (E: modulus, Ω: partial molar volume, Δc: concentration gradient)",
    "sei_growth": r"δ_SEI ∝ √t  (parabolic growth kinetics)",
    "chemo_mechanical": r"LAM ∝ \int \sigma \, dN  (stress-integrated over cycles)",
    "lithium_plating": r"Risk ↑ when V < V_plate or T < 0°C",
    "capacity_fade": r"Q_loss = Q_0 - \int I(t) dt  (integrated current loss)",
    "crack_propagation": r"\frac{da}{dN} = C(\Delta K)^m  (Paris law for fatigue)"
}

# Key battery physics terms for semantic boosting
PHYSICS_TERMS = [
    "diffusion-induced stress", "SEI formation", "chemo-mechanical coupling",
    "lithium plating", "thermal runaway", "capacity fade", "mechanical degradation",
    "stress concentration", "fracture toughness", "crack propagation",
    "activation energy", "reaction rate", "overpotential", "solid electrolyte interphase",
    "cycle life", "calendar aging", "parasitic reactions", "gas evolution"
]

# Precomputed embeddings for physics terms (will be populated)
PHYSICS_TERMS_EMBEDDINGS = []

# Predefined key terms for battery reliability (from original)
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

# -----------------------
# 2. SCIBERT LOADER & EMBEDDING UTILITIES
# -----------------------
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
    """Get normalized SciBERT embeddings for text(s)"""
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

# Precompute embeddings for physics terms
PHYSICS_TERMS_EMBEDDINGS = get_scibert_embedding(PHYSICS_TERMS)
PHYSICS_TERMS_EMBEDDINGS = [emb for emb in PHYSICS_TERMS_EMBEDDINGS if emb is not None]

# -----------------------
# 3. DATA LOADING
# -----------------------
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
# 4. PHYSICS-AUGMENTED PRIORITY SCORE CALCULATION
# -----------------------
def calculate_priority_scores(G, nodes_df, physics_boost_weight=0.15):
    """
    Calculate priority scores with physics-based augmentation.
    
    P(u) = w_f * norm_freq + w_d * C_D(u) + w_b * C_B(u) + w_s * S(u) + w_p * B_p(u)
    
    where:
    - w_f = 0.35 (frequency weight)
    - w_d = 0.25 (degree centrality weight)
    - w_b = 0.20 (betweenness centrality weight)
    - w_s = 0.10 (semantic relevance weight)
    - w_p = physics_boost_weight (default 0.15)
    - B_p(u) = physics boost term (cosine similarity to battery physics terms)
    """
    # Normalize frequency
    max_freq = nodes_df['frequency'].max() if nodes_df['frequency'].max() > 0 else 1
    nodes_df['norm_frequency'] = nodes_df['frequency'] / max_freq
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_centrality = {node: 0 for node in G.nodes()}
    
    # Calculate semantic relevance to key terms (original)
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
    
    # NEW: Calculate physics boost term
    physics_boost_scores = {}
    for node in G.nodes():
        emb = term_embeddings_dict.get(node)
        if emb is None or not PHYSICS_TERMS_EMBEDDINGS:
            physics_boost_scores[node] = 0
        else:
            similarities = [cosine_similarity([emb], [pt_emb])[0][0] for pt_emb in PHYSICS_TERMS_EMBEDDINGS if pt_emb is not None]
            physics_boost_scores[node] = max(similarities, default=0)
    
    # Combine scores with new weighting scheme
    w_f, w_d, w_b, w_s, w_p = 0.35, 0.25, 0.20, 0.10, physics_boost_weight
    # Renormalize to sum to 1.0
    total = w_f + w_d + w_b + w_s + w_p
    w_f, w_d, w_b, w_s, w_p = w_f/total, w_d/total, w_b/total, w_s/total, w_p/total
    
    priority_scores = {}
    for node in G.nodes():
        priority_scores[node] = (
            w_f * nodes_df[nodes_df['node'] == node]['norm_frequency'].iloc[0] +
            w_d * degree_centrality.get(node, 0) +
            w_b * betweenness_centrality.get(node, 0) +
            w_s * semantic_scores.get(node, 0) +
            w_p * physics_boost_scores.get(node, 0)
        )
    
    return priority_scores

# -----------------------
# 5. FAILURE ANALYSIS FUNCTIONS (Original, preserved)
# -----------------------
def analyze_failure_centrality(G_filtered, focus_terms=None):
    """
    Analyze centrality of terms related to failure mechanisms
    """
    if focus_terms is None:
        focus_terms = [
            "crack", "fracture", "degradation", "fatigue", "damage", 
            "failure", "mechanical", "cycling", "capacity fade", "SEI"
        ]
    
    # Calculate centrality measures
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
    """
    Detect communities specifically focused on failure mechanisms
    """
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
                'failure_keywords': Counter(),
                'physics_terms': Counter()  # NEW: track physics term presence
            }
        
        community_analysis[community_id]['nodes'].append(node)
        category = G_filtered.nodes[node].get('category', '')
        if category:
            community_analysis[community_id]['categories'][category] += 1
        
        # Check for failure-related keywords
        failure_keywords = ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure']
        for keyword in failure_keywords:
            if keyword in node.lower():
                community_analysis[community_id]['failure_keywords'][keyword] += 1
        
        # NEW: Check for physics-related keywords
        for term in PHYSICS_TERMS:
            if term.lower() in node.lower():
                community_analysis[community_id]['physics_terms'][term] += 1
    
    return community_analysis, partition

def analyze_ego_networks(G_filtered, central_nodes=None):
    """
    Analyze ego networks around specific failure mechanisms
    """
    if central_nodes is None:
        central_nodes = [
            "electrode cracking", "SEI formation", "cyclic mechanical damage",
            "diffusion-induced stress", "capacity fade", "lithium plating"
        ]
    
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
                    'subgraph_categories': Counter([ego_net.nodes[n].get('category', '') for n in ego_net.nodes()]),
                    'physics_terms': [n for n in ego_net.nodes() if any(term in n.lower() for term in PHYSICS_TERMS)]
                }
            except:
                ego_results[central_node] = {
                    'node_count': 0,
                    'edge_count': 0,
                    'density': 0,
                    'average_degree': 0,
                    'centrality': 0,
                    'neighbors': [],
                    'subgraph_categories': Counter(),
                    'physics_terms': []
                }
    
    return ego_results

def find_failure_pathways(G_filtered, source_terms, target_terms, require_physics_nodes=False):
    """
    Find shortest paths between different types of failure mechanisms
    """
    pathways = {}
    
    for source in source_terms:
        for target in target_terms:
            if source in G_filtered.nodes() and target in G_filtered.nodes():
                try:
                    # Try to find all shortest paths
                    paths = list(nx.all_shortest_paths(G_filtered, source=source, target=target, weight='weight'))
                    
                    if require_physics_nodes:
                        # Filter paths that contain at least one physics term
                        filtered_paths = []
                        for path in paths:
                            if any(any(term in node.lower() for term in PHYSICS_TERMS) for node in path):
                                filtered_paths.append(path)
                        paths = filtered_paths
                    
                    if paths:
                        # Take the first path
                        path = paths[0]
                        pathways[f"{source} -> {target}"] = {
                            'path': path,
                            'length': len(path) - 1,
                            'nodes': path,
                            'num_paths': len(paths),
                            'contains_physics': any(any(term in node.lower() for term in PHYSICS_TERMS) for node in path)
                        }
                    else:
                        pathways[f"{source} -> {target}"] = {
                            'path': None,
                            'length': float('inf'),
                            'nodes': [],
                            'num_paths': 0,
                            'contains_physics': False
                        }
                except nx.NetworkXNoPath:
                    pathways[f"{source} -> {target}"] = {
                        'path': None,
                        'length': float('inf'),
                        'nodes': [],
                        'num_paths': 0,
                        'contains_physics': False
                    }
    
    return pathways

def analyze_temporal_patterns(nodes_df, edges_df, time_column='year'):
    """
    Analyze how failure concepts evolve over time
    """
    if time_column in nodes_df.columns:
        time_periods = sorted(nodes_df[time_column].dropna().unique())
        temporal_analysis = {}
        
        for period in time_periods:
            period_nodes = nodes_df[nodes_df[time_column] == period]
            temporal_analysis[period] = {
                'total_concepts': len(period_nodes),
                'failure_concepts': len([n for n in period_nodes['node'] 
                                       if any(kw in n.lower() for kw in 
                                       ['crack', 'fracture', 'degrad', 'fatigue', 'damage'])]),
                'physics_concepts': len([n for n in period_nodes['node']
                                       if any(term in n.lower() for term in PHYSICS_TERMS)]),
                'top_concepts': period_nodes.nlargest(5, 'frequency')['node'].tolist()
            }
        
        return temporal_analysis
    else:
        return {"error": "Time column not found in data"}

def analyze_failure_correlations(G_filtered):
    """
    Analyze correlations between different failure mechanisms
    """
    failure_terms = [n for n in G_filtered.nodes() 
                    if any(kw in n.lower() for kw in 
                    ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure'])]
    
    corr_matrix = np.zeros((len(failure_terms), len(failure_terms)))
    
    for i, term1 in enumerate(failure_terms):
        for j, term2 in enumerate(failure_terms):
            if G_filtered.has_edge(term1, term2):
                corr_matrix[i, j] = G_filtered[term1][term2].get('weight', 0)
            else:
                corr_matrix[i, j] = 0
    
    return corr_matrix, failure_terms

# -----------------------
# 6. EXPORT HELPER FUNCTIONS
# -----------------------
def fig_to_base64(fig, format='png'):
    """Convert a matplotlib figure to a base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=200)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def create_static_visualization(G_filtered, pos, node_colors, node_sizes):
    """Create a static matplotlib visualization for export"""
    plt.figure(figsize=(16, 12))
    
    # Draw edges
    nx.draw_networkx_edges(G_filtered, pos, alpha=0.3, width=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_filtered, pos, 
                          node_color=node_colors, 
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G_filtered, pos, 
                           font_size=8, 
                           font_family='sans-serif')
    
    plt.title("Battery Research Knowledge Graph", fontsize=16)
    plt.axis('off')
    
    return plt

# Filter the graph with user-based nodes and exclusions
def filter_graph(G, min_weight, min_freq, selected_categories, selected_types, selected_nodes, excluded_terms, min_priority_score, suppress_low_priority):
    G_filtered = nx.Graph()
    valid_nodes = set()
    
    # Determine nodes to include
    if selected_nodes:
        for node in selected_nodes:
            if node in G.nodes() and G.nodes[node].get('priority_score', 0) >= min_priority_score:
                valid_nodes.add(node)
                valid_nodes.update(G.neighbors(node))
    else:
        for n, d in G.nodes(data=True):
            if (d.get("frequency", 0) >= min_freq and 
                d.get("category", "") in selected_categories and
                d.get("type", "") in selected_types and
                (not suppress_low_priority or d.get("priority_score", 0) >= min_priority_score)):
                valid_nodes.add(n)
    
    # Remove excluded terms
    valid_nodes = {n for n in valid_nodes if not any(ex in n.lower() for ex in excluded_terms)}
    
    # Add nodes
    for n in valid_nodes:
        G_filtered.add_node(n, **G.nodes[n])
    
    # Add edges
    for u, v, d in G.edges(data=True):
        if (u in G_filtered.nodes and v in G_filtered.nodes and 
            d.get("weight", 0) >= min_weight):
            G_filtered.add_edge(u, v, **d)
    
    return G_filtered

# -----------------------
# 7. UNIFIED LLM LOADER (CoreShellGPT pattern)
# -----------------------
@st.cache_resource(show_spinner="Loading LLM for intelligent analysis...")
def load_llm(backend):
    """Load the selected LLM model with caching"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, backend
    
    try:
        if "GPT-2" in backend:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            if "Qwen2-0.5B" in backend:
                model_name = "Qwen/Qwen2-0.5B-Instruct"
            else:  # Qwen2.5-0.5B
                model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype="auto", 
                device_map="auto", 
                trust_remote_code=True
            )
        
        model.eval()
        return tokenizer, model, backend
    except Exception as e:
        st.warning(f"Failed to load {backend}: {str(e)}")
        return None, None, backend

# -----------------------
# 8. INTELLIGENT NLP PARSER (Physics-aware)
# -----------------------
class BatteryNLParser:
    """
    Extract knowledge graph parameters from natural language
    Now with physics term recognition and advanced confidence merging
    """
    def __init__(self):
        self.defaults = {
            'min_weight': 10,
            'min_freq': 5,
            'min_priority_score': 0.2,
            'priority_threshold': 0.7,
            'edge_width_factor': 0.5,
            'label_font_size': 16,
            'label_max_chars': 15,
            'selected_categories': ["Crack and Fracture", "Deformation", "Degradation", "Fatigue"],
            'selected_types': ["category", "term"],
            'selected_nodes': ["electrode cracking", "SEI formation", "capacity fade"],
            'excluded_terms': ["battery", "material"],
            'suppress_low_priority': False,
            'highlight_priority': True,
            'show_labels': True,
            'analysis_type': 'Centrality Analysis',
            'focus_terms': ['crack', 'fracture', 'degradation', 'fatigue', 'damage'],
            'source_terms': ['electrode cracking'],
            'target_terms': ['capacity fade'],
            'central_nodes': ['electrode cracking', 'SEI formation', 'capacity fade'],
            'time_column': 'year',
            'physics_boost_weight': 0.15,  # NEW: physics term weight
            'require_physics_in_pathways': False,  # NEW: filter pathways
            'min_physics_similarity': 0.5  # NEW: threshold for physics relevance
        }
        
        # Physics term dictionary for recognition
        self.physics_keywords = {
            'stress': ['stress', 'strain', 'mechanical', 'elastic', 'plastic'],
            'diffusion': ['diffusion', 'transport', 'migration', 'concentration'],
            'thermal': ['thermal', 'temperature', 'heat', 'runaway'],
            'electrochemical': ['electrochemical', 'reaction', 'kinetics', 'overpotential'],
            'fracture': ['crack', 'fracture', 'fatigue', 'damage'],
            'sei': ['SEI', 'interphase', 'solid electrolyte', 'passivation'],
            'plating': ['plating', 'dendrite', 'lithium metal'],
            'degradation': ['degradation', 'fade', 'aging', 'deterioration']
        }
        
        # Regex patterns (same as before)
        self.patterns = {
            'min_weight': [
                r'min(?:imum)?\s*edge\s*weight\s*(?:of|>=|>|=)?\s*(\d+)',
                r'edge\s*weight\s*[>=]\s*(\d+)'
            ],
            'min_freq': [
                r'min(?:imum)?\s*(?:node)?\s*frequency\s*(?:of|>=|>|=)?\s*(\d+)',
                r'frequency\s*[>=]\s*(\d+)'
            ],
            'min_priority_score': [
                r'priority\s*score\s*(?:of|>=|>|=)?\s*(\d*\.?\d+)',
                r'min(?:imum)?\s*priority\s*(\d*\.?\d+)'
            ],
            'physics_boost_weight': [
                r'physics\s*boost\s*(?:of|>=|>|=)?\s*(\d*\.?\d+)',
                r'physics\s*weight\s*(\d*\.?\d+)'
            ],
            'require_physics_in_pathways': [
                r'require\s*physics',
                r'only\s*physics\s*paths?',
                r'filter\s*by\s*physics'
            ]
        }
        
        # Analysis type mapping
        self.analysis_map = {
            'centrality': 'Centrality Analysis',
            'central': 'Centrality Analysis',
            'hub': 'Centrality Analysis',
            'important': 'Centrality Analysis',
            'community': 'Community Detection',
            'cluster': 'Community Detection',
            'ego': 'Ego Network Analysis',
            'neighborhood': 'Ego Network Analysis',
            'pathway': 'Pathway Analysis',
            'path': 'Pathway Analysis',
            'connection': 'Pathway Analysis',
            'temporal': 'Temporal Analysis',
            'time': 'Temporal Analysis',
            'evolution': 'Temporal Analysis',
            'correlation': 'Correlation Analysis',
            'correlate': 'Correlation Analysis',
            'relationship': 'Correlation Analysis'
        }

    def parse_regex(self, text):
        """Fast regex-based parsing with physics term recognition"""
        if not text:
            return self.defaults.copy()
        
        params = self.defaults.copy()
        text_lower = text.lower()
        
        # Extract numerical parameters
        for param, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        if param == 'require_physics_in_pathways':
                            params[param] = True
                        elif param == 'excluded_terms':
                            terms = [t.strip() for t in match.group(1).split(',')]
                            params[param] = terms
                        else:
                            params[param] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Detect physics terms in query
        detected_physics = []
        for category, keywords in self.physics_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_physics.append(category)
        
        if detected_physics:
            params['detected_physics'] = detected_physics
        
        # Determine analysis type
        for key, analysis in self.analysis_map.items():
            if key in text_lower:
                params['analysis_type'] = analysis
                break
        
        # Extract specific terms for analysis
        if params['analysis_type'] == 'Pathway Analysis':
            source_match = re.search(r'from\s+([a-zA-Z\s\-]+?)\s+to', text_lower)
            target_match = re.search(r'to\s+([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)', text_lower)
            
            if source_match:
                params['source_terms'] = [t.strip() for t in source_match.group(1).split(',')]
            if target_match:
                params['target_terms'] = [t.strip() for t in target_match.group(1).split(',')]
            
            # Check if physics requirement is mentioned
            if 'physics' in text_lower and ('path' in text_lower or 'route' in text_lower):
                params['require_physics_in_pathways'] = True
        
        # Extract focus terms
        for pattern in [r'focus\s*on\s*([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)',
                        r'analyze\s*([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)']:
            match = re.search(pattern, text_lower)
            if match:
                params['focus_terms'] = [t.strip() for t in match.group(1).split(',')]
                break
        
        return params

    def _extract_json_robust(self, generated):
        """Extract and repair JSON from generated text"""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, generated, re.DOTALL)
        
        if not match:
            match = re.search(r'\{.*\}', generated, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            try:
                return json.loads(json_str)
            except:
                try:
                    json_str = re.sub(r',\s*}', '}', json_str)
                    return json.loads(json_str)
                except:
                    pass
        return None

    def parse_with_llm(self, text, tokenizer, model, regex_params=None, temperature=0.1):
        """Use LLM to extract parameters with physics awareness"""
        if not text or tokenizer is None or model is None:
            return regex_params if regex_params else self.defaults.copy()
        
        system = """You are an expert in battery degradation analysis with deep physics knowledge. 
Extract parameters for knowledge graph exploration from the user query. 
Recognize and prioritize physics terms (stress, diffusion, SEI, plating, etc.)."""

        examples = """
Examples with physics context:
1. "Show me pathways from electrode cracking to capacity fade that involve diffusion-induced stress"
   → {"analysis_type": "Pathway Analysis", "source_terms": ["electrode cracking"], "target_terms": ["capacity fade"], "focus_terms": ["diffusion-induced stress"], "require_physics_in_pathways": true}

2. "Analyze communities related to chemo-mechanical degradation, boost physics terms to 0.2 weight"
   → {"analysis_type": "Community Detection", "focus_terms": ["chemo-mechanical", "degradation"], "physics_boost_weight": 0.2, "selected_categories": ["Crack and Fracture", "Degradation"]}

3. "Show ego network around stress concentration and fracture, min physics similarity 0.6"
   → {"analysis_type": "Ego Network Analysis", "central_nodes": ["stress concentration", "fracture"], "min_physics_similarity": 0.6}

4. "Find correlations between thermal runaway and mechanical degradation, require physics terms"
   → {"analysis_type": "Correlation Analysis", "focus_terms": ["thermal runaway", "mechanical degradation"], "require_physics_in_pathways": true}

5. "How have SEI formation and lithium plating concepts evolved? Focus on physics terms"
   → {"analysis_type": "Temporal Analysis", "focus_terms": ["SEI formation", "lithium plating"], "time_column": "year"}

Battery physics equations for context:
• Diffusion stress: σ = E·Ω·Δc/(1-ν)
• SEI growth: δ_SEI ∝ √t
• Chemo-mechanical: LAM ∝ ∫σ dN
• Plating risk: ↑ when V < V_plate or T < 0°C
"""
        
        user = f"""{examples}
Text: "{text}"
Preliminary regex: {json.dumps(regex_params, default=str) if regex_params else 'None'}

Output ONLY a JSON object with keys from: {list(self.defaults.keys())}
Include physics_boost_weight, require_physics_in_pathways, min_physics_similarity when mentioned.

JSON:"""

        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default)')
        
        try:
            if "Qwen" in backend:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{system}\n\n{user}\n"

            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=400,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            params = self._extract_json_robust(generated)
            
            if params:
                # Merge with defaults
                for key in self.defaults:
                    if key not in params:
                        params[key] = self.defaults[key]
                
                # Clip values
                if 'min_physics_similarity' in params:
                    params['min_physics_similarity'] = np.clip(float(params['min_physics_similarity']), 0, 1)
                if 'physics_boost_weight' in params:
                    params['physics_boost_weight'] = np.clip(float(params['physics_boost_weight']), 0, 0.5)
                
                return params
            return regex_params if regex_params else self.defaults.copy()
        except Exception as e:
            return regex_params if regex_params else self.defaults.copy()

    def hybrid_parse(self, text, tokenizer=None, model=None, use_ensemble=False, ensemble_runs=3):
        """Combine regex and LLM with confidence-based merging"""
        regex_params = self.parse_regex(text)
        
        if tokenizer is None or model is None:
            return regex_params
        
        # Compute regex confidence
        regex_conf = {}
        for key in self.defaults:
            if key in regex_params:
                if isinstance(regex_params[key], (int, float)) and isinstance(self.defaults[key], (int, float)):
                    regex_conf[key] = 1.0 if abs(regex_params[key] - self.defaults[key]) > 1e-6 else 0.0
                elif isinstance(regex_params[key], list) and isinstance(self.defaults[key], list):
                    regex_conf[key] = 1.0 if set(regex_params[key]) != set(self.defaults[key]) else 0.0
                else:
                    regex_conf[key] = 1.0 if regex_params[key] != self.defaults[key] else 0.0
            else:
                regex_conf[key] = 0.0
        
        # Get LLM parameters
        if use_ensemble:
            all_llm = []
            for _ in range(ensemble_runs):
                llm = self.parse_with_llm(text, tokenizer, model, regex_params, temperature=0.2)
                all_llm.append(llm)
            
            # Ensemble voting
            llm_params = {}
            for key in self.defaults:
                values = [p[key] for p in all_llm]
                if isinstance(self.defaults[key], bool):
                    llm_params[key] = max(set(values), key=values.count)
                elif isinstance(self.defaults[key], (int, float)):
                    numeric = [v for v in values if isinstance(v, (int, float))]
                    llm_params[key] = float(np.mean(numeric)) if numeric else self.defaults[key]
                elif isinstance(self.defaults[key], list):
                    flat = []
                    for vlist in values:
                        if isinstance(vlist, list):
                            flat.extend(vlist)
                    llm_params[key] = list(set(flat))[:10] if flat else self.defaults[key]
                else:
                    llm_params[key] = max(set(values), key=values.count)
        else:
            llm_params = self.parse_with_llm(text, tokenizer, model, regex_params)
        
        # LLM confidence
        llm_conf = {}
        for key in self.defaults:
            if key in llm_params:
                if isinstance(llm_params[key], (int, float)) and isinstance(self.defaults[key], (int, float)):
                    llm_conf[key] = 0.7 if abs(llm_params[key] - self.defaults[key]) > 1e-6 else 0.3
                elif isinstance(llm_params[key], list) and isinstance(self.defaults[key], list):
                    llm_conf[key] = 0.7 if set(llm_params[key]) != set(self.defaults[key]) else 0.3
                else:
                    llm_conf[key] = 0.7 if llm_params[key] != self.defaults[key] else 0.3
            else:
                llm_conf[key] = 0.0
        
        # Merge with confidence
        final_params = {}
        for key in self.defaults:
            if regex_conf.get(key, 0) >= llm_conf.get(key, 0):
                final_params[key] = regex_params.get(key, self.defaults[key])
            else:
                final_params[key] = llm_params.get(key, self.defaults[key])
        
        return final_params

# -----------------------
# 9. PHYSICS-GROUNDED INSIGHT GENERATOR
# -----------------------
class DegradationInsightGenerator:
    """Generate intelligent insights using physics equations and graph metrics"""
    
    _cache = OrderedDict()
    _max_cache_size = 20
    
    @staticmethod
    def generate_insights(analysis_results, analysis_type, user_query, relevance_score, 
                          parsed_params, graph_stats, tokenizer, model):
        """Generate insights with physics equations and actual graph numbers"""
        
        cache_key = hashlib.md5(f"{analysis_type}_{user_query}_{relevance_score:.3f}_{str(parsed_params)[:100]}".encode()).hexdigest()
        
        if cache_key in DegradationInsightGenerator._cache:
            DegradationInsightGenerator._cache.move_to_end(cache_key)
            return DegradationInsightGenerator._cache[cache_key]
        
        # Prepare comprehensive summary with physics context
        summary = DegradationInsightGenerator._prepare_physics_summary(
            analysis_results, analysis_type, parsed_params, graph_stats
        )
        
        # Physics equations as strings
        physics_eqs = "\n".join([f"• {name}: {eq}" for name, eq in PHYSICS_EQUATIONS.items()])
        
        system = """You are a senior battery reliability engineer with expertise in:
- Mechanical degradation (cracking, fatigue, fracture)
- Electrochemical processes (SEI formation, lithium plating)
- Thermal effects (runaway, heat generation)
- Chemo-mechanical coupling
- Physics-based modeling

Use the actual graph metrics and physics equations to provide grounded insights."""

        prompt = f"""User query: "{user_query}"
Analysis type: {analysis_type}
Relevance score: {relevance_score:.2f}
Graph stats: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges

User-specified parameters:
{json.dumps({k: v for k, v in parsed_params.items() if v != BatteryNLParser().defaults.get(k)}, indent=2, default=str)}

Physics equations (use these in your reasoning):
{physics_eqs}

Analysis summary with actual numbers:
{summary}

Think step-by-step:
1. Which degradation mechanisms dominate according to the numbers? (cite specific nodes and metrics)
2. Which physical pathways connect these mechanisms? (reference the physics equations)
3. What are the practical implications for cycle life, safety, and calendar aging?
4. What recommendations would you give to a battery engineer? (materials, operating conditions, modeling focus)

Output exactly 5 bullet points starting with "•". Each bullet must:
- Reference specific node names or metrics from the analysis
- Connect to at least one physics equation
- Be actionable and concise (1-2 sentences)

Insights:"""

        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default)')
        
        try:
            if "Qwen" in backend:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                full_prompt = f"{system}\n\n{prompt}\n"

            inputs = tokenizer.encode(full_prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=500,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Insights:" in generated:
                insights = generated.split("Insights:")[-1].strip()
            elif prompt in generated:
                insights = generated.split(prompt)[-1].strip()
            else:
                insights = generated.strip()
            
            insights = insights.replace("•", "\n•")
            if not insights.startswith("•"):
                insights = "• " + insights
            
            DegradationInsightGenerator._cache[cache_key] = insights
            if len(DegradationInsightGenerator._cache) > DegradationInsightGenerator._max_cache_size:
                DegradationInsightGenerator._cache.popitem(last=False)
            
            return insights
            
        except Exception as e:
            return f"• LLM insight generation unavailable: {str(e)}\n• Review the analysis results above for degradation patterns."
    
    @staticmethod
    def _prepare_physics_summary(results, analysis_type, params, graph_stats):
        """Create summary with physics context and actual numbers"""
        summary_lines = []
        
        summary_lines.append(f"Graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
        summary_lines.append(f"Filters: min_weight={params.get('min_weight')}, min_freq={params.get('min_freq')}, min_priority={params.get('min_priority_score'):.2f}")
        
        if analysis_type == "Centrality Analysis" and isinstance(results, pd.DataFrame) and not results.empty:
            top_degree = results.nlargest(3, 'degree')
            top_between = results.nlargest(3, 'betweenness')
            
            summary_lines.append("\nTop by degree (most connected):")
            for _, row in top_degree.iterrows():
                summary_lines.append(f"  - {row['node']}: degree={row['degree']:.3f}")
            
            summary_lines.append("\nTop by betweenness (bridges):")
            for _, row in top_between.iterrows():
                summary_lines.append(f"  - {row['node']}: betweenness={row['betweenness']:.3f}")
        
        elif analysis_type == "Pathway Analysis" and isinstance(results, dict):
            valid_paths = [v for v in results.values() if v['path'] is not None]
            summary_lines.append(f"\nFound {len(valid_paths)} pathways")
            for name, data in list(results.items())[:3]:
                if data['path']:
                    physics_note = " (contains physics terms)" if data.get('contains_physics') else ""
                    summary_lines.append(f"  {name}: length {data['length']}{physics_note}")
        
        elif analysis_type == "Community Detection" and isinstance(results, dict):
            summary_lines.append(f"\nDetected {len(results)} communities")
            for comm_id, data in list(results.items())[:3]:
                physics_count = sum(data['physics_terms'].values())
                summary_lines.append(f"  Community {comm_id}: {len(data['nodes'])} nodes, {physics_count} physics terms")
        
        elif analysis_type == "Correlation Analysis" and isinstance(results, tuple) and len(results) == 2:
            corr_matrix, terms = results
            if len(terms) > 0:
                strong_corr = []
                for i in range(min(len(terms), 10)):
                    for j in range(i+1, min(len(terms), 10)):
                        if corr_matrix[i, j] > 0.3:
                            strong_corr.append((terms[i], terms[j], corr_matrix[i, j]))
                strong_corr.sort(key=lambda x: x[2], reverse=True)
                summary_lines.append(f"\nStrongest correlations:")
                for t1, t2, val in strong_corr[:5]:
                    summary_lines.append(f"  {t1} ↔ {t2}: {val:.2f}")
        
        return "\n".join(summary_lines)

# -----------------------
# 10. SIDEBAR UPDATE UTILITY
# -----------------------
def apply_params_to_sidebar(params):
    """Update session state with parsed parameters and force rerun"""
    for key, val in params.items():
        if key in ['min_weight', 'min_freq', 'min_priority_score', 'priority_threshold', 
                   'edge_width_factor', 'label_font_size', 'label_max_chars',
                   'physics_boost_weight', 'min_physics_similarity']:
            st.session_state[f'auto_{key}'] = val
        elif key == 'excluded_terms':
            if isinstance(val, list):
                st.session_state['auto_excluded'] = ', '.join(val)
            else:
                st.session_state['auto_excluded'] = str(val)
        elif key == 'selected_nodes':
            st.session_state['auto_selected_nodes'] = val if isinstance(val, list) else [val]
        elif key == 'selected_categories':
            st.session_state['auto_selected_categories'] = val if isinstance(val, list) else [val]
        elif key == 'selected_types':
            st.session_state['auto_selected_types'] = val if isinstance(val, list) else [val]
        elif key in ['suppress_low_priority', 'highlight_priority', 'show_labels', 'require_physics_in_pathways']:
            st.session_state[f'auto_{key}'] = val
        elif key == 'analysis_type':
            st.session_state['auto_analysis_type'] = val
    
    st.rerun()

# -----------------------
# 11. RUN ANALYSIS DISPATCHER
# -----------------------
def run_analysis(analysis_type, params, G_filtered, nodes_df, edges_df):
    """Execute the specified analysis and return results"""
    
    if analysis_type == "Centrality Analysis":
        focus_terms = params.get('focus_terms', ['crack', 'fracture', 'degradation'])
        return analyze_failure_centrality(G_filtered, focus_terms)
    
    elif analysis_type == "Community Detection":
        communities, partition = detect_failure_communities(G_filtered)
        return communities
    
    elif analysis_type == "Ego Network Analysis":
        central_nodes = params.get('central_nodes', 
                                   ["electrode cracking", "SEI formation", "capacity fade"])
        return analyze_ego_networks(G_filtered, central_nodes)
    
    elif analysis_type == "Pathway Analysis":
        source_terms = params.get('source_terms', ["electrode cracking"])
        target_terms = params.get('target_terms', ["capacity fade"])
        require_physics = params.get('require_physics_in_pathways', False)
        return find_failure_pathways(G_filtered, source_terms, target_terms, require_physics)
    
    elif analysis_type == "Temporal Analysis":
        time_col = params.get('time_column', 'year')
        return analyze_temporal_patterns(nodes_df, edges_df, time_col)
    
    elif analysis_type == "Correlation Analysis":
        corr_matrix, terms = analyze_failure_correlations(G_filtered)
        return (corr_matrix, terms)
    
    else:
        return None

# -----------------------
# 12. INITIALIZE SESSION STATE
# -----------------------
def initialize_session_state():
    """Initialize all session state variables"""
    parser = BatteryNLParser()
    defaults = parser.defaults.copy()
    
    auto_defaults = {
        f'auto_{k}': v for k, v in defaults.items() 
        if k not in ['focus_terms', 'source_terms', 'target_terms', 'central_nodes', 'time_column']
    }
    auto_defaults['auto_excluded'] = ', '.join(defaults['excluded_terms'])
    
    state_vars = {
        'edges_df': None,
        'nodes_df': None,
        'G': None,
        'priority_scores': None,
        'parser': None,
        'relevance_scorer': None,
        'insight_generator': None,
        'llm_tokenizer': None,
        'llm_model': None,
        'llm_backend_loaded': "GPT-2 (default)",
        'last_query': "",
        'last_params': None,
        'last_analysis_results': None,
        'last_insights': None,
        'analysis_history': []
    }
    
    for key, value in {**auto_defaults, **state_vars}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.parser is None:
        st.session_state.parser = BatteryNLParser()
    if st.session_state.relevance_scorer is None:
        st.session_state.relevance_scorer = RelevanceScorer(use_scibert=True)
    if st.session_state.insight_generator is None:
        st.session_state.insight_generator = DegradationInsightGenerator()

# -----------------------
# 13. RELEVANCE SCORER
# -----------------------
class RelevanceScorer:
    def __init__(self, use_scibert=True):
        self.use_scibert = use_scibert and scibert_tokenizer is not None and scibert_model is not None
    
    def score_query_to_nodes(self, query, nodes_list):
        if not query or not nodes_list:
            return 0.5
        
        if self.use_scibert:
            try:
                query_emb = get_scibert_embedding(query)
                sample_nodes = nodes_list[:min(100, len(nodes_list))]
                node_embs = get_scibert_embedding(sample_nodes)
                
                valid = [i for i, emb in enumerate(node_embs) if emb is not None]
                if not valid or query_emb is None:
                    return 0.5
                
                sims = [cosine_similarity([query_emb], [node_embs[i]])[0][0] for i in valid]
                return float(np.mean(sims)) if sims else 0.5
            except:
                return 0.5
        else:
            words = set(query.lower().split())
            matches = sum(1 for node in nodes_list[:100] if any(w in node.lower() for w in words))
            return min(1.0, matches / 50.0)
    
    def get_confidence_level(self, score):
        if score >= 0.8:
            return "High confidence - well-matched", "green"
        elif score >= 0.6:
            return "Moderate confidence", "blue"
        elif score >= 0.4:
            return "Low confidence - refine query", "orange"
        else:
            return "Very low confidence", "red"

# -----------------------
# 14. MAIN APPLICATION
# -----------------------
try:
    # Initialize
    initialize_session_state()
    
    # Load data
    edges_df, nodes_df = load_data()
    st.session_state.edges_df = edges_df
    st.session_state.nodes_df = nodes_df

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
        G.add_node(row["node"], 
                   type=row["type"], 
                   category=row["category"], 
                   frequency=row["frequency"],
                   unit=row.get("unit", "None"),
                   similarity_score=row.get("similarity_score", 0))

    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], 
                   weight=row["weight"], 
                   type=row["type"], 
                   label=row["label"],
                   relationship=row.get("relationship", ""),
                   strength=row.get("strength", 0))

    st.session_state.G = G

    # Calculate Physics-Augmented Priority Scores
    physics_boost = st.session_state.get('auto_physics_boost_weight', 0.15)
    priority_scores = calculate_priority_scores(G, nodes_df, physics_boost_weight=physics_boost)
    for node in G.nodes():
        G.nodes[node]['priority_score'] = priority_scores.get(node, 0)
    nodes_df['priority_score'] = nodes_df['node'].apply(lambda x: priority_scores.get(x, 0))
    st.session_state.priority_scores = priority_scores

    # UI Header
    st.title("🔋 Intelligent Battery Degradation Knowledge Explorer")
    st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .insight-box { background-color: #e8f4f8; border-left: 5px solid #0366d6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .physics-badge { background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

    # Query Interface
    with st.expander("🤖 AI-Powered Query Interface", expanded=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            user_query = st.text_area(
                "Ask about battery degradation in natural language:",
                height=100,
                placeholder="e.g., 'Show pathways from electrode cracking to capacity fade involving diffusion-induced stress' or 'Analyze communities with high physics relevance'",
                key="user_query_input"
            )
        
        with col2:
            st.markdown("### 🧠 LLM Settings")
            model_choice = st.selectbox(
                "Model",
                options=["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"],
                index=0,
                key="llm_choice"
            )
            
            if TRANSFORMERS_AVAILABLE:
                tokenizer, model, loaded = load_llm(model_choice)
                st.session_state.llm_tokenizer = tokenizer
                st.session_state.llm_model = model
                st.session_state.llm_backend_loaded = loaded
            
            use_llm = st.checkbox("Use LLM parsing", value=True)
            use_ensemble = st.checkbox("Use ensemble", value=False)
            ensemble_runs = 3
            if use_ensemble:
                ensemble_runs = st.slider("Ensemble runs", 2, 5, 3)
            
            generate_insights = st.checkbox("Generate insights", value=True)
        
        with col3:
            st.markdown("### 🚀 Actions")
            run_button = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
            if st.button("🧹 Clear History", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.last_query = ""
                st.session_state.last_insights = None
                st.rerun()

    # Process Query
    if run_button and user_query:
        with st.spinner("🔍 Analyzing your query..."):
            parser = st.session_state.parser
            
            if use_llm and st.session_state.llm_tokenizer is not None:
                params = parser.hybrid_parse(
                    user_query, 
                    st.session_state.llm_tokenizer,
                    st.session_state.llm_model,
                    use_ensemble=use_ensemble,
                    ensemble_runs=ensemble_runs
                )
            else:
                params = parser.parse_regex(user_query)
            
            st.session_state.last_params = params
            st.session_state.last_query = user_query
            
            # Show parsed parameters
            st.markdown("### 📋 Parsed Parameters")
            param_cols = st.columns(3)
            physics_detected = []
            for i, (key, val) in enumerate(params.items()):
                if key in ['physics_boost_weight', 'require_physics_in_pathways', 'min_physics_similarity']:
                    physics_detected.append(f"{key}: {val}")
                with param_cols[i % 3]:
                    if isinstance(val, list):
                        val_str = ', '.join(str(v) for v in val[:3])
                        if len(val) > 3:
                            val_str += f"... (+{len(val)-3})"
                    else:
                        val_str = str(val)
                    st.metric(key.replace('_', ' ').title(), val_str)
            
            if physics_detected:
                st.markdown("**🔬 Physics Terms Detected:** " + ", ".join(physics_detected))
            
            # Relevance
            all_nodes = list(G.nodes())
            relevance = st.session_state.relevance_scorer.score_query_to_nodes(user_query, all_nodes[:100])
            conf_text, conf_color = st.session_state.relevance_scorer.get_confidence_level(relevance)
            st.markdown(f"**Semantic Relevance:** {relevance:.3f} - "
                       f"<span style='color:{conf_color};font-weight:bold;'>{conf_text}</span>",
                       unsafe_allow_html=True)
            
            # Apply to sidebar
            apply_params_to_sidebar(params)

    # Sidebar Filters (all reading from session_state)
    with st.sidebar:
        st.markdown("## ⚙️ Filters")
        
        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.slider(
                "Min edge weight", 
                min_value=int(edges_df["weight"].min()), 
                max_value=int(edges_df["weight"].max()), 
                value=st.session_state.get('auto_min_weight', 10), 
                step=1,
                key="min_weight_slider"
            )
        with col2:
            min_node_freq = st.slider(
                "Min node frequency", 
                min_value=int(nodes_df["frequency"].min()), 
                max_value=int(nodes_df["frequency"].max()), 
                value=st.session_state.get('auto_min_freq', 5), 
                step=1,
                key="min_freq_slider"
            )

        # Categories
        categories = sorted(nodes_df["category"].dropna().unique())
        default_cats = st.session_state.get('auto_selected_categories', categories)
        selected_categories = st.multiselect(
            "Filter by category", 
            categories, 
            default=default_cats,
            key="category_multiselect"
        )

        # Node types
        node_types = sorted(nodes_df["type"].dropna().unique())
        default_types = st.session_state.get('auto_selected_types', node_types)
        selected_types = st.multiselect(
            "Filter by node type", 
            node_types, 
            default=default_types,
            key="type_multiselect"
        )

        # Priority
        min_priority = st.slider(
            "Min priority score", 
            0.0, 1.0, 
            st.session_state.get('auto_min_priority_score', 0.2), 
            0.05,
            key="priority_slider"
        )

        # Node selection
        st.subheader("🔍 Node Inclusion/Exclusion")
        default_selected = st.session_state.get('auto_selected_nodes', 
                                               ["electrode cracking", "SEI formation", "capacity fade"])
        default_selected = [n for n in default_selected if n in G.nodes()]
        selected_nodes = st.multiselect(
            "Include specific nodes", 
            options=sorted(G.nodes()),
            default=default_selected,
            key="nodes_multiselect"
        )
        
        default_excluded = st.session_state.get('auto_excluded', 'battery, material')
        excluded_input = st.text_input(
            "Exclude terms (comma-separated)", 
            value=default_excluded,
            key="excluded_input"
        )
        excluded_terms = [t.strip().lower() for t in excluded_input.split(',') if t.strip()]

        # Physics settings
        st.subheader("🔬 Physics Settings")
        physics_boost = st.slider(
            "Physics boost weight", 
            0.0, 0.5, 
            st.session_state.get('auto_physics_boost_weight', 0.15), 
            0.05,
            key="physics_boost_slider",
            help="Higher values prioritize nodes matching battery physics terms"
        )
        
        require_physics = st.checkbox(
            "Require physics terms in pathways", 
            value=st.session_state.get('auto_require_physics_in_pathways', False),
            key="physics_require_checkbox"
        )

        # Highlighting
        st.subheader("🎯 Highlighting")
        highlight = st.checkbox(
            "Highlight high-priority nodes", 
            value=st.session_state.get('auto_highlight_priority', True),
            key="highlight_checkbox"
        )
        threshold = st.slider(
            "Highlight threshold", 
            0.5, 1.0, 
            st.session_state.get('auto_priority_threshold', 0.7), 
            0.05,
            key="threshold_slider"
        )
        suppress = st.checkbox(
            "Suppress low-priority nodes", 
            value=st.session_state.get('auto_suppress_low_priority', False),
            key="suppress_checkbox"
        )

        # Labels
        st.subheader("📝 Labels")
        show_labels = st.checkbox(
            "Show labels", 
            value=st.session_state.get('auto_show_labels', True),
            key="labels_checkbox"
        )
        label_size = st.slider(
            "Font size", 
            10, 100, 
            st.session_state.get('auto_label_font_size', 16),
            key="font_slider"
        )
        max_chars = st.slider(
            "Max chars", 
            10, 30, 
            st.session_state.get('auto_label_max_chars', 15),
            key="chars_slider"
        )
        
        edge_width = st.slider(
            "Edge width factor", 
            0.1, 2.0, 
            st.session_state.get('auto_edge_width_factor', 0.5),
            key="edge_slider"
        )

        # Apply filters
        G_filtered = filter_graph(
            G, min_weight, min_node_freq, selected_categories, selected_types,
            selected_nodes, excluded_terms, min_priority, suppress
        )
        st.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # Run analysis if we have a query
    if G_filtered.number_of_nodes() > 0 and st.session_state.last_params:
        analysis_type = st.session_state.last_params.get('analysis_type', 'Centrality Analysis')
        
        # Run analysis
        results = run_analysis(analysis_type, st.session_state.last_params, 
                              G_filtered, nodes_df, edges_df)
        st.session_state.last_analysis_results = results
        
        # Generate insights
        if generate_insights and st.session_state.llm_tokenizer is not None and results is not None:
            with st.spinner("🤖 Generating physics-grounded insights..."):
                graph_stats = {'nodes': G_filtered.number_of_nodes(), 'edges': G_filtered.number_of_edges()}
                insights = st.session_state.insight_generator.generate_insights(
                    results, analysis_type, user_query, 
                    relevance if 'relevance' in locals() else 0.5,
                    st.session_state.last_params,
                    graph_stats,
                    st.session_state.llm_tokenizer, 
                    st.session_state.llm_model
                )
                st.session_state.last_insights = insights
        
        # Add to history
        st.session_state.analysis_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': user_query,
            'analysis_type': analysis_type,
            'nodes': G_filtered.number_of_nodes(),
            'edges': G_filtered.number_of_edges()
        })
        if len(st.session_state.analysis_history) > 10:
            st.session_state.analysis_history = st.session_state.analysis_history[-10:]

    # Display insights
    if st.session_state.last_insights:
        st.markdown("### 🤖 Physics-Grounded Insights")
        st.markdown(f"""
        <div class="insight-box">
        {st.session_state.last_insights.replace('•', '<br>•')}
        </div>
        """, unsafe_allow_html=True)

    # Visualization
    if G_filtered.number_of_nodes() > 0:
        # FIX: Define analysis_type with a default value for the visualization title
        if st.session_state.last_params:
            analysis_type_display = st.session_state.last_params.get('analysis_type', 'General View')
        else:
            analysis_type_display = 'General View'
        
        # Node coloring by category
        cats = list(set([G_filtered.nodes[n].get('category', 'Unknown') for n in G_filtered.nodes()]))
        color_map = {c: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] for i, c in enumerate(cats)}
        node_colors = [color_map.get(G_filtered.nodes[n].get('category', 'Unknown'), 'lightgray') for n in G_filtered.nodes()]
        
        # Layout
        pos = nx.spring_layout(G_filtered, k=1, iterations=100, seed=42, weight='weight')
        
        # Node sizes based on priority
        scores = [G_filtered.nodes[n].get('priority_score', 0) for n in G_filtered.nodes()]
        min_s, max_s = 15, 60
        if max(scores) > min(scores):
            sizes = [min_s + (max_s - min_s) * (s - min(scores)) / (max(scores) - min(scores)) for s in scores]
        else:
            sizes = [30] * len(scores)
        
        # Edges
        edge_x, edge_y = [], []
        for u, v in G_filtered.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=edge_width, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Nodes
        node_x, node_y, node_text, node_labels, node_symbols = [], [], [], [], []
        for node in G_filtered.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            d = G_filtered.nodes[node]
            node_text.append(
                f"{node}<br>"
                f"Category: {d.get('category', 'N/A')}<br>"
                f"Frequency: {d.get('frequency', 'N/A')}<br>"
                f"Priority: {d.get('priority_score', 0):.3f}"
            )
            if len(node) > max_chars:
                node_labels.append(node[:max_chars] + "...")
            else:
                node_labels.append(node)
            if highlight and d.get('priority_score', 0) > threshold:
                node_symbols.append('star')
            else:
                node_symbols.append('circle')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            text=node_labels if show_labels else None,
            textfont=dict(size=label_size, color='black'),
            textposition='middle center',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                color=node_colors,
                size=sizes,
                symbol=node_symbols,
                line=dict(width=1, color='darkgray')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        for cat, col in color_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=col), name=cat, showlegend=True
            ))
        
        fig.update_layout(
            title=f"Battery Degradation Knowledge Graph - {analysis_type_display}",
            showlegend=True,
            legend=dict(x=1.05, y=1),
            hovermode='closest',
            margin=dict(b=20, l=5, r=150, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Analysis Results Display
    if st.session_state.last_analysis_results and st.session_state.last_params:
        atype = st.session_state.last_params.get('analysis_type', 'Centrality Analysis')
        st.markdown(f"### 📊 {atype} Results")
        res = st.session_state.last_analysis_results
        
        if atype == "Centrality Analysis" and isinstance(res, pd.DataFrame) and not res.empty:
            st.dataframe(res[['node', 'degree', 'betweenness', 'closeness', 'category']].head(20))
            fig = px.scatter(res, x='degree', y='betweenness', color='category', 
                           size='eigenvector', hover_data=['node'])
            st.plotly_chart(fig)
        
        elif atype == "Community Detection" and isinstance(res, dict):
            for cid, data in list(res.items())[:5]:
                with st.expander(f"Community {cid} ({len(data['nodes'])} nodes)"):
                    st.write("**Physics terms:**", dict(data['physics_terms'].most_common(5)))
                    st.write("**Sample nodes:**", ", ".join(data['nodes'][:10]))
        
        elif atype == "Pathway Analysis" and isinstance(res, dict):
            valid = sum(1 for v in res.values() if v['path'] is not None)
            st.write(f"Found {valid} pathways")
            for name, data in list(res.items())[:10]:
                if data['path']:
                    badge = "🔬" if data.get('contains_physics') else ""
                    st.write(f"**{name}** {badge}: {' → '.join(data['nodes'])}")
        
        elif atype == "Correlation Analysis" and isinstance(res, tuple) and len(res) == 2:
            corr, terms = res
            if len(terms) > 0:
                fig = px.imshow(corr, x=terms, y=terms, color_continuous_scale='Reds')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)

    # History
    if st.session_state.analysis_history:
        with st.expander("📜 Analysis History"):
            for entry in reversed(st.session_state.analysis_history[-5:]):
                st.write(f"**{entry['timestamp']}** - {entry['query'][:50]}...")
                st.write(f"Type: {entry['analysis_type']}, Nodes: {entry['nodes']}")

    # Export
    st.sidebar.subheader("💾 Export")
    if st.sidebar.button("Export Filtered Graph"):
        nodes_exp = pd.DataFrame([
            {'node': n, **{k: v for k, v in G_filtered.nodes[n].items()}}
            for n in G_filtered.nodes()
        ])
        edges_exp = pd.DataFrame([
            {'source': u, 'target': v, **G_filtered.edges[u, v]}
            for u, v in G_filtered.edges()
        ])
        st.sidebar.download_button("Download Nodes", nodes_exp.to_csv(index=False), "nodes.csv")
        st.sidebar.download_button("Download Edges", edges_exp.to_csv(index=False), "edges.csv")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.text(traceback.format_exc())
