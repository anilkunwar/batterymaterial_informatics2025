#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT BATTERY DEGRADATION KNOWLEDGE EXPLORER
===================================================
- Natural language interface that ACTUALLY updates sidebar widgets
- Hybrid regex + LLM parsing with confidence merging
- Auto-rerun with parsed parameters applied to ALL controls
- LLM-powered degradation insights with full context
- All original visualizations preserved
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
    """
    Calculate priority scores for nodes based on frequency, centrality, and semantic relevance.
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
    
    # Calculate semantic relevance to key terms
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
    
    # Combine scores (weights: frequency=0.4, degree=0.3, betweenness=0.2, semantic=0.1)
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
                'failure_keywords': Counter()
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
                
                # Calculate metrics for this ego network
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
    """
    Find shortest paths between different types of failure mechanisms
    """
    pathways = {}
    
    for source in source_terms:
        for target in target_terms:
            if source in G_filtered.nodes() and target in G_filtered.nodes():
                try:
                    path = nx.shortest_path(G_filtered, source=source, target=target, weight='weight')
                    pathways[f"{source} -> {target}"] = {
                        'path': path,
                        'length': len(path) - 1,  # Number of edges
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
# 4. Helper Functions for Export
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
# 5. UNIFIED LLM LOADER (CoreShellGPT pattern)
# -----------------------
@st.cache_resource(show_spinner="Loading LLM for intelligent analysis...")
def load_llm(backend):
    """Load the selected LLM model with caching - exact CoreShellGPT pattern"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, backend
    
    try:
        if "GPT-2" in backend:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            # Set pad token for GPT-2
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
# 6. INTELLIGENT NLP PARSER (Full CoreShellGPT implementation)
# -----------------------
class BatteryNLParser:
    """
    Extract knowledge graph parameters from natural language
    Implements full hybrid parsing with confidence merging (CoreShellGPT pattern)
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
            'time_column': 'year'
        }
        
        # Regex patterns for fast extraction
        self.patterns = {
            'min_weight': [
                r'min(?:imum)?\s*edge\s*weight\s*(?:of|>=|>|=)?\s*(\d+)',
                r'edge\s*weight\s*[>=]\s*(\d+)',
                r'weight\s*[>=]\s*(\d+)'
            ],
            'min_freq': [
                r'min(?:imum)?\s*(?:node)?\s*frequency\s*(?:of|>=|>|=)?\s*(\d+)',
                r'frequency\s*[>=]\s*(\d+)'
            ],
            'min_priority_score': [
                r'priority\s*score\s*(?:of|>=|>|=)?\s*(\d*\.?\d+)',
                r'min(?:imum)?\s*priority\s*(\d*\.?\d+)'
            ],
            'priority_threshold': [
                r'highlight\s*threshold\s*(?:of|>=|>|=)?\s*(\d*\.?\d+)',
                r'threshold\s*(\d*\.?\d+)'
            ],
            'excluded_terms': [
                r'exclude\s*(?:terms?)?\s*:?\s*([a-zA-Z,\s]+)',
                r'ignore\s*([a-zA-Z,\s]+)',
                r'without\s*([a-zA-Z,\s]+)'
            ],
            'suppress_low_priority': [
                r'suppress\s*low\s*priority',
                r'hide\s*low\s*priority',
                r'only\s*high\s*priority'
            ],
            'highlight_priority': [
                r'highlight\s*high\s*priority',
                r'show\s*priority',
                r'mark\s*important'
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
            'group': 'Community Detection',
            'ego': 'Ego Network Analysis',
            'neighborhood': 'Ego Network Analysis',
            'neighbours': 'Ego Network Analysis',
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
        """Fast regex-based parsing - returns dict with defaults + extracted values"""
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
                        if param in ['suppress_low_priority', 'highlight_priority']:
                            params[param] = True
                        elif param == 'excluded_terms':
                            terms = [t.strip() for t in match.group(1).split(',')]
                            params[param] = terms
                        else:
                            params[param] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Determine analysis type
        for key, analysis in self.analysis_map.items():
            if key in text_lower:
                params['analysis_type'] = analysis
                break
        
        # Extract specific terms for analysis
        if params['analysis_type'] == 'Pathway Analysis':
            # Look for source and target
            source_match = re.search(r'from\s+([a-zA-Z\s\-]+?)\s+to', text_lower)
            target_match = re.search(r'to\s+([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)', text_lower)
            
            if source_match:
                params['source_terms'] = [t.strip() for t in source_match.group(1).split(',')]
            if target_match:
                params['target_terms'] = [t.strip() for t in target_match.group(1).split(',')]
        
        elif params['analysis_type'] == 'Ego Network Analysis':
            # Look for central nodes
            central_match = re.search(r'around\s+([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)', text_lower)
            if central_match:
                params['central_nodes'] = [t.strip() for t in central_match.group(1).split(',')]
        
        # Extract categories
        categories = ["Crack and Fracture", "Deformation", "Degradation", "Fatigue", "Mechanical", "Chemical"]
        found_cats = []
        for cat in categories:
            if cat.lower() in text_lower:
                found_cats.append(cat)
        if found_cats:
            params['selected_categories'] = found_cats
        
        return params

    def _extract_json_robust(self, generated):
        """Extract and repair JSON from generated text - CoreShellGPT pattern"""
        # Try to find JSON block
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, generated, re.DOTALL)
        
        if not match:
            # Try more lenient pattern
            match = re.search(r'\{.*\}', generated, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            
            # Common repairs
            json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = re.sub(r'([}\]{])\s*([}\]])', r'\1\2', json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        return None

    def parse_with_llm(self, text, tokenizer, model, regex_params=None, temperature=0.1):
        """Use LLM to extract parameters with robust JSON extraction"""
        if not text or tokenizer is None or model is None:
            return regex_params if regex_params else self.defaults.copy()
        
        # Build comprehensive few-shot prompt with 10+ battery examples
        system = """You are an expert in battery degradation analysis. Extract parameters for knowledge graph exploration from the user query. Always return valid JSON with the exact keys specified."""
        
        examples = """
Examples:
1. "Show me pathways from electrode cracking to capacity fade with min edge weight 20 and high priority nodes only" 
   → {"analysis_type": "Pathway Analysis", "source_terms": ["electrode cracking"], "target_terms": ["capacity fade"], "min_weight": 20, "min_priority_score": 0.7, "highlight_priority": true}

2. "Analyze communities related to mechanical degradation, include nodes with frequency > 10" 
   → {"analysis_type": "Community Detection", "focus_terms": ["mechanical", "degradation"], "min_freq": 10, "selected_categories": ["Crack and Fracture", "Degradation"]}

3. "Show ego network around SEI formation and lithium plating, suppress low priority nodes" 
   → {"analysis_type": "Ego Network Analysis", "central_nodes": ["SEI formation", "lithium plating"], "suppress_low_priority": true, "min_priority_score": 0.5}

4. "How have failure concepts evolved over time? Highlight high-priority nodes above 0.6" 
   → {"analysis_type": "Temporal Analysis", "highlight_priority": true, "priority_threshold": 0.6, "time_column": "year"}

5. "Find correlations between cracking and degradation mechanisms, exclude generic terms like 'battery'" 
   → {"analysis_type": "Correlation Analysis", "focus_terms": ["cracking", "degradation"], "excluded_terms": ["battery"]}

6. "Show me the most important degradation mechanisms with min edge weight 15, min frequency 8, and priority score above 0.3" 
   → {"analysis_type": "Centrality Analysis", "min_weight": 15, "min_freq": 8, "min_priority_score": 0.3, "focus_terms": ["degradation"]}

7. "Focus only on crack and fracture categories, with edge width factor 0.8 and larger labels" 
   → {"selected_categories": ["Crack and Fracture"], "edge_width_factor": 0.8, "label_font_size": 18, "show_labels": true}

8. "Find pathways from micro-cracking to thermal runaway, exclude 'electrolyte' from analysis" 
   → {"analysis_type": "Pathway Analysis", "source_terms": ["micro-cracking"], "target_terms": ["thermal runaway"], "excluded_terms": ["electrolyte"]}

9. "Show me the ego network around cyclic mechanical damage with neighbors up to radius 2, highlight important nodes" 
   → {"analysis_type": "Ego Network Analysis", "central_nodes": ["cyclic mechanical damage"], "highlight_priority": true, "priority_threshold": 0.7}

10. "Analyze temporal patterns of SEI formation and electrode cracking, include all categories" 
    → {"analysis_type": "Temporal Analysis", "focus_terms": ["SEI formation", "electrode cracking"], "selected_categories": ["Crack and Fracture", "Deformation", "Degradation", "Fatigue"]}
"""
        
        regex_hint = ""
        if regex_params:
            # Only include params that differ from defaults
            diff_params = {k: v for k, v in regex_params.items() 
                          if k in self.defaults and v != self.defaults[k]}
            if diff_params:
                regex_hint = f"\nPreliminary regex extraction (use as a hint, but prioritize the text): {json.dumps(diff_params, default=str)}"
        
        defaults_json = json.dumps(self.defaults, default=str)
        
        user = f"""{examples}
{regex_hint}
Defaults: {defaults_json}

Keys must be exactly: {list(self.defaults.keys())}

Text: "{text}"

Output ONLY the JSON object with the extracted parameters. Use the exact keys shown above.
JSON:"""

        # Format prompt based on model type
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
            
            # Extract JSON
            params = self._extract_json_robust(generated)
            
            if params:
                # Fill in missing keys with defaults
                for key in self.defaults:
                    if key not in params:
                        params[key] = self.defaults[key]
                
                # Clip numerical values to valid ranges
                if 'min_weight' in params:
                    params['min_weight'] = max(1, int(params['min_weight']))
                if 'min_freq' in params:
                    params['min_freq'] = max(1, int(params['min_freq']))
                if 'min_priority_score' in params:
                    params['min_priority_score'] = np.clip(float(params['min_priority_score']), 0, 1)
                if 'priority_threshold' in params:
                    params['priority_threshold'] = np.clip(float(params['priority_threshold']), 0, 1)
                if 'edge_width_factor' in params:
                    params['edge_width_factor'] = np.clip(float(params['edge_width_factor']), 0.1, 2.0)
                if 'label_font_size' in params:
                    params['label_font_size'] = max(10, min(100, int(params['label_font_size'])))
                
                return params
            else:
                return regex_params if regex_params else self.defaults.copy()
                
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}")
            return regex_params if regex_params else self.defaults.copy()

    def parse_with_ensemble(self, text, tokenizer, model, n_runs=3, temperature=0.2):
        """Run multiple LLM parses and combine with voting - CoreShellGPT pattern"""
        regex_params = self.parse_regex(text)
        
        all_params = []
        for i in range(n_runs):
            params = self.parse_with_llm(text, tokenizer, model, regex_params, temperature)
            all_params.append(params)
        
        # Combine by voting/averaging
        combined = {}
        for key in self.defaults:
            values = [p[key] for p in all_params]
            
            if isinstance(self.defaults[key], bool):
                # Boolean: majority vote
                combined[key] = max(set(values), key=values.count)
            elif isinstance(self.defaults[key], (int, float)):
                # Numeric: average (filter out None)
                numeric_vals = [v for v in values if isinstance(v, (int, float))]
                if numeric_vals:
                    combined[key] = float(np.mean(numeric_vals))
                else:
                    combined[key] = self.defaults[key]
            elif isinstance(self.defaults[key], list):
                # List: take most common elements
                flat_vals = []
                for vlist in values:
                    if isinstance(vlist, list):
                        flat_vals.extend(vlist)
                if flat_vals:
                    # Get unique items, keep order by frequency
                    from collections import Counter
                    counter = Counter(flat_vals)
                    # Limit to reasonable length
                    top_items = [item for item, _ in counter.most_common(10)]
                    combined[key] = top_items
                else:
                    combined[key] = self.defaults[key]
            else:
                # String: majority vote
                str_vals = [str(v) for v in values if v is not None]
                if str_vals:
                    combined[key] = max(set(str_vals), key=str_vals.count)
                else:
                    combined[key] = self.defaults[key]
        
        return combined

    def hybrid_parse(self, text, tokenizer=None, model=None, use_ensemble=False, ensemble_runs=3):
        """Combine regex and LLM parsing with confidence-based merging - CoreShellGPT pattern"""
        regex_params = self.parse_regex(text)
        
        # If no LLM available, return regex
        if tokenizer is None or model is None:
            return regex_params
        
        # Compute regex confidence (1.0 if changed from default, 0.0 otherwise)
        regex_conf = {}
        for key in self.defaults:
            if key in regex_params:
                if isinstance(regex_params[key], (int, float)) and isinstance(self.defaults[key], (int, float)):
                    if abs(regex_params[key] - self.defaults[key]) > 1e-6:
                        regex_conf[key] = 1.0
                    else:
                        regex_conf[key] = 0.0
                elif isinstance(regex_params[key], list) and isinstance(self.defaults[key], list):
                    if set(regex_params[key]) != set(self.defaults[key]):
                        regex_conf[key] = 1.0
                    else:
                        regex_conf[key] = 0.0
                elif regex_params[key] != self.defaults[key]:
                    regex_conf[key] = 1.0
                else:
                    regex_conf[key] = 0.0
            else:
                regex_conf[key] = 0.0
        
        # Get LLM parameters
        if use_ensemble:
            llm_params = self.parse_with_ensemble(text, tokenizer, model, n_runs=ensemble_runs)
        else:
            llm_params = self.parse_with_llm(text, tokenizer, model, regex_params)
        
        # LLM confidence: 0.7 if changed from default, 0.3 otherwise
        llm_conf = {}
        for key in self.defaults:
            if key in llm_params:
                if isinstance(llm_params[key], (int, float)) and isinstance(self.defaults[key], (int, float)):
                    if abs(llm_params[key] - self.defaults[key]) > 1e-6:
                        llm_conf[key] = 0.7
                    else:
                        llm_conf[key] = 0.3
                elif isinstance(llm_params[key], list) and isinstance(self.defaults[key], list):
                    if set(llm_params[key]) != set(self.defaults[key]):
                        llm_conf[key] = 0.7
                    else:
                        llm_conf[key] = 0.3
                elif llm_params[key] != self.defaults[key]:
                    llm_conf[key] = 0.7
                else:
                    llm_conf[key] = 0.3
            else:
                llm_conf[key] = 0.0
        
        # Merge with confidence
        final_params = {}
        for key in self.defaults:
            if regex_conf.get(key, 0) >= llm_conf.get(key, 0):
                final_params[key] = regex_params.get(key, self.defaults[key])
            else:
                final_params[key] = llm_params.get(key, self.defaults[key])
        
        # Special handling for analysis_type - prefer LLM for complex cases
        if 'analysis_type' in llm_params and llm_params['analysis_type'] != self.defaults['analysis_type']:
            if regex_params.get('analysis_type') == self.defaults['analysis_type']:
                final_params['analysis_type'] = llm_params['analysis_type']
        
        return final_params

    def get_explanation(self, params, original_text):
        """Generate explanation of parsed parameters"""
        lines = ["### 📋 Parsed Parameters from Query", ""]
        lines.append(f"**Query:** _{original_text}_")
        lines.append("")
        lines.append("| Parameter | Value | Status |")
        lines.append("|-----------|-------|--------|")
        
        for key, val in params.items():
            if key in ['selected_categories', 'selected_types', 'selected_nodes', 'excluded_terms', 'focus_terms', 'source_terms', 'target_terms', 'central_nodes']:
                if isinstance(val, list):
                    val_str = ', '.join(str(v) for v in val[:5])
                    if len(val) > 5:
                        val_str += f"... (+{len(val)-5})"
                else:
                    val_str = str(val)
            else:
                if isinstance(val, float):
                    val_str = f"{val:.3f}"
                elif isinstance(val, bool):
                    val_str = "✓" if val else "✗"
                else:
                    val_str = str(val)
            
            status = "📌 Extracted" if val != self.defaults.get(key) else "⚪ Default"
            lines.append(f"| {key} | {val_str} | {status} |")
        
        return "\n".join(lines)

# -----------------------
# 7. SEMANTIC RELEVANCE SCORER
# -----------------------
class RelevanceScorer:
    """Compute semantic relevance between user query and graph nodes"""
    
    def __init__(self, use_scibert=True):
        self.use_scibert = use_scibert and scibert_tokenizer is not None and scibert_model is not None
    
    def score_query_to_nodes(self, query, nodes_list):
        """Score relevance of query to a list of node terms"""
        if not query or not nodes_list:
            return 0.5
        
        if self.use_scibert:
            try:
                # Get embeddings
                query_emb = get_scibert_embedding(query)
                # Sample nodes for speed
                sample_nodes = nodes_list[:min(100, len(nodes_list))]
                node_embs = get_scibert_embedding(sample_nodes)
                
                # Filter out None
                valid_indices = [i for i, emb in enumerate(node_embs) if emb is not None]
                if not valid_indices or query_emb is None:
                    return 0.5
                
                # Compute similarities
                similarities = []
                for i in valid_indices:
                    sim = cosine_similarity([query_emb], [node_embs[i]])[0][0]
                    similarities.append(sim)
                
                return float(np.mean(similarities)) if similarities else 0.5
                
            except Exception as e:
                return 0.5
        else:
            # Fallback: simple keyword matching
            query_lower = query.lower()
            query_words = set(query_lower.split())
            matches = 0
            for node in nodes_list[:100]:
                node_lower = node.lower()
                if any(word in node_lower for word in query_words):
                    matches += 1
            return min(1.0, matches / 50.0)  # Normalize
    
    def get_confidence_level(self, score):
        """Return confidence text and color based on score"""
        if score >= 0.8:
            return "High confidence - query well-matched to knowledge base", "green"
        elif score >= 0.6:
            return "Moderate confidence - reasonable match", "blue"
        elif score >= 0.4:
            return "Low confidence - consider refining query", "orange"
        else:
            return "Very low confidence - query may not be well-represented", "red"

# -----------------------
# 8. LLM-BASED INSIGHT GENERATOR (with full context)
# -----------------------
class DegradationInsightGenerator:
    """Generate intelligent insights from analysis results using LLM"""
    
    _cache = OrderedDict()
    _max_cache_size = 20
    
    @staticmethod
    def _extract_json_robust(generated):
        """Extract JSON from generated text"""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, generated, re.DOTALL)
        
        if not match:
            match = re.search(r'\{.*\}', generated, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except:
                pass
        return None

    @staticmethod
    def generate_insights(analysis_results, analysis_type, user_query, relevance_score, 
                          parsed_params, tokenizer, model):
        """Generate insights with full context including parsed parameters"""
        
        # Create cache key
        cache_key = hashlib.md5(f"{analysis_type}_{user_query}_{relevance_score:.3f}_{str(parsed_params)[:100]}".encode()).hexdigest()
        
        # Check cache
        if cache_key in DegradationInsightGenerator._cache:
            DegradationInsightGenerator._cache.move_to_end(cache_key)
            return DegradationInsightGenerator._cache[cache_key]
        
        # Prepare comprehensive summary
        summary = DegradationInsightGenerator._prepare_summary(analysis_results, analysis_type, parsed_params)
        
        system = "You are a senior battery degradation expert. Provide concise, actionable insights based on the knowledge graph analysis. Output as bullet points."
        
        prompt = f"""User query: "{user_query}"
Analysis type: {analysis_type}
Relevance score: {relevance_score:.2f} (0-1 scale, higher means better match)

User-specified parameters:
{json.dumps({k: v for k, v in parsed_params.items() if v != BatteryNLParser().defaults.get(k)}, indent=2, default=str)}

Analysis summary:
{summary}

Based on this knowledge graph analysis, provide 4-6 bullet points explaining:
1. Key degradation mechanisms identified and why they're important (connect to the user's query)
2. Relationships between different failure modes (mention specific pathways/correlations if found)
3. Practical implications for battery reliability and safety
4. Recommendations for further investigation or mitigation strategies

Keep each bullet point concise (1-2 sentences). Use scientific terminology but explain technical terms.
Start each bullet with "•" and keep the response focused on the actual analysis results.

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
                    max_new_tokens=400,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the insights part
            if "Insights:" in generated:
                insights = generated.split("Insights:")[-1].strip()
            elif prompt in generated:
                insights = generated.split(prompt)[-1].strip()
            else:
                insights = generated.strip()
            
            # Clean up
            insights = insights.replace("•", "\n•")
            if not insights.startswith("•"):
                insights = "• " + insights
            
            # Cache
            DegradationInsightGenerator._cache[cache_key] = insights
            if len(DegradationInsightGenerator._cache) > DegradationInsightGenerator._max_cache_size:
                DegradationInsightGenerator._cache.popitem(last=False)
            
            return insights
            
        except Exception as e:
            return f"• LLM insight generation unavailable: {str(e)}\n• Review the analysis results above for degradation patterns."
    
    @staticmethod
    def _prepare_summary(results, analysis_type, params):
        """Create a concise text summary including user parameters"""
        summary_lines = []
        
        # Add user constraints
        summary_lines.append(f"User filters: min_weight={params.get('min_weight', 'N/A')}, min_freq={params.get('min_freq', 'N/A')}, min_priority={params.get('min_priority_score', 'N/A'):.2f}")
        if params.get('selected_nodes'):
            summary_lines.append(f"Focused nodes: {', '.join(params['selected_nodes'][:3])}")
        if params.get('excluded_terms'):
            summary_lines.append(f"Excluded: {', '.join(params['excluded_terms'])}")
        
        summary_lines.append("")
        
        if analysis_type == "Centrality Analysis" and isinstance(results, pd.DataFrame) and not results.empty:
            top_degree = results.nlargest(5, 'degree')[['node', 'degree']]
            top_between = results.nlargest(5, 'betweenness')[['node', 'betweenness']]
            
            summary_lines.append("Top 5 failure terms by degree centrality (most connected):")
            for _, row in top_degree.iterrows():
                summary_lines.append(f"  - {row['node']}: {row['degree']:.3f}")
            
            summary_lines.append("\nTop 5 failure terms by betweenness centrality (bridges):")
            for _, row in top_between.iterrows():
                summary_lines.append(f"  - {row['node']}: {row['betweenness']:.3f}")
        
        elif analysis_type == "Community Detection" and isinstance(results, dict):
            summary_lines.append(f"Detected {len(results)} communities")
            for comm_id, data in list(results.items())[:3]:
                summary_lines.append(f"Community {comm_id}: {len(data['nodes'])} nodes")
                if data['failure_keywords']:
                    keywords = ", ".join([f"{k}({c})" for k, c in data['failure_keywords'].most_common(3)])
                    summary_lines.append(f"  Failure keywords: {keywords}")
        
        elif analysis_type == "Pathway Analysis" and isinstance(results, dict):
            pathways_found = sum(1 for v in results.values() if v['path'] is not None)
            summary_lines.append(f"Found {pathways_found} pathways out of {len(results)} requested")
            for name, data in list(results.items())[:3]:
                if data['path']:
                    summary_lines.append(f"  {name}: length {data['length']}, path: {' → '.join(data['nodes'][:3])}...")
        
        elif analysis_type == "Correlation Analysis" and isinstance(results, tuple) and len(results) == 2:
            corr_matrix, terms = results
            if len(terms) > 0:
                strong_corr = []
                for i in range(min(len(terms), 10)):
                    for j in range(i+1, min(len(terms), 10)):
                        if corr_matrix[i, j] > 0.3:
                            strong_corr.append((terms[i], terms[j], corr_matrix[i, j]))
                
                strong_corr.sort(key=lambda x: x[2], reverse=True)
                summary_lines.append(f"Analyzed {len(terms)} failure terms")
                for t1, t2, val in strong_corr[:5]:
                    summary_lines.append(f"  {t1} ↔ {t2}: {val:.2f}")
        
        return "\n".join(summary_lines)

# -----------------------
# 9. Apply Parsed Parameters to Sidebar (CoreShellGPT pattern)
# -----------------------
def apply_params_to_sidebar(params):
    """Update session state with parsed parameters and force rerun"""
    for key, val in params.items():
        if key in ['min_weight', 'min_freq', 'min_priority_score', 'priority_threshold', 
                   'edge_width_factor', 'label_font_size', 'label_max_chars']:
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
        elif key in ['suppress_low_priority', 'highlight_priority', 'show_labels']:
            st.session_state[f'auto_{key}'] = val
        elif key == 'analysis_type':
            st.session_state['auto_analysis_type'] = val
    
    # Force rerun to update widgets
    st.rerun()

# -----------------------
# 10. Run Selected Analysis
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
        return find_failure_pathways(G_filtered, source_terms, target_terms)
    
    elif analysis_type == "Temporal Analysis":
        time_col = params.get('time_column', 'year')
        return analyze_temporal_patterns(nodes_df, edges_df, time_col)
    
    elif analysis_type == "Correlation Analysis":
        corr_matrix, terms = analyze_failure_correlations(G_filtered)
        return (corr_matrix, terms)
    
    else:
        return None

# -----------------------
# 11. Initialize Session State
# -----------------------
def initialize_session_state():
    """Initialize all session state variables with defaults"""
    parser = BatteryNLParser()
    defaults = parser.defaults.copy()
    
    # Add auto-prefixed defaults for widgets
    auto_defaults = {
        f'auto_{k}': v for k, v in defaults.items() 
        if k not in ['focus_terms', 'source_terms', 'target_terms', 'central_nodes', 'time_column']
    }
    
    # Special handling for excluded terms (string)
    auto_defaults['auto_excluded'] = ', '.join(defaults['excluded_terms'])
    
    # Add other session state variables
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
        'llm_backend_loaded': "GPT-2 (default, fastest startup)",
        'last_query': "",
        'last_params': None,
        'last_analysis_results': None,
        'last_insights': None,
        'llm_cache': OrderedDict(),
        'analysis_history': []
    }
    
    # Merge all defaults
    for key, value in {**auto_defaults, **state_vars}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize components
    if st.session_state.parser is None:
        st.session_state.parser = BatteryNLParser()
    if st.session_state.relevance_scorer is None:
        st.session_state.relevance_scorer = RelevanceScorer(use_scibert=True)
    if st.session_state.insight_generator is None:
        st.session_state.insight_generator = DegradationInsightGenerator()

# -----------------------
# 12. Main Application
# -----------------------
try:
    # Initialize session state
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

    # Calculate Priority Scores and Add to Graph
    priority_scores = calculate_priority_scores(G, nodes_df)
    for node in G.nodes():
        G.nodes[node]['priority_score'] = priority_scores.get(node, 0)
    nodes_df['priority_score'] = nodes_df['node'].apply(lambda x: priority_scores.get(x, 0))
    st.session_state.priority_scores = priority_scores

    # Sidebar and Main UI
    st.title("🔋 Intelligent Battery Degradation Knowledge Explorer")
    
    # Enhanced CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #0366d6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-high { color: green; font-weight: bold; }
    .confidence-moderate { color: blue; font-weight: bold; }
    .confidence-low { color: orange; font-weight: bold; }
    .confidence-very-low { color: red; font-weight: bold; }
    .params-table {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="big-font">
    Explore key concepts in **battery mechanical degradation research** using natural language.
    - **Nodes** = Terms (colored by category).  
    - **Size** = Priority score.  
    - **Edges** = Relationships (thicker = stronger).  
    </div>
    """, unsafe_allow_html=True)

    # ==================== INTELLIGENT QUERY INTERFACE ====================
    with st.expander("🤖 AI-Powered Query Interface", expanded=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            user_query = st.text_area(
                "Ask about battery degradation in natural language:",
                height=100,
                placeholder="e.g., 'Show me pathways from electrode cracking to capacity fade with min edge weight 20' or 'Analyze communities related to mechanical degradation with high priority nodes'",
                key="user_query_input"
            )
        
        with col2:
            st.markdown("### 🧠 LLM Settings")
            
            # Model selection
            model_choice = st.selectbox(
                "Model",
                options=["GPT-2 (default, fastest startup)", 
                         "Qwen2-0.5B-Instruct", 
                         "Qwen2.5-0.5B-Instruct (recommended)"],
                index=0,
                key="llm_choice"
            )
            
            # Load LLM
            if TRANSFORMERS_AVAILABLE:
                tokenizer, model, loaded_backend = load_llm(model_choice)
                st.session_state.llm_tokenizer = tokenizer
                st.session_state.llm_model = model
                st.session_state.llm_backend_loaded = loaded_backend
                
                if tokenizer is not None:
                    st.caption(f"✅ {loaded_backend} loaded")
                else:
                    st.caption("❌ Model failed to load")
            
            # LLM options
            use_llm = st.checkbox("Use LLM parsing", value=True)
            use_ensemble = st.checkbox("Use ensemble (slower but robust)", value=False)
            if use_ensemble:
                ensemble_runs = st.slider("Ensemble runs", 2, 5, 3)
            else:
                ensemble_runs = 3
            
            generate_insights = st.checkbox("Generate LLM insights", value=True)
        
        with col3:
            st.markdown("### 🚀 Actions")
            run_button = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
            
            if st.button("🧹 Clear History", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.last_query = ""
                st.session_state.last_insights = None
                st.rerun()
    
    # ==================== PROCESS QUERY ====================
    if run_button and user_query:
        with st.spinner("🔍 Analyzing your query..."):
            parser = st.session_state.parser
            
            # Parse query
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
            
            # Store for later use
            st.session_state.last_params = params
            st.session_state.last_query = user_query
            
            # Show parsed parameters
            st.markdown(parser.get_explanation(params, user_query))
            
            # Compute relevance score
            all_nodes = list(G.nodes())
            relevance_score = st.session_state.relevance_scorer.score_query_to_nodes(user_query, all_nodes[:100])
            confidence_text, confidence_color = st.session_state.relevance_scorer.get_confidence_level(relevance_score)
            
            st.markdown(f"**Semantic Relevance:** {relevance_score:.3f} - "
                       f"<span style='color:{confidence_color};font-weight:bold;'>{confidence_text}</span>",
                       unsafe_allow_html=True)
            
            # CRITICAL: Apply parsed parameters to sidebar and rerun
            apply_params_to_sidebar(params)

    # ==================== SIDEBAR FILTERS ====================
    # ALL widgets now read from session_state with auto_* values from parsed query
    
    # Create two columns for filters
    col1, col2 = st.sidebar.columns(2)

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

    # Category filter
    categories = sorted(nodes_df["category"].dropna().unique())
    default_cats = st.session_state.get('auto_selected_categories', categories)
    selected_categories = st.sidebar.multiselect(
        "Filter by category", 
        categories, 
        default=default_cats,
        key="category_multiselect"
    )

    # Node type filter
    node_types = sorted(nodes_df["type"].dropna().unique())
    default_types = st.session_state.get('auto_selected_types', node_types)
    selected_types = st.sidebar.multiselect(
        "Filter by node type", 
        node_types, 
        default=default_types,
        key="type_multiselect"
    )

    # Priority score filter
    min_priority_score = st.sidebar.slider(
        "Min priority score", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.get('auto_min_priority_score', 0.2), 
        step=0.05,
        key="priority_slider"
    )

    # Node inclusion/exclusion
    st.sidebar.subheader("🔍 Node Inclusion/Exclusion")
    
    default_selected = st.session_state.get('auto_selected_nodes', 
                                           ["electrode cracking", "SEI formation", "capacity fade"])
    default_selected = [n for n in default_selected if n in G.nodes()]
    
    selected_nodes = st.sidebar.multiselect(
        "Include specific nodes (optional)", 
        options=sorted(G.nodes()),
        default=default_selected,
        key="nodes_multiselect"
    )
    
    default_excluded = st.session_state.get('auto_excluded', 'battery, material')
    excluded_terms_input = st.sidebar.text_input(
        "Exclude terms (comma-separated)", 
        value=default_excluded,
        key="excluded_input"
    )
    excluded_terms = [t.strip().lower() for t in excluded_terms_input.split(',') if t.strip()]

    # Priority highlighting
    st.sidebar.subheader("🎯 Priority Highlighting")
    highlight_priority = st.sidebar.checkbox(
        "Highlight high-priority nodes", 
        value=st.session_state.get('auto_highlight_priority', True),
        key="highlight_checkbox"
    )
    
    priority_threshold = st.sidebar.slider(
        "Priority highlight threshold", 
        0.5, 1.0, 
        st.session_state.get('auto_priority_threshold', 0.7), 
        step=0.05,
        key="threshold_slider"
    )
    
    suppress_low_priority = st.sidebar.checkbox(
        "Suppress low-priority nodes", 
        value=st.session_state.get('auto_suppress_low_priority', False),
        key="suppress_checkbox"
    )

    # Label settings
    st.sidebar.subheader("📝 Label Settings")
    show_labels = st.sidebar.checkbox(
        "Show Node Labels", 
        value=st.session_state.get('auto_show_labels', True),
        key="labels_checkbox"
    )
    label_font_size = st.sidebar.slider(
        "Label Font Size", 
        10, 100, 
        st.session_state.get('auto_label_font_size', 16),
        key="font_slider"
    )
    label_max_chars = st.sidebar.slider(
        "Max Characters per Label", 
        10, 30, 
        st.session_state.get('auto_label_max_chars', 15),
        key="chars_slider"
    )
    
    # Edge width control
    edge_width_factor = st.sidebar.slider(
        "Edge Width Factor", 
        0.1, 2.0, 
        st.session_state.get('auto_edge_width_factor', 0.5),
        key="edge_slider"
    )

    # Graph stats
    G_filtered = filter_graph(G, min_weight, min_node_freq, selected_categories, selected_types, 
                               selected_nodes, excluded_terms, min_priority_score, suppress_low_priority)
    st.sidebar.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # ==================== RUN ANALYSIS WITH PARSED PARAMETERS ====================
    if G_filtered.number_of_nodes() > 0 and st.session_state.last_params:
        # Get analysis type from parsed params or default
        analysis_type = st.session_state.last_params.get('analysis_type', 'Centrality Analysis')
        
        # Run the specified analysis
        analysis_results = run_analysis(analysis_type, st.session_state.last_params, 
                                        G_filtered, nodes_df, edges_df)
        st.session_state.last_analysis_results = analysis_results
        
        # Generate insights if requested
        if generate_insights and st.session_state.llm_tokenizer is not None and analysis_results is not None:
            with st.spinner("🤖 Generating intelligent insights..."):
                insights = st.session_state.insight_generator.generate_insights(
                    analysis_results, analysis_type, user_query, 
                    relevance_score if 'relevance_score' in locals() else 0.5,
                    st.session_state.last_params,
                    st.session_state.llm_tokenizer, 
                    st.session_state.llm_model
                )
                st.session_state.last_insights = insights
        else:
            st.session_state.last_insights = None
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': user_query,
            'analysis_type': analysis_type,
            'node_count': G_filtered.number_of_nodes(),
            'edge_count': G_filtered.number_of_edges()
        }
        st.session_state.analysis_history.append(history_entry)
        if len(st.session_state.analysis_history) > 10:
            st.session_state.analysis_history = st.session_state.analysis_history[-10:]

    # ==================== DISPLAY INSIGHTS ====================
    if st.session_state.last_insights:
        st.markdown("### 🤖 AI-Generated Degradation Insights")
        st.markdown(f"""
        <div class="insight-box">
        {st.session_state.last_insights.replace('•', '<br>•')}
        </div>
        """, unsafe_allow_html=True)

    # ==================== VISUALIZATION ====================
    if G_filtered.number_of_nodes() > 0:
        # Color nodes by category
        categories_in_graph = list(set([G_filtered.nodes[node].get('category', 'Unknown') for node in G_filtered.nodes()]))
        
        if len(categories_in_graph) <= 10:
            color_palette = px.colors.qualitative.Set3
        else:
            color_palette = px.colors.qualitative.Alphabet
            
        category_color_map = {}
        for i, category in enumerate(categories_in_graph):
            category_color_map[category] = color_palette[i % len(color_palette)]
        
        node_colors = [category_color_map.get(G_filtered.nodes[node].get('category', 'Unknown'), 'lightgray') 
                      for node in G_filtered.nodes()]
        
        # Node positions
        pos = nx.spring_layout(G_filtered, k=1, iterations=100, seed=42, weight='weight')
        
        # Node sizes based on priority score
        priority_scores_filtered = [G_filtered.nodes[node].get('priority_score', 0) for node in G_filtered.nodes()]
        min_size, max_size = 15, 60
        if max(priority_scores_filtered) > min(priority_scores_filtered):
            node_sizes = [min_size + (max_size - min_size) * 
                         (score - min(priority_scores_filtered)) / (max(priority_scores_filtered) - min(priority_scores_filtered)) 
                         for score in priority_scores_filtered]
        else:
            node_sizes = [30] * len(priority_scores_filtered)
        
        # Build edge traces
        edge_x = []
        edge_y = []
        for edge in G_filtered.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=edge_width_factor, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_labels = []
        node_symbols = []
        
        for node in G_filtered.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = G_filtered.nodes[node]
            node_text.append(
                f"{node}<br>"
                f"Category: {node_data.get('category', 'N/A')}<br>"
                f"Type: {node_data.get('type', 'N/A')}<br>"
                f"Frequency: {node_data.get('frequency', 'N/A')}<br>"
                f"Priority Score: {node_data.get('priority_score', 0):.3f}"
            )
            if len(node) > label_max_chars:
                node_labels.append(node[:label_max_chars] + "...")
            else:
                node_labels.append(node)
            
            if highlight_priority and node_data.get('priority_score', 0) > priority_threshold:
                node_symbols.append('star')
            else:
                node_symbols.append('circle')
        
        if show_labels:
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_labels,
                textfont=dict(size=label_font_size, color='black'),
                textposition='middle center',
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=node_sizes,
                    symbol=node_symbols,
                    line=dict(width=2, color='darkgray')
                )
            )
        else:
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=node_sizes,
                    symbol=node_symbols,
                    line=dict(width=2, color='darkgray')
                )
            )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Add category legend
        for category, color in category_color_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=category,
                showlegend=True
            ))
        
        fig.update_layout(
            title=dict(
                text=f'Battery Degradation Knowledge Graph - {st.session_state.last_params.get("analysis_type", "Explorer") if st.session_state.last_params else "Explorer"}',
                font=dict(size=24)
            ),
            showlegend=True,
            legend=dict(x=1.05, y=1, xanchor='left'),
            hovermode='closest',
            margin=dict(b=20, l=5, r=150, t=80),
            annotations=[dict(
                text=f"Query: {st.session_state.last_query if st.session_state.last_query else 'Explore the graph'}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # ==================== ANALYSIS RESULTS DISPLAY ====================
    if st.session_state.last_analysis_results is not None and st.session_state.last_params:
        analysis_type = st.session_state.last_params.get('analysis_type', 'Centrality Analysis')
        st.markdown(f"### 📊 {analysis_type} Results")
        
        results = st.session_state.last_analysis_results
        
        if analysis_type == "Centrality Analysis" and isinstance(results, pd.DataFrame) and not results.empty:
            st.dataframe(results[['node', 'degree', 'betweenness', 'closeness', 'category']].head(20))
            
            fig = px.scatter(
                results, x='degree', y='betweenness',
                color='category', size='eigenvector',
                hover_data=['node', 'closeness'],
                title=f"Centrality of Failure-Related Terms"
            )
            st.plotly_chart(fig)
        
        elif analysis_type == "Community Detection" and isinstance(results, dict):
            for comm_id, data in list(results.items())[:5]:
                with st.expander(f"Community {comm_id} ({len(data['nodes'])} nodes)"):
                    st.write("**Top Categories:**")
                    for category, count in data['categories'].most_common(3):
                        st.write(f"- {category}: {count} nodes")
                    
                    st.write("**Failure Keywords:**")
                    for keyword, count in data['failure_keywords'].most_common(5):
                        st.write(f"- {keyword}: {count} occurrences")
                    
                    st.write("**Sample Nodes:**")
                    st.write(", ".join(data['nodes'][:10]))
        
        elif analysis_type == "Pathway Analysis" and isinstance(results, dict):
            pathways_found = sum(1 for v in results.values() if v['path'] is not None)
            st.write(f"Found {pathways_found} pathways out of {len(results)}")
            
            for name, data in list(results.items())[:10]:
                if data['path']:
                    st.write(f"**{name}** (length: {data['length']})")
                    st.write(" → ".join(data['nodes']))
                else:
                    st.write(f"**{name}**: No path found")
        
        elif analysis_type == "Correlation Analysis" and isinstance(results, tuple) and len(results) == 2:
            corr_matrix, terms = results
            if len(terms) > 0:
                fig = px.imshow(
                    corr_matrix,
                    x=terms,
                    y=terms,
                    title="Correlation Between Failure Mechanisms",
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
        
        elif analysis_type == "Ego Network Analysis" and isinstance(results, dict):
            for node, data in results.items():
                with st.expander(f"Ego Network: {node}"):
                    st.write(f"**Nodes:** {data['node_count']}, **Edges:** {data['edge_count']}")
                    st.write(f"**Density:** {data['density']:.3f}")
                    st.write("**Neighbors:**")
                    st.write(", ".join(data['neighbors'][:15]))

    # ==================== ADDITIONAL ANALYTICS ====================
    if G_filtered.number_of_nodes() > 0:
        st.subheader("📊 Graph Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = nodes_df['category'].value_counts()
            fig_cat = px.pie(values=category_counts.values, names=category_counts.index, 
                            title="Node Distribution by Category")
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            top_nodes = nodes_df.nlargest(10, 'priority_score')[['node', 'priority_score']]
            fig_nodes = px.bar(top_nodes, x='priority_score', y='node', orientation='h',
                              title="Top Nodes by Priority Score")
            st.plotly_chart(fig_nodes, use_container_width=True)
        
        # Edge type distribution
        if 'type' in edges_df.columns:
            edge_type_counts = edges_df['type'].value_counts()
            fig_edge = px.bar(x=edge_type_counts.index, y=edge_type_counts.values,
                             title="Edge Type Distribution",
                             labels={'x': 'Edge Type', 'y': 'Count'})
            st.plotly_chart(fig_edge, use_container_width=True)

    # ==================== ANALYSIS HISTORY ====================
    if st.session_state.analysis_history:
        with st.expander("📜 Analysis History"):
            for entry in reversed(st.session_state.analysis_history[-5:]):
                st.write(f"**{entry['timestamp']}** - {entry['query'][:50]}...")
                st.write(f"Type: {entry['analysis_type']}, Nodes: {entry['node_count']}")

    # ==================== DATA EXPORT ====================
    st.sidebar.subheader("💾 Export Data")
    if st.sidebar.button("Export Filtered Graph"):
        filtered_nodes = []
        for node, data in G_filtered.nodes(data=True):
            filtered_nodes.append({
                'node': node,
                'type': data.get('type', ''),
                'category': data.get('category', ''),
                'frequency': data.get('frequency', 0),
                'priority_score': data.get('priority_score', 0)
            })
        
        filtered_edges = []
        for u, v, data in G_filtered.edges(data=True):
            filtered_edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0),
                'type': data.get('type', '')
            })
        
        nodes_export = pd.DataFrame(filtered_nodes)
        edges_export = pd.DataFrame(filtered_edges)
        
        st.sidebar.download_button(
            label="Download Nodes CSV",
            data=nodes_export.to_csv(index=False),
            file_name="filtered_nodes.csv",
            mime="text/csv"
        )
        
        st.sidebar.download_button(
            label="Download Edges CSV",
            data=edges_export.to_csv(index=False),
            file_name="filtered_edges.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.text("Detailed error information:")
    st.text(traceback.format_exc())
