# integrated LLM interface to knowledge graph and data analysis 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT BATTERY DEGRADATION KNOWLEDGE EXPLORER
===================================================
- Natural language interface (regex + GPT/Qwen hybrid parsing)
- Intelligent filtering and analysis selection
- Semantic relevance scoring with SciBERT
- LLM-powered degradation insights and recommendations
- All original visualizations preserved and enhanced
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

# Predefined key terms for battery reliability (from original script)
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
# 5. INTELLIGENT NLP PARSER FOR BATTERY DEGRADATION (NEW)
# -----------------------
class BatteryNLParser:
    """
    Extract knowledge graph parameters and analysis types from natural language
    using regex and optionally GPT/Qwen.
    """
    def __init__(self):
        self.defaults = {
            'min_weight': 10,
            'min_freq': 5,
            'selected_categories': [],
            'selected_types': [],
            'selected_nodes': [],
            'excluded_terms': ['battery', 'material'],
            'min_priority_score': 0.2,
            'suppress_low_priority': False,
            'analysis_type': 'Centrality Analysis',
            'focus_terms': ['crack', 'fracture', 'degradation', 'fatigue', 'damage'],
            'source_terms': ['electrode cracking'],
            'target_terms': ['capacity fade'],
            'central_nodes': ['electrode cracking', 'SEI formation', 'capacity fade'],
            'highlight_priority': True,
            'priority_threshold': 0.7,
            'time_column': 'year'
        }
        
        # Analysis type mapping for synonyms
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
        
        # Regex patterns for direct extraction
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
            ]
        }
        
        # Term extraction for node selection
        self.term_patterns = [
            r'focus\s*(?:on)?\s*([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)',
            r'analyze\s*([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)',
            r'terms?\s*:?\s*([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)',
            r'nodes?\s*:?\s*([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)',
            r'concepts?\s*:?\s*([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)'
        ]

    def parse(self, text):
        """Fast regex-based parsing"""
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
                        if param == 'excluded_terms':
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
            for pattern in self.term_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    params['central_nodes'] = [t.strip() for t in match.group(1).split(',')]
                    break
        
        # General term extraction for focus
        for pattern in self.term_patterns:
            match = re.search(pattern, text_lower)
            if match:
                params['focus_terms'] = [t.strip() for t in match.group(1).split(',')]
                break
        
        return params

    def parse_with_llm(self, text, tokenizer, model, regex_params=None, temperature=0.1):
        """Use LLM to extract parameters with robust JSON extraction"""
        if not text:
            return self.defaults.copy()
        
        # Build few-shot prompt
        system = """You are an expert in battery degradation analysis. Extract parameters for knowledge graph exploration from the user query. Always return valid JSON with the exact keys specified."""
        
        examples = """
Examples:
- "Show me the most important degradation mechanisms with min edge weight 15 and min frequency 10" 
  → {"min_weight": 15, "min_freq": 10, "analysis_type": "Centrality Analysis", "focus_terms": ["degradation"], "selected_categories": [], "min_priority_score": 0.2}

- "Find pathways from electrode cracking to capacity fade, exclude generic terms like 'battery'" 
  → {"analysis_type": "Pathway Analysis", "source_terms": ["electrode cracking"], "target_terms": ["capacity fade"], "excluded_terms": ["battery"], "min_priority_score": 0.2}

- "Analyze the ego network around SEI formation and lithium plating with high priority nodes only" 
  → {"analysis_type": "Ego Network Analysis", "central_nodes": ["SEI formation", "lithium plating"], "suppress_low_priority": true, "min_priority_score": 0.5}

- "Show communities related to mechanical degradation, include nodes with frequency > 8" 
  → {"analysis_type": "Community Detection", "focus_terms": ["mechanical", "degradation"], "min_freq": 8, "selected_categories": []}

- "How have failure concepts evolved over time? Highlight high-priority nodes above 0.6" 
  → {"analysis_type": "Temporal Analysis", "highlight_priority": true, "priority_threshold": 0.6}
"""
        
        regex_hint = ""
        if regex_params:
            regex_hint = f"\nPreliminary regex extraction (use as a hint, but prioritize the text): {json.dumps(regex_params, default=str)}"
        
        defaults_json = json.dumps(self.defaults, default=str)
        
        user = f"""{examples}
{regex_hint}
Defaults: {defaults_json}

Keys must be exactly: min_weight, min_freq, selected_categories, selected_types, selected_nodes, excluded_terms, min_priority_score, suppress_low_priority, analysis_type, focus_terms, source_terms, target_terms, central_nodes, highlight_priority, priority_threshold, time_column

Text: "{text}"

JSON:"""

        # Format prompt based on model type
        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default)')
        
        if "Qwen" in backend:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\n\n{user}\n"

        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=300,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Robust JSON extraction
            params = self._extract_json_robust(generated)
            
            if params:
                # Fill in missing keys with defaults
                for key in self.defaults:
                    if key not in params:
                        params[key] = self.defaults[key]
                
                # Clip numerical values
                if 'min_weight' in params:
                    params['min_weight'] = max(1, int(params['min_weight']))
                if 'min_freq' in params:
                    params['min_freq'] = max(1, int(params['min_freq']))
                if 'min_priority_score' in params:
                    params['min_priority_score'] = np.clip(float(params['min_priority_score']), 0, 1)
                if 'priority_threshold' in params:
                    params['priority_threshold'] = np.clip(float(params['priority_threshold']), 0, 1)
                
                return params
            else:
                st.warning("LLM did not produce valid JSON. Falling back to regex.")
                return regex_params if regex_params else self.defaults.copy()
                
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}. Falling back to regex.")
            return regex_params if regex_params else self.defaults.copy()

    def _extract_json_robust(self, generated):
        """Extract and repair JSON from generated text"""
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
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None

    def parse_with_ensemble(self, text, tokenizer, model, n_runs=3, temperature=0.2):
        """Run multiple LLM parses and combine with voting"""
        regex_params = self.parse(text)
        
        all_params = []
        for _ in range(n_runs):
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
                # Numeric: average
                combined[key] = np.mean([v for v in values if v is not None])
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
                    combined[key] = [item for item, _ in counter.most_common()]
                else:
                    combined[key] = self.defaults[key]
            else:
                # String: majority vote
                combined[key] = max(set(values), key=values.count)
        
        return combined

    def hybrid_parse(self, text, tokenizer=None, model=None, use_ensemble=False, ensemble_runs=3):
        """Combine regex and LLM parsing with confidence-based merging"""
        regex_params = self.parse(text)
        
        if tokenizer is None or model is None:
            return regex_params
        
        # Get LLM parameters
        if use_ensemble:
            llm_params = self.parse_with_ensemble(text, tokenizer, model, n_runs=ensemble_runs)
        else:
            llm_params = self.parse_with_llm(text, tokenizer, model, regex_params)
        
        # Simple merge: prefer regex for numerical if different from default,
        # otherwise use LLM
        final_params = {}
        for key in self.defaults:
            regex_val = regex_params.get(key, self.defaults[key])
            llm_val = llm_params.get(key, self.defaults[key])
            
            # Check if regex found something (different from default)
            regex_found = False
            if isinstance(regex_val, (int, float)) and isinstance(self.defaults[key], (int, float)):
                if abs(regex_val - self.defaults[key]) > 1e-6:
                    regex_found = True
            elif isinstance(regex_val, list) and isinstance(self.defaults[key], list):
                if set(regex_val) != set(self.defaults[key]):
                    regex_found = True
            elif regex_val != self.defaults[key]:
                regex_found = True
            
            if regex_found:
                final_params[key] = regex_val
            else:
                final_params[key] = llm_val
        
        return final_params

# -----------------------
# 6. SEMANTIC RELEVANCE SCORER (NEW)
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
                node_embs = get_scibert_embedding(nodes_list)
                
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
                st.warning(f"Relevance scoring failed: {e}")
                return 0.5
        else:
            # Fallback: simple keyword matching
            query_lower = query.lower()
            matches = sum(1 for node in nodes_list if any(word in node.lower() for word in query_lower.split()))
            return min(1.0, matches / max(1, len(nodes_list)) * 2)
    
    def get_confidence_level(self, score):
        """Return confidence text and color based on score"""
        if score >= 0.8:
            return "High confidence - query well-matched to knowledge base", "green"
        elif score >= 0.6:
            return "Moderate confidence - reasonable match", "blue"
        elif score >= 0.4:
            return "Low confidence - consider refining query", "orange"
        else:
            return "Very low confidence - query may not be well-represented in knowledge base", "red"

# -----------------------
# 7. LLM-BASED INSIGHT GENERATOR (NEW)
# -----------------------
class DegradationInsightGenerator:
    """Generate intelligent insights from analysis results using LLM"""
    
    _cache = OrderedDict()
    _max_cache_size = 20
    
    @staticmethod
    def generate_insights(analysis_results, analysis_type, user_query, relevance_score, tokenizer, model):
        """Generate insights from analysis results using LLM with caching"""
        
        # Create cache key
        cache_key = hashlib.md5(f"{analysis_type}_{user_query}_{relevance_score:.3f}".encode()).hexdigest()
        
        # Check cache
        if cache_key in DegradationInsightGenerator._cache:
            # Move to end (LRU)
            DegradationInsightGenerator._cache.move_to_end(cache_key)
            return DegradationInsightGenerator._cache[cache_key]
        
        # Prepare a concise summary of results
        summary = DegradationInsightGenerator._prepare_summary(analysis_results, analysis_type)
        
        system = "You are a senior battery degradation expert. Provide concise, actionable insights based on the knowledge graph analysis."
        
        prompt = f"""User query: "{user_query}"
Analysis type: {analysis_type}
Relevance score: {relevance_score:.2f} (0-1 scale, higher means better match to knowledge base)

Analysis summary:
{summary}

Based on this knowledge graph analysis, provide 3-5 bullet points explaining:
1. Key degradation mechanisms identified and why they're important
2. Relationships between different failure modes
3. Practical implications for battery reliability
4. Recommendations for further investigation

Keep each bullet point concise (1-2 sentences). Use scientific terminology but explain technical terms.
Start each bullet with "•" and keep the response focused on the actual analysis results.

Insights:"""

        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default)')
        
        if "Qwen" in backend:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = f"{system}\n\n{prompt}\n"

        try:
            inputs = tokenizer.encode(full_prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the insights part (after the prompt)
            if "Insights:" in generated:
                insights = generated.split("Insights:")[-1].strip()
            else:
                insights = generated.split(prompt)[-1].strip() if prompt in generated else generated.strip()
            
            # Clean up
            insights = insights.replace("•", "\n•")
            if not insights.startswith("•"):
                insights = "• " + insights
            
            # Cache with LRU
            DegradationInsightGenerator._cache[cache_key] = insights
            if len(DegradationInsightGenerator._cache) > DegradationInsightGenerator._max_cache_size:
                DegradationInsightGenerator._cache.popitem(last=False)
            
            return insights
            
        except Exception as e:
            return f"• LLM insight generation failed: {str(e)}\n• Review the analysis results above for degradation patterns."
    
    @staticmethod
    def _prepare_summary(results, analysis_type):
        """Create a concise text summary of analysis results"""
        summary_lines = []
        
        if analysis_type == "Centrality Analysis" and isinstance(results, pd.DataFrame) and not results.empty:
            top_degree = results.nlargest(5, 'degree')[['node', 'degree']]
            top_between = results.nlargest(5, 'betweenness')[['node', 'betweenness']]
            
            summary_lines.append("Top 5 failure terms by degree centrality:")
            for _, row in top_degree.iterrows():
                summary_lines.append(f"  - {row['node']}: {row['degree']:.3f}")
            
            summary_lines.append("\nTop 5 failure terms by betweenness centrality (bridges):")
            for _, row in top_between.iterrows():
                summary_lines.append(f"  - {row['node']}: {row['betweenness']:.3f}")
        
        elif analysis_type == "Community Detection" and isinstance(results, dict):
            summary_lines.append(f"Detected {len(results)} communities")
            for comm_id, data in list(results.items())[:3]:  # First 3 communities
                summary_lines.append(f"Community {comm_id}: {len(data['nodes'])} nodes")
                if data['failure_keywords']:
                    keywords = ", ".join([f"{k}({c})" for k, c in data['failure_keywords'].most_common(3)])
                    summary_lines.append(f"  Failure keywords: {keywords}")
        
        elif analysis_type == "Pathway Analysis" and isinstance(results, dict):
            pathways_found = sum(1 for v in results.values() if v['path'] is not None)
            summary_lines.append(f"Found {pathways_found} pathways out of {len(results)} possible connections")
            for name, data in list(results.items())[:3]:
                if data['path']:
                    summary_lines.append(f"  {name}: length {data['length']}")
        
        elif analysis_type == "Correlation Analysis" and isinstance(results, tuple) and len(results) == 2:
            corr_matrix, terms = results
            if len(terms) > 0:
                # Find strongest correlations
                strong_corr = []
                for i in range(len(terms)):
                    for j in range(i+1, len(terms)):
                        if corr_matrix[i, j] > 0.5:
                            strong_corr.append((terms[i], terms[j], corr_matrix[i, j]))
                
                strong_corr.sort(key=lambda x: x[2], reverse=True)
                summary_lines.append(f"Analyzed {len(terms)} failure terms")
                for t1, t2, val in strong_corr[:5]:
                    summary_lines.append(f"  {t1} ↔ {t2}: {val:.2f}")
        
        else:
            summary_lines.append(f"Analysis produced {type(results).__name__} results")
            if isinstance(results, pd.DataFrame):
                summary_lines.append(f"DataFrame with {len(results)} rows")
            elif isinstance(results, dict):
                summary_lines.append(f"Dictionary with {len(results)} keys")
        
        return "\n".join(summary_lines)

# -----------------------
# 8. UNIFIED LLM LOADER (NEW)
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
# 9. Initialize Session State
# -----------------------
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
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
    
    for key, value in defaults.items():
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
# 10. Auto-apply Parameters to Sidebar
# -----------------------
def apply_params_to_sidebar(params):
    """Update sidebar widgets with parsed parameters"""
    # This is tricky because Streamlit widgets need reruns
    # Instead, we'll store in session state and use them in the main flow
    
    # Min weight
    if 'min_weight' in params:
        st.session_state['auto_min_weight'] = params['min_weight']
    
    # Min frequency
    if 'min_freq' in params:
        st.session_state['auto_min_freq'] = params['min_freq']
    
    # Min priority score
    if 'min_priority_score' in params:
        st.session_state['auto_min_priority'] = params['min_priority_score']
    
    # Excluded terms
    if 'excluded_terms' in params and params['excluded_terms']:
        st.session_state['auto_excluded'] = ', '.join(params['excluded_terms'])
    
    # Suppress low priority
    if 'suppress_low_priority' in params:
        st.session_state['auto_suppress_low'] = params['suppress_low_priority']
    
    # Highlight priority
    if 'highlight_priority' in params:
        st.session_state['auto_highlight'] = params['highlight_priority']
    
    # Priority threshold
    if 'priority_threshold' in params:
        st.session_state['auto_priority_threshold'] = params['priority_threshold']
    
    # Selected nodes
    if 'selected_nodes' in params and params['selected_nodes']:
        st.session_state['auto_selected_nodes'] = params['selected_nodes']

# -----------------------
# 11. Run Selected Analysis
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

    # Sidebar Controls
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
                placeholder="e.g., 'Show me pathways from electrode cracking to capacity fade' or 'Analyze communities related to mechanical degradation with high priority nodes'",
                key="user_query"
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
            
            # Load LLM if available
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
            use_llm = st.checkbox("Use LLM parsing", value=True, 
                                  help="Use LLM to extract parameters from query")
            use_ensemble = st.checkbox("Use ensemble (slower but more robust)", value=False,
                                       help="Run multiple generations and combine")
            if use_ensemble:
                ensemble_runs = st.slider("Ensemble runs", 2, 5, 3)
            else:
                ensemble_runs = 3
            
            generate_insights = st.checkbox("Generate LLM insights", value=True,
                                           help="Explain results in natural language")
        
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
            # Parse query
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
                params = parser.parse(user_query)
            
            # Store for later use
            st.session_state.last_params = params
            st.session_state.last_query = user_query
            
            # Auto-apply parameters (store in session for widget defaults)
            apply_params_to_sidebar(params)
            
            # Show parsed parameters
            st.markdown("### 📋 Parsed Parameters")
            
            param_cols = st.columns(3)
            param_items = list(params.items())
            
            for i, (key, value) in enumerate(param_items[:9]):  # Show first 9
                with param_cols[i % 3]:
                    if isinstance(value, list):
                        val_str = ', '.join(str(v) for v in value[:3])
                        if len(value) > 3:
                            val_str += f"... (+{len(value)-3})"
                    else:
                        val_str = str(value)
                    st.metric(key.replace('_', ' ').title(), val_str)
            
            # Compute relevance score
            all_nodes = list(G.nodes())
            relevance_score = st.session_state.relevance_scorer.score_query_to_nodes(user_query, all_nodes[:100])  # Sample for speed
            confidence_text, confidence_color = st.session_state.relevance_scorer.get_confidence_level(relevance_score)
            
            st.markdown(f"**Semantic Relevance:** {relevance_score:.3f} - "
                       f"<span style='color:{confidence_color};font-weight:bold;'>{confidence_text}</span>",
                       unsafe_allow_html=True)
    
    # ==================== SIDEBAR FILTERS ====================
    # (with auto-filled values from query if available)
    
    # Create two columns for filters
    col1, col2 = st.sidebar.columns(2)

    with col1:
        default_min_weight = st.session_state.get('auto_min_weight', 10)
        min_weight = st.slider(
            "Min edge weight", 
            min_value=int(edges_df["weight"].min()), 
            max_value=int(edges_df["weight"].max()), 
            value=default_min_weight, step=1
        )
        
    with col2:
        default_min_freq = st.session_state.get('auto_min_freq', 5)
        min_node_freq = st.slider(
            "Min node frequency", 
            min_value=int(nodes_df["frequency"].min()), 
            max_value=int(nodes_df["frequency"].max()), 
            value=default_min_freq, step=1
        )

    # Category filter
    categories = sorted(nodes_df["category"].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Filter by category", 
        categories, 
        default=categories
    )

    # Node type filter
    node_types = sorted(nodes_df["type"].dropna().unique())
    selected_types = st.sidebar.multiselect(
        "Filter by node type", 
        node_types, 
        default=node_types
    )

    # Priority score filter
    default_min_priority = st.session_state.get('auto_min_priority', 0.2)
    min_priority_score = st.sidebar.slider(
        "Min priority score", 
        min_value=0.0, 
        max_value=1.0, 
        value=default_min_priority, 
        step=0.05
    )

    # User-based node selection and exclusion
    st.sidebar.subheader("🔍 Node Inclusion/Exclusion")
    
    default_selected = st.session_state.get('auto_selected_nodes', 
                                           ["electrode cracking", "SEI formation", "capacity fade"])
    # Filter to only those that exist in graph
    default_selected = [n for n in default_selected if n in G.nodes()]
    
    selected_nodes = st.sidebar.multiselect(
        "Include specific nodes (optional)", 
        options=sorted(G.nodes()),
        default=default_selected
    )
    
    default_excluded = st.session_state.get('auto_excluded', 'battery, material')
    excluded_terms_input = st.sidebar.text_input(
        "Exclude terms (comma-separated)", 
        value=default_excluded
    )
    excluded_terms = [t.strip().lower() for t in excluded_terms_input.split(',') if t.strip()]

    # Highlight high-priority nodes
    st.sidebar.subheader("🎯 Priority Highlighting")
    default_highlight = st.session_state.get('auto_highlight', True)
    highlight_priority = st.sidebar.checkbox("Highlight high-priority nodes", value=default_highlight)
    
    default_threshold = st.session_state.get('auto_priority_threshold', 0.7)
    priority_threshold = st.sidebar.slider("Priority highlight threshold", 0.5, 1.0, default_threshold, step=0.05)
    
    default_suppress = st.session_state.get('auto_suppress_low', False)
    suppress_low_priority = st.sidebar.checkbox("Suppress low-priority nodes", value=default_suppress)

    # Label size controls
    st.sidebar.subheader("📝 Label Settings")
    show_labels = st.sidebar.checkbox("Show Node Labels", value=True)
    label_font_size = st.sidebar.slider("Label Font Size", 10, 100, 16)
    label_max_chars = st.sidebar.slider("Max Characters per Label", 10, 30, 15)
    
    # Edge width control
    edge_width_factor = st.sidebar.slider("Edge Width Factor", 0.1, 2.0, 0.5)

    # ==================== APPLY FILTERS ====================
    # Use parameters from query if available
    if st.session_state.last_params:
        params = st.session_state.last_params
        # Override sidebar values with parsed ones where appropriate
        min_weight = params.get('min_weight', min_weight)
        min_node_freq = params.get('min_freq', min_node_freq)
        min_priority_score = params.get('min_priority_score', min_priority_score)
        highlight_priority = params.get('highlight_priority', highlight_priority)
        priority_threshold = params.get('priority_threshold', priority_threshold)
        suppress_low_priority = params.get('suppress_low_priority', suppress_low_priority)
        
        # Get analysis type for later
        analysis_type = params.get('analysis_type', 'Centrality Analysis')
    else:
        analysis_type = "Centrality Analysis"  # default

    # Apply filter_graph function
    G_filtered = filter_graph(G, min_weight, min_node_freq, selected_categories, selected_types, 
                               selected_nodes, excluded_terms, min_priority_score, suppress_low_priority)

    # Show graph stats
    st.sidebar.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # ==================== RUN ANALYSIS ====================
    if G_filtered.number_of_nodes() > 0 and st.session_state.last_params:
        # Run the specified analysis
        analysis_results = run_analysis(analysis_type, params, G_filtered, nodes_df, edges_df)
        st.session_state.last_analysis_results = analysis_results
        
        # Generate insights if requested
        if generate_insights and st.session_state.llm_tokenizer is not None and analysis_results is not None:
            with st.spinner("🤖 Generating intelligent insights..."):
                insights = st.session_state.insight_generator.generate_insights(
                    analysis_results, analysis_type, user_query, 
                    relevance_score if 'relevance_score' in locals() else 0.5,
                    st.session_state.llm_tokenizer, st.session_state.llm_model
                )
                st.session_state.last_insights = insights
        else:
            st.session_state.last_insights = None
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': user_query,
            'analysis_type': analysis_type,
            'params': params,
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
        
        # Show confidence
        if 'relevance_score' in locals():
            conf_class = "confidence-high" if relevance_score >= 0.8 else \
                        "confidence-moderate" if relevance_score >= 0.6 else \
                        "confidence-low" if relevance_score >= 0.4 else "confidence-very-low"
            st.markdown(f"<span class='{conf_class}'>Confidence: {confidence_text}</span>", 
                       unsafe_allow_html=True)

    # ==================== VISUALIZATION ====================
    if G_filtered.number_of_nodes() > 0:
        # COLOR NODES BY CATEGORY
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
        
        # Display category legend in sidebar
        st.sidebar.subheader("🎨 Category Legend")
        for category, color in category_color_map.items():
            st.sidebar.markdown(f"<span style='color: {color}; font-size: 14px;'>●</span> {category}", 
                               unsafe_allow_html=True)
        
        # Node Positions
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
        edge_weights = []
        for edge in G_filtered.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G_filtered.edges[edge].get("weight", 1))
        
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            if max_weight > min_weight:
                edge_widths = [0.5 + 4.5 * edge_width_factor * (w - min_weight) / (max_weight - min_weight) for w in edge_weights]
            else:
                edge_widths = [2 * edge_width_factor] * len(edge_weights)
        else:
            edge_widths = [2 * edge_width_factor] * len(edge_weights)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
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
                textfont=dict(size=label_font_size, color='black', family='Arial, bold'),
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
        
        fig.update_layout(
            title=dict(
                text=f'Battery Degradation Knowledge Graph - {analysis_type}',
                font=dict(size=24, family='Arial, bold')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
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
        
        # Export static visualization
        st.sidebar.subheader("📤 Export")
        if st.sidebar.button("Generate Static Image"):
            static_fig = create_static_visualization(G_filtered, pos, node_colors, node_sizes)
            img_str = fig_to_base64(static_fig)
            href = f'<a href="data:image/png;base64,{img_str}" download="knowledge_graph.png">Download PNG</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
            
    else:
        st.warning("No nodes match the current filter criteria. Try adjusting the filters.")

    # ==================== ANALYSIS RESULTS DISPLAY ====================
    if st.session_state.last_analysis_results is not None:
        st.markdown(f"### 📊 {analysis_type} Results")
        
        results = st.session_state.last_analysis_results
        
        if analysis_type == "Centrality Analysis" and isinstance(results, pd.DataFrame) and not results.empty:
            # Display as table and chart
            st.dataframe(results[['node', 'degree', 'betweenness', 'closeness', 'category']].head(20))
            
            fig = px.scatter(
                results, x='degree', y='betweenness',
                color='category', size='eigenvector',
                hover_data=['node', 'closeness'],
                title=f"Centrality of Failure-Related Terms"
            )
            st.plotly_chart(fig)
        
        elif analysis_type == "Community Detection" and isinstance(results, dict):
            for comm_id, data in list(results.items())[:5]:  # First 5 communities
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
        
        elif analysis_type == "Temporal Analysis" and isinstance(results, dict):
            if "error" in results:
                st.warning(results["error"])
            else:
                df_temp = pd.DataFrame(results).T
                st.line_chart(df_temp[['total_concepts', 'failure_concepts']])

    # ==================== HUB NODES AND EXPLORER ====================
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
            st.sidebar.write(f"**Category:** {node_data.get('category','N/A')}")
            st.sidebar.write(f"**Type:** {node_data.get('type','N/A')}")
            st.sidebar.write(f"**Frequency:** {node_data.get('frequency', 'N/A')}")
            st.sidebar.write(f"**Priority Score:** {node_data.get('priority_score', 0):.3f}")
            
            node_centrality = centrality_df[centrality_df['node'] == selected_node]
            if not node_centrality.empty:
                st.sidebar.write(f"**Degree Centrality:** {node_centrality['degree'].values[0]:.3f}")
                st.sidebar.write(f"**Betweenness Centrality:** {node_centrality['betweenness'].values[0]:.3f}")
            
            neighbors = list(G_filtered.neighbors(selected_node))
            if neighbors:
                st.sidebar.write("**Connected Terms:**")
                for n in neighbors[:10]:
                    w = G_filtered.edges[selected_node, n].get("weight", 1)
                    st.sidebar.write(f"- {n} (weight: {w})")
            else:
                st.sidebar.write("No connected terms above current filter threshold.")
        
        # Additional Analytics
        st.subheader("📊 Graph Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = nodes_df['category'].value_counts()
            fig_cat = px.pie(values=category_counts.values, names=category_counts.index, 
                            title="Node Distribution by Category")
            fig_cat.update_layout(title_font_size=18, font=dict(size=14))
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            top_nodes = nodes_df.nlargest(10, 'priority_score')[['node', 'priority_score']]
            fig_nodes = px.bar(top_nodes, x='priority_score', y='node', orientation='h',
                              title="Top Nodes by Priority Score")
            fig_nodes.update_layout(title_font_size=18, font=dict(size=14))
            st.plotly_chart(fig_nodes, use_container_width=True)
        
        # Community analysis
        if G_filtered.number_of_nodes() > 0:
            st.subheader("👥 Community Analysis")
            try:
                G_weighted = nx.Graph()
                for n, d in G_filtered.nodes(data=True):
                    G_weighted.add_node(n, **d)
                for u, v, d in G_filtered.edges(data=True):
                    G_weighted.add_edge(u, v, weight=d.get("weight", 1))
                
                partition = community_louvain.best_partition(G_weighted, weight='weight')
                comm_map = partition
                
                community_summary = {}
                for node, comm_id in comm_map.items():
                    if comm_id not in community_summary:
                        community_summary[comm_id] = []
                    community_summary[comm_id].append(node)
                
                for comm_id, nodes in list(community_summary.items())[:5]:
                    with st.expander(f"Community {comm_id} ({len(nodes)} nodes)"):
                        st.write("**Nodes:** " + ", ".join(nodes[:10]))
                        if len(nodes) > 10:
                            st.write(f"*... and {len(nodes) - 10} more*")
            except:
                st.write("Community detection not available for this graph configuration.")

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

    # ==================== POST-PROCESSING GUIDE ====================
    with st.expander("📖 Post-Processing Guide: How to Study Battery Failure Mechanisms"):
        st.markdown("""
        ## Post-Processing Techniques for Battery Failure Analysis
        
        ### 1. Centrality Analysis
        - **Purpose**: Identify the most important failure mechanisms
        - **Interpretation**: High degree = many connections; High betweenness = bridges between clusters
        
        ### 2. Community Detection
        - **Purpose**: Discover groups of related failure concepts
        - **Look for**: Communities focused on mechanical vs. chemical degradation
        
        ### 3. Ego Network Analysis
        - **Purpose**: Study specific failure mechanisms in detail
        - **Examine**: Immediate connections and network properties
        
        ### 4. Pathway Analysis
        - **Purpose**: Find connections between different failure mechanisms
        - **Analyze**: Shortest paths reveal mechanistic relationships
        
        ### 5. Temporal Analysis
        - **Purpose**: Analyze how failure concepts evolve over time
        - **Trends**: Emerging vs. declining research areas
        
        ### 6. Correlation Analysis
        - **Purpose**: Identify relationships between failure mechanisms
        - **Focus**: Strong correlations indicate coupled degradation modes
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.text("Detailed error information:")
    st.text(traceback.format_exc())
