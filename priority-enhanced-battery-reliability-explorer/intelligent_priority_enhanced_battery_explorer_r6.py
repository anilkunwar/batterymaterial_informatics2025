#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT BATTERY DEGRADATION KNOWLEDGE EXPLORER
===================================================
Expanded version with performance optimizations, mathematical robustness,
LLM enhancements, advanced graph analytics, uncertainty quantification,
scalability, UX improvements, and physics integration.
"""

import os
import pathlib
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import spearmanr, mannwhitneyu
import json
import re
import hashlib
from datetime import datetime
import warnings
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import logging
import functools
from multiprocessing import Pool, cpu_count
import pickle
import sympy as sp
from pydantic import BaseModel, validator, ValidationError
import faiss
import redis
from scipy.stats import bootstrap

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Determine if transformers are available (optional)
# ----------------------------------------------------------------------------
try:
    from transformers import AutoModel, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = AutoTokenizer = GPT2Tokenizer = GPT2LMHeadModel = AutoModelForCausalLM = None
    torch = None

# ----------------------------------------------------------------------------
# Import optional libraries (graceful fallback)
# ----------------------------------------------------------------------------
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Fast similarity search disabled.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Caching disabled.")

# ============================================================================
# GLOBAL CONFIGURATION – DATA DIRECTORY DETECTION
# ============================================================================
def get_data_dir() -> str:
    env_dir = os.environ.get("BATTERY_DATA_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
        if p.is_dir():
            return str(p.resolve())

    def has_data_files(dir_path: pathlib.Path) -> bool:
        return (dir_path / "knowledge_graph_nodes.csv").exists() or (dir_path / "knowledge_graph_edges.csv").exists()

    if '__file__' in globals():
        script_dir = pathlib.Path(__file__).resolve().parent
        if has_data_files(script_dir):
            return str(script_dir)
        data_sub = script_dir / "data"
        if data_sub.is_dir() and has_data_files(data_sub):
            return str(data_sub)

    cwd = pathlib.Path.cwd()
    if has_data_files(cwd):
        return str(cwd)
    cwd_data = cwd / "data"
    if cwd_data.is_dir() and has_data_files(cwd_data):
        return str(cwd_data)

    return str(pathlib.Path(__file__).resolve().parent) if '__file__' in globals() else str(cwd)

DB_DIR = get_data_dir()
logger.info(f"Data directory set to: {DB_DIR}")

# ============================================================================
# PHYSICS EQUATIONS AND KEY TERMS (unchanged)
# ============================================================================
PHYSICS_EQUATIONS = {
    "diffusion_stress": {
        "equation": r"σ = \frac{E \Omega \Delta c}{1 - \nu}",
        "description": "Diffusion-induced stress in active particles",
        "variables": {"E": "Young's modulus (GPa)", "Ω": "Partial molar volume (m³/mol)",
                      "Δc": "Concentration gradient (mol/m³)", "ν": "Poisson's ratio"},
        "unit": "MPa"
    },
    "sei_growth": {
        "equation": r"δ_{SEI} \propto \sqrt{t}",
        "description": "SEI layer growth kinetics (parabolic)",
        "variables": {"δ_SEI": "SEI thickness (nm)", "t": "Time (hours)"},
        "unit": "nm"
    },
    "chemo_mechanical": {
        "equation": r"LAM \propto \int \sigma \, dN",
        "description": "Loss of Active Material from stress cycling",
        "variables": {"LAM": "Loss of Active Material (%)", "σ": "Stress (MPa)", "N": "Cycle number"},
        "unit": "%"
    },
    "lithium_plating": {
        "equation": r"Risk \uparrow \text{ when } V < V_{plate} \text{ or } T < 0°C",
        "description": "Lithium plating risk conditions",
        "variables": {"V": "Cell voltage (V)", "V_plate": "Plating potential (~0.1V vs Li/Li+)", "T": "Temperature (°C)"},
        "unit": "V/°C"
    },
    "capacity_fade": {
        "equation": r"Q_{loss} = Q_0 - \int I(t) \, dt",
        "description": "Integrated current loss over time",
        "variables": {"Q_loss": "Capacity loss (mAh)", "Q_0": "Initial capacity (mAh)", "I": "Current (A)"},
        "unit": "%"
    },
    "crack_propagation": {
        "equation": r"\frac{da}{dN} = C(\Delta K)^m",
        "description": "Paris law for fatigue crack growth",
        "variables": {"a": "Crack length (μm)", "N": "Cycle number", "ΔK": "Stress intensity factor (MPa·m^½)", "C, m": "Material constants"},
        "unit": "μm"
    },
    "nernst_equation": {
        "equation": r"E = E^0 - \frac{RT}{nF} \ln Q",
        "description": "Electrode potential under non-standard conditions",
        "variables": {"E": "Cell potential (V)", "E^0": "Standard potential (V)", "R": "Gas constant", "T": "Temperature (K)", "n": "Number of electrons", "F": "Faraday constant", "Q": "Reaction quotient"},
        "unit": "V"
    },
    "butler_volmer": {
        "equation": r"j = j_0 \left[ \exp\left(\frac{\alpha n F \eta}{RT}\right) - \exp\left(-\frac{(1-\alpha) n F \eta}{RT}\right) \right]",
        "description": "Electrode kinetics (current density)",
        "variables": {"j": "Current density (A/m²)", "j_0": "Exchange current density", "α": "Charge transfer coefficient", "η": "Overpotential (V)"},
        "unit": "A/m²"
    },
    "heat_generation": {
        "equation": r"Q_{gen} = I^2 R + I T \frac{dU}{dT}",
        "description": "Total heat generation (Joule + reversible)",
        "variables": {"Q_gen": "Heat generation rate (W)", "I": "Current (A)", "R": "Internal resistance (Ω)", "dU/dT": "Entropy coefficient (V/K)"},
        "unit": "W"
    },
    "thermal_runaway": {
        "equation": r"\frac{dT}{dt} = \frac{Q_{gen} - Q_{diss}}{m C_p}",
        "description": "Temperature rise rate",
        "variables": {"T": "Temperature (°C)", "Q_gen": "Heat generation (W)", "Q_diss": "Heat dissipation (W)", "m": "Mass (kg)", "C_p": "Specific heat (J/kg·K)"},
        "unit": "°C"
    },
    "calendar_aging": {
        "equation": r"Q_{loss} \propto \sqrt{t} \cdot \exp\left(-\frac{E_a}{RT}\right)",
        "description": "Calendar aging with Arrhenius temperature dependence",
        "variables": {"Q_loss": "Capacity loss (%)", "t": "Time (months)", "E_a": "Activation energy (kJ/mol)", "T": "Temperature (K)"},
        "unit": "%/year"
    },
    "cycle_aging": {
        "equation": r"Q_{loss} = A \cdot N^z \cdot \text{DOD}^b",
        "description": "Cycle aging model",
        "variables": {"N": "Cycle number", "DOD": "Depth of Discharge (%)", "A, z, b": "Fitting parameters"},
        "unit": "cycles"
    },
    "diffusion_coefficient": {
        "equation": r"D = D_0 \exp\left(-\frac{E_a}{RT}\right)",
        "description": "Temperature-dependent diffusion coefficient",
        "variables": {"D": "Diffusion coefficient (cm²/s)", "D_0": "Pre-exponential factor", "E_a": "Activation energy (kJ/mol)"},
        "unit": "cm²/s"
    },
    "migration_flux": {
        "equation": r"J = -D \nabla c + \frac{z F D c}{RT} \nabla \phi",
        "description": "Nernst-Planck ion flux (diffusion + migration)",
        "variables": {"J": "Ion flux (mol/m²·s)", "c": "Concentration (mol/m³)", "φ": "Electric potential (V)"},
        "unit": "mol/m²·s"
    },
    "gas_evolution": {
        "equation": r"V_{gas} \propto \int I_{parasitic} \, dt",
        "description": "Gas volume from parasitic reactions",
        "variables": {"V_gas": "Gas volume (mL)", "I_parasitic": "Parasitic current (mA)"},
        "unit": "mL"
    }
}

PHYSICS_TERMS = [
    "diffusion-induced stress", "SEI formation", "chemo-mechanical coupling",
    "lithium plating", "mechanical degradation", "stress concentration",
    "fracture toughness", "crack propagation", "particle cracking",
    "electrode swelling", "volume expansion", "strain localization",
    "solid electrolyte interphase", "charge transfer", "overpotential",
    "exchange current density", "double layer", "intercalation",
    "deintercalation", "redox reaction", "electrochemical kinetics",
    "thermal runaway", "heat generation", "temperature gradient",
    "cooling system", "thermal management", "arrhenius behavior",
    "cycle life", "calendar aging", "capacity fade", "parasitic reactions",
    "gas evolution", "electrolyte decomposition", "cathode degradation",
    "anode degradation", "transition metal dissolution", "oxygen release",
    "ion diffusion", "mass transport", "concentration gradient",
    "migration flux", "conductivity", "permeability",
    "thermal stability", "short circuit", "internal resistance",
    "impedance growth", "voltage drop", "current density"
]

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

OPERATIONAL_CONSTRAINTS = {
    "c_rate": {"min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1, "unit": "C"},
    "voltage": {"min": 2.5, "max": 4.5, "default": 3.7, "step": 0.1, "unit": "V"},
    "temperature": {"min": -20.0, "max": 80.0, "default": 25.0, "step": 1.0, "unit": "°C"},
    "soc": {"min": 0.0, "max": 100.0, "default": 50.0, "step": 5.0, "unit": "%"},
    "dod": {"min": 0.0, "max": 100.0, "default": 80.0, "step": 5.0, "unit": "%"}
}

# ============================================================================
# DATA CLASS FOR PARSED PARAMETERS (extended with confidence and uncertainty)
# ============================================================================
@dataclass
class ParsedParameters:
    """Container for all parameters extracted from a natural language query."""
    min_weight: int = 10
    min_freq: int = 5
    min_priority_score: float = 0.2
    priority_threshold: float = 0.7
    edge_width_factor: float = 0.5
    label_font_size: int = 16
    label_max_chars: int = 15
    selected_categories: List[str] = field(default_factory=list)
    selected_types: List[str] = field(default_factory=list)
    selected_nodes: List[str] = field(default_factory=list)
    excluded_terms: List[str] = field(default_factory=list)
    suppress_low_priority: bool = False
    highlight_priority: bool = True
    show_labels: bool = True
    analysis_type: str = "Centrality Analysis"
    focus_terms: List[str] = field(default_factory=lambda: ['crack', 'fracture', 'degradation', 'fatigue', 'damage'])
    source_terms: List[str] = field(default_factory=lambda: ['electrode cracking'])
    target_terms: List[str] = field(default_factory=lambda: ['capacity fade'])
    central_nodes: List[str] = field(default_factory=lambda: ['electrode cracking', 'SEI formation', 'capacity fade'])
    time_column: str = "year"
    physics_boost_weight: float = 0.15
    require_physics_in_pathways: bool = False
    min_physics_similarity: float = 0.5
    c_rate: float = 1.0
    voltage: float = 3.7
    temperature: float = 25.0
    soc: float = 50.0
    dod: float = 80.0
    confidence_score: float = 0.0
    parsing_method: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    uncertainty: Dict[str, float] = field(default_factory=dict)  # placeholder for bootstrap uncertainties

    def to_dict(self) -> Dict:
        return asdict(self)

# ============================================================================
# PYDANTIC MODEL FOR LLM OUTPUT VALIDATION
# ============================================================================
class ParsedQuery(BaseModel):
    analysis_type: str
    focus_terms: List[str] = []
    physics_boost_weight: float = 0.15
    c_rate: float = 1.0
    voltage: float = 3.7
    temperature: float = 25.0
    source_terms: List[str] = []
    target_terms: List[str] = []
    central_nodes: List[str] = []
    require_physics_in_pathways: bool = False

    @validator('physics_boost_weight')
    def validate_boost(cls, v):
        return max(0.0, min(0.5, v))

    @validator('c_rate')
    def validate_c_rate(cls, v):
        return max(0.1, min(10.0, v))

    @validator('voltage')
    def validate_voltage(cls, v):
        return max(2.5, min(4.5, v))

    @validator('temperature')
    def validate_temperature(cls, v):
        return max(-20.0, min(80.0, v))

# ============================================================================
# SCIBERT LOADER & EMBEDDING UTILITIES (cached, with FAISS)
# ============================================================================
scibert_tokenizer = None
scibert_model = None
KEY_TERMS_EMBEDDINGS = []
PHYSICS_TERMS_EMBEDDINGS = []
EMBEDDING_INDEX = None  # FAISS index
EMBEDDING_INDEX_NODES = []  # list of node names corresponding to index

@st.cache_resource
def load_scibert():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"Failed to load SciBERT: {str(e)}. Semantic similarity will be disabled.")
        return None, None

def get_scibert_embedding(texts):
    global scibert_tokenizer, scibert_model
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

@st.cache_data
def compute_embeddings(texts):
    return get_scibert_embedding(texts)

def build_faiss_index(embeddings, nodes):
    if not FAISS_AVAILABLE:
        return None, None
    if not embeddings:
        return None, None
    # Filter out None embeddings
    valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
    if not valid_idx:
        return None, None
    valid_embeddings = np.array([embeddings[i] for i in valid_idx], dtype=np.float32)
    valid_nodes = [nodes[i] for i in valid_idx]
    index = faiss.IndexFlatIP(valid_embeddings.shape[1])  # inner product (cosine for normalized vectors)
    index.add(valid_embeddings)
    return index, valid_nodes

def fast_similarity_search(query_emb, index, nodes_list, k=10):
    """Return indices and scores of top k similar nodes."""
    if query_emb is None or index is None:
        return [], []
    if len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
    scores, indices = index.search(query_emb.astype(np.float32), k)
    # scores are cosine similarity because vectors are normalized
    return [nodes_list[i] for i in indices[0]], scores[0]

# ============================================================================
# UNCERTAINTY QUANTIFICATION FUNCTIONS
# ============================================================================
def bootstrap_centrality(G, n_samples=100, metric='degree'):
    """Compute bootstrap confidence intervals for a centrality measure."""
    if G.number_of_nodes() == 0:
        return {}, {}, {}
    # Create a copy of G with all nodes
    nodes = list(G.nodes())
    # Collect all edges
    edges = list(G.edges())
    n_edges = len(edges)
    centrality_samples = []
    for _ in range(n_samples):
        # Sample edges with replacement (bootstrap)
        sampled_edges = np.random.choice(len(edges), size=n_edges, replace=True)
        G_sample = nx.Graph()
        G_sample.add_nodes_from(nodes)
        for i in sampled_edges:
            u, v = edges[i]
            G_sample.add_edge(u, v, **G.edges[u, v])
        if metric == 'degree':
            cent = nx.degree_centrality(G_sample)
        elif metric == 'betweenness':
            cent = nx.betweenness_centrality(G_sample)
        elif metric == 'closeness':
            cent = nx.closeness_centrality(G_sample)
        elif metric == 'eigenvector':
            try:
                cent = nx.eigenvector_centrality(G_sample, max_iter=1000)
            except:
                cent = {n: 0 for n in nodes}
        else:
            cent = nx.degree_centrality(G_sample)
        centrality_samples.append(cent)

    # Compute mean, std, and 95% CI
    mean_cent = {}
    std_cent = {}
    ci_lower = {}
    ci_upper = {}
    for n in nodes:
        vals = [c[n] for c in centrality_samples if n in c]
        if vals:
            mean_cent[n] = np.mean(vals)
            std_cent[n] = np.std(vals)
            ci_lower[n] = np.percentile(vals, 2.5)
            ci_upper[n] = np.percentile(vals, 97.5)
        else:
            mean_cent[n] = 0
            std_cent[n] = 0
            ci_lower[n] = 0
            ci_upper[n] = 0
    return mean_cent, ci_lower, ci_upper

def monte_carlo_priority_score(G, nodes_df, n_samples=50, **kwargs):
    """Compute Monte Carlo uncertainty for priority scores."""
    base_priority = calculate_priority_scores(G, nodes_df, **kwargs)
    samples = []
    for _ in range(n_samples):
        # Slightly perturb frequencies and weights
        nodes_df_perturbed = nodes_df.copy()
        # Add small noise to frequencies
        nodes_df_perturbed['frequency'] = nodes_df_perturbed['frequency'] * (1 + np.random.normal(0, 0.05))
        # Perturb operational parameters
        op_params = kwargs.get('operational_params', {})
        pert_op = {}
        for k, v in op_params.items():
            pert_op[k] = v * (1 + np.random.normal(0, 0.1))
        priority = calculate_priority_scores(G, nodes_df_perturbed,
                                            physics_boost_weight=kwargs.get('physics_boost_weight', 0.15),
                                            operational_params=pert_op,
                                            focus_terms=kwargs.get('focus_terms'))
        samples.append(priority)
    # Compute mean and std for each node
    nodes = list(G.nodes())
    mean_priority = {}
    std_priority = {}
    for n in nodes:
        vals = [s[n] for s in samples if n in s]
        if vals:
            mean_priority[n] = np.mean(vals)
            std_priority[n] = np.std(vals)
        else:
            mean_priority[n] = base_priority.get(n, 0)
            std_priority[n] = 0
    return mean_priority, std_priority

# ============================================================================
# ADVANCED GRAPH ANALYTICS
# ============================================================================
def multi_resolution_community(G, resolutions=[0.5, 1.0, 1.5, 2.0], weight='weight'):
    """Run Louvain community detection at multiple resolutions and compute stability."""
    if G.number_of_nodes() == 0:
        return {}
    partitions = {}
    for r in resolutions:
        try:
            partitions[r] = community_louvain.best_partition(G, weight=weight, resolution=r)
        except:
            partitions[r] = {n: 0 for n in G.nodes()}
    # Compute stability as NMI between each resolution and the baseline (1.0)
    baseline = partitions.get(1.0, {})
    stability = {}
    for r, part in partitions.items():
        if r == 1.0:
            stability[r] = 1.0
        else:
            # Convert to list of labels
            labels1 = [baseline.get(n, 0) for n in G.nodes()]
            labels2 = [part.get(n, 0) for n in G.nodes()]
            try:
                nmi = normalized_mutual_info_score(labels1, labels2)
            except:
                nmi = 0
            stability[r] = nmi
    return {'partitions': partitions, 'stability': stability}

def comprehensive_centrality(G):
    """Compute multiple centrality measures."""
    if G.number_of_nodes() == 0:
        return {}
    return {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G, max_iter=1000) if G.number_of_nodes() > 1 else {n: 1 for n in G.nodes()},
        'pagerank': nx.pagerank(G, weight='weight'),
        'katz': nx.katz_centrality(G, alpha=0.1, beta=1.0) if G.number_of_nodes() > 1 else {n: 1 for n in G.nodes()}
    }

def k_shortest_paths(G, source, target, k=5, weight='weight'):
    """Return up to k shortest paths between source and target."""
    try:
        return list(nx.shortest_simple_paths(G, source, target, weight=weight))[:k]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

def trend_detection(values):
    """Perform Mann-Kendall trend test on a sequence of values."""
    from scipy.stats import mstats
    # Mann-Kendall test
    n = len(values)
    if n < 3:
        return {'trend': 'insufficient data', 'p': 1.0}
    # Use scipy's implementation if available, else simple correlation
    try:
        from scipy.stats import kendalltau
        tau, p = kendalltau(range(n), values)
        trend = 'increasing' if tau > 0 else 'decreasing' if tau < 0 else 'no trend'
        return {'trend': trend, 'p': p, 'tau': tau}
    except:
        # fallback to Pearson correlation
        r, p = pearsonr(range(n), values)
        trend = 'increasing' if r > 0.1 else 'decreasing' if r < -0.1 else 'no trend'
        return {'trend': trend, 'p': p, 'correlation': r}

# ============================================================================
# PHYSICS INTEGRATION (symbolic verification)
# ============================================================================
def verify_physics_equation(equation_str, variables):
    """Use sympy to verify that the equation is dimensionally consistent (simplified)."""
    try:
        # Replace LaTeX with sympy syntax
        # This is a simplified placeholder; full LaTeX parsing would require more work.
        # For now, just check that all variables are defined.
        expr = sp.sympify(equation_str.replace('\\', ''))
        # Check if all variables in expr are in variables list
        free_symbols = expr.free_symbols
        missing = [str(s) for s in free_symbols if str(s) not in variables]
        if missing:
            return False, f"Missing variables: {missing}"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def dimensional_analysis(equation, units):
    """Check if all terms have consistent units (stub)."""
    # Placeholder – would require unit database.
    return True, "Dimensional analysis not implemented."

# ============================================================================
# PERFORMANCE OPTIMIZATIONS: PARALLEL EMBEDDING
# ============================================================================
def parallel_embedding_compute(texts, n_workers=None):
    """Compute embeddings in parallel using multiprocessing."""
    if n_workers is None:
        n_workers = cpu_count()
    # Embeddings are not serializable; we cannot parallelize the model.
    # Instead, we'll use batch processing. For true parallelism, we'd need to load models in each process,
    # which is heavy. So we'll just chunk and run sequentially with tqdm.
    # Using st.progress for UI.
    chunk_size = max(1, len(texts) // (n_workers * 2))
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    embeddings = []
    for i, chunk in enumerate(chunks):
        # Update progress (optional)
        emb = get_scibert_embedding(chunk)
        embeddings.extend(emb)
    return embeddings

# ============================================================================
# CACHING UTILITIES (Redis, disk)
# ============================================================================
def get_cache():
    """Get a Redis client if available."""
    if REDIS_AVAILABLE:
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            return r
        except:
            logger.warning("Redis connection failed. Using disk cache.")
    return None

def disk_cache_get(key):
    """Load from file."""
    cache_dir = os.path.join(DB_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def disk_cache_set(key, value):
    """Save to file."""
    cache_dir = os.path.join(DB_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(value, f)

# ============================================================================
# LLM LOADER (cached, with support for larger models)
# ============================================================================
@st.cache_resource(show_spinner="Loading LLM for intelligent parsing...")
def load_llm(backend: str):
    if not TRANSFORMERS_AVAILABLE:
        return None, None, backend
    try:
        if "GPT-2" in backend:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif "Qwen2.5-7B" in backend:
            # 7B model – requires more memory
            model_name = "Qwen/Qwen2.5-7B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
            model.eval()
        elif "Qwen2.5-0.5B" in backend:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
            model.eval()
        else:
            # Default: Qwen2-0.5B
            model_name = "Qwen/Qwen2-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
            model.eval()
        return tokenizer, model, backend
    except Exception as e:
        st.warning(f"⚠️ Failed to load {backend}: {str(e)}")
        return None, None, backend

def adaptive_temperature(query):
    """Determine temperature based on query complexity."""
    # Simple heuristic: longer query -> lower temperature
    return max(0.05, min(0.3, 0.2 * (1 - len(query.split()) / 50)))

def parse_with_validation(text, tokenizer, model) -> ParsedQuery:
    """Use LLM to generate and validate with Pydantic."""
    prompt = f"""Extract battery degradation parameters from the query as JSON. Output only JSON.
Query: "{text}"
JSON:"""
    try:
        if "Qwen" in st.session_state.get('llm_backend_loaded', ''):
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_formatted = prompt
        inputs = tokenizer.encode(prompt_formatted, return_tensors='pt', truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=400, temperature=adaptive_temperature(text),
                                     do_sample=True, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract JSON
        match = re.search(r'\{.*\}', generated, re.DOTALL)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            parsed = json.loads(json_str)
            return ParsedQuery(**parsed)
    except Exception as e:
        logger.warning(f"LLM parsing with validation failed: {e}")
    return ParsedQuery(analysis_type="Centrality Analysis")

# ============================================================================
# RELEVANCE SCORER (with embedding quality)
# ============================================================================
class RelevanceScorer:
    def __init__(self, use_scibert=True):
        global scibert_tokenizer, scibert_model
        self.use_scibert = use_scibert and scibert_tokenizer is not None and scibert_model is not None
    def score_query_to_nodes(self, query: str, nodes_list: List[str]) -> Tuple[float, float]:
        """Return (score, confidence) where confidence is embedding quality."""
        if not query or not nodes_list:
            return 0.5, 0.0
        if self.use_scibert:
            try:
                q_emb = get_scibert_embedding(query)
                if q_emb is None:
                    return 0.5, 0.0
                sample = nodes_list[:100]
                n_emb = get_scibert_embedding(sample)
                valid = [i for i, e in enumerate(n_emb) if e is not None]
                if not valid:
                    return 0.5, 0.0
                sims = [cosine_similarity([q_emb], [n_emb[i]])[0][0] for i in valid]
                avg_sim = np.mean(sims) if sims else 0.5
                # Confidence: low if embeddings are None, else high
                conf = 0.8 if all(e is not None for e in n_emb) else 0.5
                return float(avg_sim), conf
            except:
                return 0.5, 0.0
        else:
            words = set(query.lower().split())
            matches = sum(1 for n in nodes_list[:100] if any(w in n.lower() for w in words))
            return min(1.0, matches / 50.0), 0.3

# ============================================================================
# QUERY-DRIVEN GRAPH RECONSTRUCTION (with FAISS, parallel)
# ============================================================================
def llm_expand_vocabulary(query: str, tokenizer, model) -> List[str]:
    """Use LLM to generate a list of 15‑25 related technical terms."""
    if tokenizer is None or model is None:
        return []
    prompt = f"""You are a battery degradation expert. Given the user query below, output ONLY a JSON list of 15-25 precise technical terms/phrases that are semantically proximal or causally related (including physics mechanisms, failure modes, materials, etc.).
Query: "{query}"
JSON list:"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=300, temperature=adaptive_temperature(query), do_sample=True,
                                     pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r'\[.*\]', generated, re.DOTALL)
        if match:
            lst = json.loads(match.group(0))
            if isinstance(lst, list):
                return lst[:25]
    except Exception as e:
        logger.warning(f"LLM vocabulary expansion failed: {e}")
    return []

def llm_suggest_missing_edges(query: str, current_nodes: List[str], tokenizer, model) -> List[Tuple[str, str, float]]:
    """Ask LLM to suggest 0‑3 missing edges (source, target, weight 1‑10)."""
    if tokenizer is None or model is None:
        return []
    node_sample = current_nodes[:30]
    prompt = f"""Current nodes: {node_sample}
Suggest 0-3 NEW directed edges that should exist for this query but are missing. Output ONLY JSON list of [source, target, weight_0_10].
Query: "{query}"
JSON:"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=200, temperature=adaptive_temperature(query), do_sample=True,
                                     pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r'\[.*\]', generated, re.DOTALL)
        if match:
            edges = json.loads(match.group(0))
            if isinstance(edges, list):
                validated = []
                for e in edges:
                    if isinstance(e, list) and len(e) == 3:
                        src, tgt, w = e
                        if isinstance(w, (int, float)):
                            validated.append((str(src), str(tgt), float(w)))
                return validated[:3]
    except Exception as e:
        logger.warning(f"LLM edge suggestion failed: {e}")
    return []

def reconstruct_graph_with_attention(G_orig: nx.Graph,
                                     user_query: str,
                                     params: Dict,
                                     tokenizer,
                                     model,
                                     term_embeddings_dict: Dict[str, np.ndarray],
                                     max_nodes: int = 250,
                                     use_faiss: bool = True) -> nx.Graph:
    """
    Build a new graph G_influenced from the original G_orig using:
      - Query embedding
      - Vocabulary expansion (LLM)
      - Attention scoring (softmax over cosine similarities)
      - Multiplicative boosts (physics, degree)
      - Edge re-weighting
      - LLM‑suggested missing edges
      - Top‑K node selection (max_nodes)
    """
    # Step 1: Query embedding
    q_emb = get_scibert_embedding(user_query)
    if q_emb is None:
        return G_orig.copy()  # fallback

    # Step 2: Seed terms from parser
    seed_terms = params.get('focus_terms', []) + params.get('source_terms', []) + params.get('target_terms', [])
    seed_terms = [t for t in seed_terms if t]

    # Step 3: LLM vocabulary expansion
    llm_terms = llm_expand_vocabulary(user_query, tokenizer, model)
    all_seed_terms = list(set(seed_terms + llm_terms))

    # Step 4: Build node list and embedding matrix for nodes that have embeddings
    node_list = [n for n in G_orig.nodes() if term_embeddings_dict.get(n) is not None]
    if not node_list:
        return G_orig.copy()
    node_embs = np.stack([term_embeddings_dict[n] for n in node_list])  # shape (|V|, 768)

    # Step 5: Compute raw attention scores
    tau = 0.1
    logits = np.dot(node_embs, q_emb) / tau
    exp_logits = np.exp(logits - np.max(logits))
    attention_raw = exp_logits / np.sum(exp_logits)

    # Step 6: Multiplicative boosts
    # a) Semantic proximity to seed terms
    seed_embs = [term_embeddings_dict.get(s) for s in all_seed_terms if term_embeddings_dict.get(s) is not None]
    if seed_embs:
        seed_embs = np.stack(seed_embs)
        sim_to_seeds = np.max(np.dot(node_embs, seed_embs.T), axis=1)
    else:
        sim_to_seeds = np.ones(len(node_list))

    # b) Physics boost
    phys_boost = np.array([1.3 if any(p in n.lower() for p in PHYSICS_TERMS) else 1.0 for n in node_list])

    # c) Degree boost
    degrees = np.array([G_orig.degree(n) for n in node_list])
    max_deg = max(degrees) if max(degrees) > 0 else 1
    deg_boost = 1 + 0.5 * (degrees / max_deg)

    attention = attention_raw * sim_to_seeds * phys_boost * deg_boost

    # Step 7: Node selection – top‑K by attention
    if len(attention) > max_nodes:
        selected_idx = np.argsort(attention)[-max_nodes:]
        selected_nodes = [node_list[i] for i in selected_idx]
    else:
        selected_nodes = node_list

    # Map node → attention score
    node_attention = {node_list[i]: attention[i] for i in range(len(node_list))}

    # Step 8: Build influenced subgraph
    G_inf = G_orig.subgraph(selected_nodes).copy()

    # Store attention as node attribute
    for n in G_inf.nodes():
        G_inf.nodes[n]['attention'] = node_attention.get(n, 0)

    # Step 9: Re‑weight edges using average attention
    for u, v, d in G_inf.edges(data=True):
        att_u = node_attention.get(u, 0.5)
        att_v = node_attention.get(v, 0.5)
        d['weight'] = d.get('weight', 1) * (att_u + att_v) / 2 * 1.5

    # Step 10: Add LLM‑suggested missing edges
    if tokenizer is not None and model is not None:
        suggested = llm_suggest_missing_edges(user_query, list(G_inf.nodes()), tokenizer, model)
        for src, tgt, w in suggested:
            if src in G_inf and tgt in G_inf and not G_inf.has_edge(src, tgt):
                G_inf.add_edge(src, tgt, weight=w, type="LLM-inferred")

    return G_inf

# ============================================================================
# PRIORITY SCORE CALCULATION (with robust scaling)
# ============================================================================
def robust_normalize(values, epsilon=1e-8):
    median = np.median(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25) + epsilon
    return (values - median) / iqr

def safe_normalize(values, epsilon=1e-8):
    min_v, max_v = np.min(values), np.max(values)
    range_v = max_v - min_v + epsilon
    return (values - min_v) / range_v

def calculate_priority_scores(G, nodes_df, physics_boost_weight=0.15, operational_params=None, focus_terms=None):
    """
    Priority scoring combining:
      - Frequency (0.35)
      - Degree centrality (0.25)
      - Betweenness centrality (0.20)
      - Semantic relevance to focus terms or default KEY_TERMS (0.10)
      - Physics boost (physics_boost_weight, default 0.15)
    Uses robust scaling for frequencies.
    """
    global KEY_TERMS_EMBEDDINGS, PHYSICS_TERMS_EMBEDDINGS
    # Use robust scaling for frequencies
    freq_vals = nodes_df['frequency'].values
    if len(freq_vals) > 0:
        norm_freq = robust_normalize(freq_vals)
        # Clip to [0,1]
        norm_freq = np.clip(norm_freq, 0, 1)
    else:
        norm_freq = np.array([])
    nodes_df['norm_frequency'] = norm_freq if len(norm_freq) > 0 else 0

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
    physics_scores = {}
    for node in G.nodes():
        emb = term_embeddings_dict.get(node)
        if emb is None:
            semantic_scores[node] = 0
            physics_scores[node] = 0
        else:
            # Semantic relevance to focus terms (if provided) else KEY_TERMS
            if focus_terms:
                focus_embs = get_scibert_embedding(focus_terms)
                focus_embs = [e for e in focus_embs if e is not None]
                if focus_embs:
                    sem_sims = [cosine_similarity([emb], [fe])[0][0] for fe in focus_embs]
                    semantic_scores[node] = max(sem_sims, default=0)
                else:
                    semantic_scores[node] = 0
            else:
                sem_sims = [cosine_similarity([emb], [kt_emb])[0][0] for kt_emb in KEY_TERMS_EMBEDDINGS if kt_emb is not None]
                semantic_scores[node] = max(sem_sims, default=0)

            # Physics relevance
            phys_sims = [cosine_similarity([emb], [pt_emb])[0][0] for pt_emb in PHYSICS_TERMS_EMBEDDINGS if pt_emb is not None]
            phys_score = max(phys_sims, default=0)

            # Operational constraint modifiers
            if operational_params:
                if operational_params.get('temperature', 25) > 45 and any(t in node.lower() for t in ['thermal', 'temp']):
                    phys_score *= 1.3
                if operational_params.get('c_rate', 1.0) > 2.0 and any(t in node.lower() for t in ['rate', 'power']):
                    phys_score *= 1.3
                if operational_params.get('voltage', 3.7) > 4.2 and any(t in node.lower() for t in ['voltage', 'potential']):
                    phys_score *= 1.3
            physics_scores[node] = min(phys_score, 1.0)

    w_f, w_d, w_b, w_s, w_p = 0.35, 0.25, 0.20, 0.10, physics_boost_weight
    # Use softmax to renormalize if sum not 1
    weights = np.array([w_f, w_d, w_b, w_s, w_p])
    weights = np.exp(weights) / np.sum(np.exp(weights))
    w_f, w_d, w_b, w_s, w_p = weights

    priority_scores = {}
    for node in G.nodes():
        freq = nodes_df[nodes_df['node'] == node]['norm_frequency'].iloc[0] if len(nodes_df[nodes_df['node'] == node]) > 0 else 0
        priority_scores[node] = (
            w_f * freq +
            w_d * degree_centrality.get(node, 0) +
            w_b * betweenness_centrality.get(node, 0) +
            w_s * semantic_scores.get(node, 0) +
            w_p * physics_scores.get(node, 0)
        )
    return priority_scores

# ============================================================================
# FILTER GRAPH (with connected component analysis)
# ============================================================================
def filter_graph(G, min_weight, min_freq, selected_categories, selected_types, selected_nodes, excluded_terms, min_priority_score, suppress_low_priority):
    Gf = nx.Graph()
    valid = set()
    if selected_nodes:
        for n in selected_nodes:
            if n in G.nodes() and G.nodes[n].get('priority_score', 0) >= min_priority_score:
                valid.add(n)
                valid.update(G.neighbors(n))
    else:
        for n, d in G.nodes(data=True):
            if (d.get('frequency',0) >= min_freq and
                d.get('category','') in selected_categories and
                d.get('type','') in selected_types and
                (not suppress_low_priority or d.get('priority_score',0) >= min_priority_score)):
                valid.add(n)
    valid = {n for n in valid if not any(ex in n.lower() for ex in excluded_terms)}
    for n in valid:
        Gf.add_node(n, **G.nodes[n])
    for u, v, d in G.edges(data=True):
        if u in Gf.nodes and v in Gf.nodes and d.get('weight',0) >= min_weight:
            Gf.add_edge(u, v, **d)
    # Ensure we keep only the largest connected component if graph is disconnected
    if len(Gf) > 0 and not nx.is_connected(Gf):
        largest_cc = max(nx.connected_components(Gf), key=len)
        Gf = Gf.subgraph(largest_cc).copy()
    return Gf

# ============================================================================
# STRUCTURED INSIGHT GENERATOR (with uncertainty)
# ============================================================================
class DegradationInsightGenerator:
    @staticmethod
    def generate_structured_insights(
        analysis_results,
        analysis_type: str,
        parsed_params: Dict,
        graph_stats: Dict,
        user_query: str = "",
        uncertainty: Dict[str, Dict] = None
    ) -> Dict:
        """
        Returns a fully numerical, countable JSON object with optional uncertainty.
        """
        global PHYSICS_TERMS_EMBEDDINGS
        data = {
            "summary": {
                "nodes": graph_stats["nodes"],
                "edges": graph_stats["edges"],
                "analysis_type": analysis_type,
                "query_focus_score": 0.0
            },
            "ranked_mechanisms": [],
            "pathways": [],
            "communities": [],
            "strong_correlations": [],
            "temporal_trend": {},
            "operational_constraints": {
                "c_rate": parsed_params.get("c_rate", 1.0),
                "voltage": parsed_params.get("voltage", 3.7),
                "temperature": parsed_params.get("temperature", 25.0)
            },
            "uncertainty": uncertainty if uncertainty else {}
        }

        if analysis_type == "Centrality Analysis" and isinstance(analysis_results, pd.DataFrame) and not analysis_results.empty:
            top = analysis_results.nlargest(10, "degree")
            for _, row in top.iterrows():
                node_emb = get_scibert_embedding(row["node"])
                phys_match = 0.0
                if node_emb is not None and PHYSICS_TERMS_EMBEDDINGS:
                    sims = [cosine_similarity([node_emb], [pt_emb])[0][0] for pt_emb in PHYSICS_TERMS_EMBEDDINGS if pt_emb is not None]
                    phys_match = max(sims, default=0.0)
                operational_boost = 1.0
                if parsed_params.get("temperature", 25) > 45 and any(t in row["node"].lower() for t in ["thermal", "temp"]):
                    operational_boost = 1.3
                if parsed_params.get("c_rate", 1.0) > 2.0 and any(t in row["node"].lower() for t in ["rate", "power"]):
                    operational_boost = 1.3
                if parsed_params.get("voltage", 3.7) > 4.2 and any(t in row["node"].lower() for t in ["voltage", "potential"]):
                    operational_boost = 1.3
                composite_weight = (
                    0.4 * row["degree"] +
                    0.3 * phys_match +
                    0.2 * operational_boost +
                    0.1 * row.get("betweenness", 0)
                )
                equation = ""
                for eq_name, eq_data in PHYSICS_EQUATIONS.items():
                    if any(k in row["node"].lower() for k in eq_name.lower().split('_')):
                        equation = eq_data["equation"]
                        break
                entry = {
                    "name": row["node"],
                    "degree": round(row["degree"], 3),
                    "betweenness": round(row.get("betweenness", 0), 3),
                    "physics_match": round(phys_match, 3),
                    "operational_boost": round(operational_boost, 2),
                    "composite_weight": round(composite_weight, 3),
                    "equation": equation
                }
                data["ranked_mechanisms"].append(entry)

        elif analysis_type == "Pathway Analysis" and isinstance(analysis_results, dict):
            for name, p in analysis_results.items():
                if p.get("path"):
                    phys_score = p.get("avg_physics_similarity", 0)
                    composite = (0.5 * (1 / (p["length"] + 1)) + 0.5 * phys_score) * 1.0
                    data["pathways"].append({
                        "path": name,
                        "length": p["length"],
                        "avg_physics_similarity": round(phys_score, 3),
                        "composite_weight": round(composite, 3),
                        "contains_physics": p.get("contains_physics", False)
                    })

        elif analysis_type == "Community Detection" and isinstance(analysis_results, dict):
            for cid, c in list(analysis_results.items())[:5]:
                phys_count = sum(c.get("physics_terms", Counter()).values())
                size = len(c["nodes"])
                data["communities"].append({
                    "community_id": cid,
                    "size": size,
                    "physics_terms_count": phys_count,
                    "composite_weight": round(phys_count / max(size, 1), 3)
                })

        elif analysis_type == "Correlation Analysis" and isinstance(analysis_results, tuple) and len(analysis_results)==2:
            corr, terms = analysis_results
            if len(terms) > 0:
                top_corrs = []
                for i in range(min(len(terms),10)):
                    for j in range(i+1, min(len(terms),10)):
                        if corr[i,j] > 0:
                            top_corrs.append((terms[i], terms[j], corr[i,j]))
                top_corrs.sort(key=lambda x: x[2], reverse=True)
                for t1, t2, val in top_corrs[:3]:
                    data["strong_correlations"].append({
                        "term1": t1,
                        "term2": t2,
                        "strength": round(val, 3)
                    })

        elif analysis_type == "Temporal Analysis" and isinstance(analysis_results, dict):
            periods = sorted(analysis_results.keys())
            if periods:
                first, last = periods[0], periods[-1]
                fc_first = analysis_results[first].get('failure_concepts', 0) if isinstance(analysis_results[first], dict) else 0
                fc_last = analysis_results[last].get('failure_concepts', 0) if isinstance(analysis_results[last], dict) else 0
                # Trend detection
                values = [analysis_results[p].get('failure_concepts', 0) for p in periods]
                trend = trend_detection(values)
                data["temporal_trend"] = {
                    "first_period": first,
                    "last_period": last,
                    "failure_concepts_change": fc_last - fc_first,
                    "direction": "increasing" if fc_last > fc_first else "decreasing",
                    "trend_test": trend
                }

        if user_query:
            query_emb = get_scibert_embedding(user_query)
            if query_emb is not None:
                for mech in data["ranked_mechanisms"]:
                    mech_emb = get_scibert_embedding(mech["name"])
                    if mech_emb is not None:
                        sim = cosine_similarity([query_emb], [mech_emb])[0][0]
                        mech["composite_weight"] = round(mech["composite_weight"] * (1 + 0.5 * sim), 3)
                        mech["query_relevance"] = round(sim, 3)
                if data["ranked_mechanisms"]:
                    data["summary"]["query_focus_score"] = round(max(
                        (m.get("query_relevance", 0) for m in data["ranked_mechanisms"]), default=0.0
                    ), 3)

        data["ranked_mechanisms"].sort(key=lambda x: x["composite_weight"], reverse=True)
        data["pathways"].sort(key=lambda x: x["composite_weight"], reverse=True)
        return data

# ============================================================================
# DATA LOADING (cached)
# ============================================================================
@st.cache_data
def load_data():
    nodes_file = os.path.join(DB_DIR, 'knowledge_graph_nodes.csv')
    edges_file = os.path.join(DB_DIR, 'knowledge_graph_edges.csv')
    missing = []
    if not os.path.exists(nodes_file):
        missing.append('knowledge_graph_nodes.csv')
    if not os.path.exists(edges_file):
        missing.append('knowledge_graph_edges.csv')
    if missing:
        st.error(f"❌ Required data file(s) not found in {DB_DIR}: {', '.join(missing)}.")
        st.stop()
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    for col in ['node', 'type', 'category', 'frequency']:
        if col not in nodes_df.columns:
            nodes_df[col] = '' if col != 'frequency' else 0
    for col in ['source', 'target', 'weight']:
        if col not in edges_df.columns:
            edges_df[col] = '' if col != 'weight' else 0
    return edges_df, nodes_df

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.set_page_config(layout="wide", page_title="Intelligent Battery Degradation Explorer")
    st.markdown("<h1 style='text-align:center;'>🔋 Intelligent Battery Degradation Knowledge Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Expanded: performance optimizations, LLM enhancements, advanced analytics, uncertainty quantification, and physics integration.</p>", unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # Initialize global models and embeddings
    # ------------------------------------------------------------------------
    global scibert_tokenizer, scibert_model, KEY_TERMS_EMBEDDINGS, PHYSICS_TERMS_EMBEDDINGS, EMBEDDING_INDEX, EMBEDDING_INDEX_NODES
    scibert_tokenizer, scibert_model = load_scibert()
    KEY_TERMS_EMBEDDINGS = compute_embeddings(KEY_TERMS)
    PHYSICS_TERMS_EMBEDDINGS = compute_embeddings(PHYSICS_TERMS)
    KEY_TERMS_EMBEDDINGS = [emb for emb in KEY_TERMS_EMBEDDINGS if emb is not None]
    PHYSICS_TERMS_EMBEDDINGS = [emb for emb in PHYSICS_TERMS_EMBEDDINGS if emb is not None]

    # Load data
    try:
        edges_df, nodes_df = load_data()
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    def norm(t): return t.lower().strip() if isinstance(t, str) else ""
    nodes_df["node"] = nodes_df["node"].apply(norm)
    edges_df["source"] = edges_df["source"].apply(norm)
    edges_df["target"] = edges_df["target"].apply(norm)

    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["node"], type=r["type"], category=r["category"], frequency=r["frequency"],
                   unit=r.get("unit","None"), similarity_score=r.get("similarity_score",0))
    for _, r in edges_df.iterrows():
        G.add_edge(r["source"], r["target"], weight=r["weight"], type=r["type"], label=r["label"],
                   relationship=r.get("relationship",""), strength=r.get("strength",0))

    G_original = G.copy()

    @st.cache_data
    def compute_node_embeddings_dict(nodes_list):
        emb_list = get_scibert_embedding(nodes_list)
        return dict(zip(nodes_list, emb_list))

    node_list_all = list(G.nodes())
    term_embeddings_dict = compute_node_embeddings_dict(node_list_all)

    # Build FAISS index
    if FAISS_AVAILABLE:
        embeddings_list = [term_embeddings_dict[n] for n in node_list_all]
        EMBEDDING_INDEX, EMBEDDING_INDEX_NODES = build_faiss_index(embeddings_list, node_list_all)
    else:
        EMBEDDING_INDEX, EMBEDDING_INDEX_NODES = None, None

    # Initialize session state
    if "parser" not in st.session_state:
        from dataclasses import asdict
        st.session_state.parser = BatteryNLParser()
    if "relevance_scorer" not in st.session_state:
        st.session_state.relevance_scorer = RelevanceScorer(use_scibert=True)
    if "insight_generator" not in st.session_state:
        st.session_state.insight_generator = DegradationInsightGenerator()
    if "llm_tokenizer" not in st.session_state:
        st.session_state.llm_tokenizer = None
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = None
    if "last_params" not in st.session_state:
        st.session_state.last_params = None
    if "last_analysis_results" not in st.session_state:
        st.session_state.last_analysis_results = None
    if "last_structured_insights" not in st.session_state:
        st.session_state.last_structured_insights = None

    # ------------------------------------------------------------------------
    # Sidebar filters (enhanced with UX)
    # ------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ⚙️ Manual Filters (optional)")
        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.slider("Min edge weight", int(edges_df["weight"].min()), int(edges_df["weight"].max()), 10, 1, key="min_weight")
        with col2:
            min_freq = st.slider("Min node frequency", int(nodes_df["frequency"].min()), int(nodes_df["frequency"].max()), 5, 1, key="min_freq")

        categories = sorted(nodes_df["category"].dropna().unique())
        selected_cats = st.multiselect("Filter by category", categories, default=categories, key="cats")
        types = sorted(nodes_df["type"].dropna().unique())
        selected_types = st.multiselect("Filter by node type", types, default=types, key="types")

        min_priority = st.slider("Min priority score", 0.0, 1.0, 0.2, 0.05, key="min_priority")
        selected_nodes = st.multiselect("Include specific nodes", sorted(G.nodes()), default=["electrode cracking", "SEI formation", "capacity fade"] if all(n in G.nodes() for n in ["electrode cracking", "SEI formation", "capacity fade"]) else [], key="selected_nodes")
        excluded_input = st.text_input("Exclude terms (comma-separated)", value="battery, material", key="excluded")
        excluded_terms = [t.strip().lower() for t in excluded_input.split(',') if t.strip()]

        st.markdown("### 🔬 Physics Settings")
        physics_boost = st.slider("Physics boost weight", 0.0, 0.5, 0.15, 0.05, key="physics_boost")
        require_physics = st.checkbox("Require physics in pathways", value=False, key="require_physics")
        min_phys_sim = st.slider("Min physics similarity", 0.0, 1.0, 0.5, 0.1, key="min_phys_sim")

        st.markdown("### ⚡ Operational Constraints")
        c_rate = st.slider("C-rate", OPERATIONAL_CONSTRAINTS["c_rate"]["min"], OPERATIONAL_CONSTRAINTS["c_rate"]["max"], OPERATIONAL_CONSTRAINTS["c_rate"]["default"], OPERATIONAL_CONSTRAINTS["c_rate"]["step"], key="c_rate")
        voltage = st.slider("Voltage (V)", OPERATIONAL_CONSTRAINTS["voltage"]["min"], OPERATIONAL_CONSTRAINTS["voltage"]["max"], OPERATIONAL_CONSTRAINTS["voltage"]["default"], OPERATIONAL_CONSTRAINTS["voltage"]["step"], key="voltage")
        temperature = st.slider("Temperature (°C)", OPERATIONAL_CONSTRAINTS["temperature"]["min"], OPERATIONAL_CONSTRAINTS["temperature"]["max"], OPERATIONAL_CONSTRAINTS["temperature"]["default"], OPERATIONAL_CONSTRAINTS["temperature"]["step"], key="temperature")
        soc = st.slider("SOC (%)", OPERATIONAL_CONSTRAINTS["soc"]["min"], OPERATIONAL_CONSTRAINTS["soc"]["max"], OPERATIONAL_CONSTRAINTS["soc"]["default"], OPERATIONAL_CONSTRAINTS["soc"]["step"], key="soc")
        dod = st.slider("DOD (%)", OPERATIONAL_CONSTRAINTS["dod"]["min"], OPERATIONAL_CONSTRAINTS["dod"]["max"], OPERATIONAL_CONSTRAINTS["dod"]["default"], OPERATIONAL_CONSTRAINTS["dod"]["step"], key="dod")

        st.markdown("### 🎯 Highlighting")
        highlight = st.checkbox("Highlight high-priority nodes", value=True, key="highlight")
        threshold = st.slider("Highlight threshold", 0.5, 1.0, 0.7, 0.05, key="threshold")
        suppress = st.checkbox("Suppress low-priority", value=False, key="suppress")

        st.markdown("### 📝 Labels")
        show_labels = st.checkbox("Show labels", value=True, key="show_labels")
        label_size = st.slider("Font size", 10, 100, 16, key="label_size")
        max_chars = st.slider("Max chars", 10, 30, 15, key="max_chars")
        edge_width = st.slider("Edge width factor", 0.1, 2.0, 0.5, key="edge_width")

        st.markdown("### 🧠 Query-Driven Reconstruction")
        use_reconstruction = st.checkbox("Reconstruct graph with LLM attention pooling", value=False, key="use_reconstruction")
        max_recon_nodes = st.slider("Max nodes in influenced graph", 50, 600, 250, step=25, key="max_recon_nodes")

        st.markdown("### 🔬 Advanced Analytics")
        use_multi_res = st.checkbox("Multi-resolution communities", value=False, key="multi_res")
        use_bootstrap = st.checkbox("Bootstrap centrality", value=False, key="bootstrap")
        bootstrap_samples = st.slider("Bootstrap samples", 10, 200, 50, key="bootstrap_samples") if use_bootstrap else 50

    # ------------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------------
    with st.expander("🤖 AI-Powered Query Interface", expanded=True):
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            user_query = st.text_area("Ask about battery degradation in natural language:", height=100,
                                      placeholder="e.g., 'Show pathways from electrode cracking to capacity fade involving diffusion-induced stress'",
                                      key="user_query")
        with col2:
            st.markdown("### 🧠 LLM Settings")
            model_choice = st.selectbox("Model", ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", "Qwen2.5-7B-Instruct"], index=0, key="llm_choice")
            if TRANSFORMERS_AVAILABLE:
                tok, mod, loaded = load_llm(model_choice)
                st.session_state.llm_tokenizer = tok
                st.session_state.llm_model = mod
                st.session_state.llm_backend_loaded = loaded
            use_llm = st.checkbox("Use LLM parsing", value=True)
            use_ensemble = st.checkbox("Use ensemble", value=False)
            ensemble_runs = 3 if use_ensemble else 1
        with col3:
            st.markdown("### 🚀 Actions")
            run_button = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
            if st.button("🧹 Clear History", use_container_width=True):
                st.session_state.last_params = None
                st.session_state.last_analysis_results = None
                st.session_state.last_structured_insights = None
                st.rerun()

    if run_button and user_query:
        with st.spinner("🔍 Parsing query..."):
            parser = st.session_state.parser
            if use_llm and st.session_state.llm_tokenizer is not None:
                params = parser.hybrid_parse(user_query, st.session_state.llm_tokenizer, st.session_state.llm_model,
                                             use_ensemble=use_ensemble, ensemble_runs=ensemble_runs)
            else:
                params = parser.parse_regex(user_query)
            # Validate with Pydantic
            try:
                validated = ParsedQuery(**params)
                params = validated.dict()
            except ValidationError as e:
                st.warning(f"LLM output validation failed: {e}. Using regex parsed.")
            st.session_state.last_params = ParsedParameters(**params)

            all_nodes = list(G.nodes())
            relevance, conf = st.session_state.relevance_scorer.score_query_to_nodes(user_query, all_nodes[:100])
            st.info(f"**Semantic Relevance:** {relevance:.3f} (confidence: {conf:.2f})")

            st.markdown("### 📋 Parsed Parameters")
            cols = st.columns(3)
            for i, (k, v) in enumerate(params.items()):
                if k in ['confidence_score','parsing_method','timestamp']:
                    continue
                with cols[i % 3]:
                    st.metric(k.replace('_',' ').title(), str(v)[:50])
    else:
        if st.session_state.last_params is not None:
            params = asdict(st.session_state.last_params)
        else:
            params = asdict(ParsedParameters())

    # Override with sidebar values
    params['min_weight'] = min_weight
    params['min_freq'] = min_freq
    params['selected_categories'] = selected_cats
    params['selected_types'] = selected_types
    params['min_priority_score'] = min_priority
    params['selected_nodes'] = selected_nodes
    params['excluded_terms'] = excluded_terms
    params['physics_boost_weight'] = physics_boost
    params['require_physics_in_pathways'] = require_physics
    params['min_physics_similarity'] = min_phys_sim
    params['c_rate'] = c_rate
    params['voltage'] = voltage
    params['temperature'] = temperature
    params['soc'] = soc
    params['dod'] = dod
    params['highlight_priority'] = highlight
    params['priority_threshold'] = threshold
    params['suppress_low_priority'] = suppress
    params['show_labels'] = show_labels
    params['label_font_size'] = label_size
    params['label_max_chars'] = max_chars
    params['edge_width_factor'] = edge_width

    # ------------------------------------------------------------------------
    # Priority scores (using current params, with Monte Carlo if desired)
    # ------------------------------------------------------------------------
    operational = {"c_rate": c_rate, "voltage": voltage, "temperature": temperature}
    priority_scores = calculate_priority_scores(G, nodes_df, physics_boost_weight=physics_boost,
                                                operational_params=operational,
                                                focus_terms=params.get('focus_terms'))
    for n in G.nodes():
        G.nodes[n]['priority_score'] = priority_scores.get(n, 0)
    nodes_df['priority_score'] = nodes_df['node'].apply(lambda x: priority_scores.get(x, 0))

    # Optional Monte Carlo uncertainty for priority scores
    if use_bootstrap:
        with st.spinner("Computing Monte Carlo uncertainty for priority scores..."):
            mean_priority, std_priority = monte_carlo_priority_score(G, nodes_df, n_samples=bootstrap_samples,
                                                                     physics_boost_weight=physics_boost,
                                                                     operational_params=operational,
                                                                     focus_terms=params.get('focus_terms'))
            # Store uncertainties in node attributes
            for n in G.nodes():
                G.nodes[n]['priority_mean'] = mean_priority.get(n, priority_scores.get(n, 0))
                G.nodes[n]['priority_std'] = std_priority.get(n, 0)
            params['uncertainty'] = {'priority_scores': {n: round(std_priority.get(n, 0), 4) for n in G.nodes() if std_priority.get(n, 0) > 0}}

    G_filtered = filter_graph(G,
                              min_weight=params['min_weight'],
                              min_freq=params['min_freq'],
                              selected_categories=params['selected_categories'],
                              selected_types=params['selected_types'],
                              selected_nodes=params['selected_nodes'],
                              excluded_terms=params['excluded_terms'],
                              min_priority_score=params['min_priority_score'],
                              suppress_low_priority=params['suppress_low_priority'])

    st.sidebar.markdown(f"**Graph Stats (filtered):** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # ------------------------------------------------------------------------
    # Optional: reconstruct influenced graph
    # ------------------------------------------------------------------------
    if use_reconstruction and run_button and user_query:
        with st.spinner("🔄 Reconstructing graph with attention + LLM..."):
            G_influenced = reconstruct_graph_with_attention(
                G_original, user_query, params,
                st.session_state.llm_tokenizer, st.session_state.llm_model,
                term_embeddings_dict,
                max_nodes=max_recon_nodes,
                use_faiss=FAISS_AVAILABLE
            )
            st.session_state.G_influenced = G_influenced
    else:
        G_influenced = G_filtered.copy()

    # ------------------------------------------------------------------------
    # Run analysis on both graphs
    # ------------------------------------------------------------------------
    analysis_type = params.get('analysis_type', 'Centrality Analysis')

    def run_analysis_on_graph(graph):
        if graph.number_of_nodes() == 0:
            return None
        if analysis_type == "Centrality Analysis":
            focus = params.get('focus_terms', ['crack','fracture','degradation'])
            return analyze_failure_centrality(graph, focus)
        elif analysis_type == "Community Detection":
            if use_multi_res:
                return multi_resolution_community(graph, weight='weight')
            else:
                return detect_failure_communities(graph)[0]
        elif analysis_type == "Ego Network Analysis":
            central = params.get('central_nodes', ["electrode cracking","SEI formation","capacity fade"])
            return analyze_ego_networks(graph, central)
        elif analysis_type == "Pathway Analysis":
            src = params.get('source_terms', ['electrode cracking'])
            tgt = params.get('target_terms', ['capacity fade'])
            req = params.get('require_physics_in_pathways', False)
            min_phys = params.get('min_physics_similarity', 0.5)
            return find_failure_pathways(graph, src, tgt, require_physics=req, min_physics_similarity=min_phys)
        elif analysis_type == "Temporal Analysis":
            time_col = params.get('time_column', 'year')
            return analyze_temporal_patterns(nodes_df, edges_df, time_col)
        elif analysis_type == "Correlation Analysis":
            return analyze_failure_correlations(graph)
        else:
            return analyze_failure_centrality(graph)

    results_orig = run_analysis_on_graph(G_filtered) if G_filtered.number_of_nodes() > 0 else None
    results_inf = run_analysis_on_graph(G_influenced) if G_influenced.number_of_nodes() > 0 else None

    graph_stats_orig = {'nodes': G_filtered.number_of_nodes(), 'edges': G_filtered.number_of_edges()}
    graph_stats_inf = {'nodes': G_influenced.number_of_nodes(), 'edges': G_influenced.number_of_edges()}

    # Bootstrap uncertainty for centrality if requested
    uncertainty_orig = {}
    if use_bootstrap and results_orig is not None and analysis_type == "Centrality Analysis":
        with st.spinner("Computing bootstrap centrality uncertainty..."):
            mean_cent, ci_lower, ci_upper = bootstrap_centrality(G_filtered, n_samples=bootstrap_samples, metric='degree')
            uncertainty_orig['degree_centrality'] = {n: {'mean': mean_cent[n], 'lower': ci_lower[n], 'upper': ci_upper[n]} for n in G_filtered.nodes()}

    if results_orig is not None:
        structured_orig = st.session_state.insight_generator.generate_structured_insights(
            results_orig, analysis_type, params, graph_stats_orig, user_query if 'user_query' in locals() else "",
            uncertainty=uncertainty_orig
        )
    else:
        structured_orig = {}

    if results_inf is not None:
        structured_inf = st.session_state.insight_generator.generate_structured_insights(
            results_inf, analysis_type, params, graph_stats_inf, user_query if 'user_query' in locals() else ""
        )
    else:
        structured_inf = {}

    # ------------------------------------------------------------------------
    # NEW: Quantitative Measures Tabs (Reconstruction Mode)
    # ------------------------------------------------------------------------
    if use_reconstruction and results_orig is not None and results_inf is not None:
        st.subheader("📊 Structured Insights Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Filtered Graph**")
            st.json(structured_orig)
        with col2:
            st.markdown("**LLM‑Influenced Graph**")
            st.json(structured_inf)

        metrics = benchmark_graphs(G_filtered, G_influenced, analysis_type)
        
        st.subheader("📏 Current Level of Detachment: A Quantitative Perspective")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Set Overlap", "Degree & Centrality", "Community Structure", "Pathway Analysis"])

        with tab1:
            st.markdown("### Set Theoretic Metrics")
            df_overlap = pd.DataFrame({
                "Metric": ["Node Jaccard", "Edge Jaccard", "Node Count Delta (%)"],
                "Value": [metrics["node_jaccard"], metrics["edge_jaccard"], metrics["node_count_delta_pct"]],
                "Description": [
                    "Similarity of node sets between Original and Influenced graphs.",
                    "Similarity of edge structures.",
                    "Percentage change in the number of nodes."
                ]
            })
            st.dataframe(df_overlap, use_container_width=True)
            fig_pie = px.pie(df_overlap, values="Value", names="Metric", title="Overlap & Size Change", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.markdown("### Degree & Centrality Dynamics")
            col_left, col_right = st.columns(2)
            with col_left:
                avg_deg_filtered = sum(dict(G_filtered.degree()).values()) / G_filtered.number_of_nodes() if G_filtered.number_of_nodes() > 0 else 0
                avg_deg_influenced = sum(dict(G_influenced.degree()).values()) / G_influenced.number_of_nodes() if G_influenced.number_of_nodes() > 0 else 0
                fig_bar_deg = px.bar(
                    x=["Original (filtered)", "Influenced"],
                    y=[avg_deg_filtered, avg_deg_influenced],
                    title="Average Node Degree",
                    color=["Original", "Influenced"],
                    color_discrete_sequence=["#636EFA", "#EF553B"],
                    labels={"x": "Graph Type", "y": "Average Degree"}
                )
                st.plotly_chart(fig_bar_deg, use_container_width=True)
                st.caption(f"Delta: {metrics['avg_degree_delta']:.3f}")
            with col_right:
                fig_bar_cent = px.bar(
                    x=["Spearman ρ degree centrality"],
                    y=[metrics["centrality_correlation"]],
                    title="Centrality Correlation (Spearman)",
                    range_y=[-1,1],
                    labels={"x": "Metric", "y": "Correlation Coefficient"}
                )
                fig_bar_cent.add_hline(y=0, line_dash="dot", annotation_text="no correlation")
                fig_bar_cent.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="perfect correlation")
                st.plotly_chart(fig_bar_cent, use_container_width=True)
                st.caption("Measures rank preservation of node importance.")

        with tab3:
            st.markdown("### Community Structure Similarity")
            fig_nmi = px.bar(
                x=["Community NMI (Louvain)"],
                y=[metrics["community_nmi"]],
                title="Community Similarity (Normalized Mutual Information)",
                range_y=[0,1],
                labels={"x": "Metric", "y": "NMI Score"}
            )
            fig_nmi.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="good similarity")
            fig_nmi.add_hline(y=0.0, line_dash="dot", annotation_text="random")
            st.plotly_chart(fig_nmi, use_container_width=True)
            st.json(metrics)

        with tab4:
            st.markdown("### Pathway Analysis Delta")
            if analysis_type == "Pathway Analysis":
                st.info("Comparing shortest path lengths between source and target terms in both graphs.")
                path_orig = structured_orig.get("pathways", [])
                path_inf = structured_inf.get("pathways", [])
                path_comp = []
                for p_orig in path_orig:
                    name = p_orig["path"]
                    len_orig = p_orig["length"]
                    p_inf = next((p for p in path_inf if p["path"] == name), None)
                    if p_inf:
                        len_inf = p_inf["length"]
                        delta = len_orig - len_inf
                        path_comp.append({
                            "Path": name,
                            "Length (Original)": len_orig,
                            "Length (Influenced)": len_inf,
                            "Delta": delta
                        })
                if path_comp:
                    st.dataframe(pd.DataFrame(path_comp))
                else:
                    st.write("No common pathways found for direct comparison.")
            else:
                st.warning("Analysis type is not set to 'Pathway Analysis'. Switch analysis type in the query or sidebar to see pathway deltas.")
    else:
        if results_orig is not None:
            with st.expander("📊 Structured Numerical Insights (Original)", expanded=False):
                st.json(structured_orig)
            if structured_orig.get("ranked_mechanisms"):
                st.subheader("🏆 Ranked Failure Mechanisms (Original)")
                df_rank = pd.DataFrame(structured_orig["ranked_mechanisms"])
                st.dataframe(df_rank[["name", "composite_weight", "degree", "physics_match", "equation"]])

    # ------------------------------------------------------------------------
    # Visualisation – improved readability (no permanent labels, attention‑based sizing)
    # ------------------------------------------------------------------------
    def plot_graph(graph, title, size_attr="priority_score", use_attention_hover=False):
        if graph.number_of_nodes() == 0:
            return None
        cats_in_graph = list(set([graph.nodes[n].get('category','Unknown') for n in graph.nodes()]))
        color_pal = px.colors.qualitative.Set3 if len(cats_in_graph) <= 10 else px.colors.qualitative.Alphabet
        color_map = {c: color_pal[i % len(color_pal)] for i, c in enumerate(cats_in_graph)}
        node_colors = [color_map.get(graph.nodes[n].get('category','Unknown'), 'lightgray') for n in graph.nodes()]

        pos = nx.spring_layout(graph, k=1, iterations=100, seed=42, weight='weight')

        # Sizing based on attribute
        if size_attr == "attention" and all('attention' in graph.nodes[n] for n in graph.nodes()):
            sizes_vals = [graph.nodes[n].get('attention', 0) for n in graph.nodes()]
        else:
            sizes_vals = [graph.nodes[n].get('priority_score', 0) for n in graph.nodes()]

        min_s, max_s = 15, 60
        if max(sizes_vals) > min(sizes_vals):
            sizes = [min_s + (max_s-min_s)*(s-min(sizes_vals))/(max(sizes_vals)-min(sizes_vals)) for s in sizes_vals]
        else:
            sizes = [30]*len(sizes_vals)

        edge_traces = []
        edge_weights = [d.get('weight',1) for u,v,d in graph.edges(data=True)]
        if edge_weights:
            ew_min, ew_max = min(edge_weights), max(edge_weights)
        else:
            ew_min = ew_max = 1
        for u, v, d in graph.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            w = d.get('weight',1)
            if ew_max > ew_min:
                width = 0.5 + 4.5 * edge_width * (w - ew_min) / (ew_max - ew_min)
            else:
                width = 2.0 * edge_width
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color="#888"),
                    hoverinfo='text',
                    text=f"{u} — {v}<br>weight: {w}<br>type: {d.get('type','')}"
                )
            )

        node_x, node_y, hover_texts = [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            d = graph.nodes[node]
            base = f"{node}<br>Category: {d.get('category','N/A')}<br>Type: {d.get('type','N/A')}<br>Frequency: {d.get('frequency',0)}"
            if use_attention_hover and 'attention' in d:
                base += f"<br>Attention: {d['attention']:.3f}"
            else:
                base += f"<br>Priority: {d.get('priority_score',0):.3f}"
            if 'priority_std' in d:
                base += f"<br>Uncertainty: ±{d['priority_std']:.3f}"
            hover_texts.append(base)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=hover_texts,
            marker=dict(
                color=node_colors,
                size=sizes,
                line=dict(width=1, color='darkgray')
            )
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        for cat, col in color_map.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=col), name=cat, showlegend=True))

        fig.update_layout(
            title=title,
            title_font_size=20,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        return fig

    if G_filtered.number_of_nodes() > 0:
        if use_reconstruction and G_influenced.number_of_nodes() > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig_orig = plot_graph(G_filtered, "Original Filtered Graph", size_attr="priority_score", use_attention_hover=False)
                if fig_orig:
                    st.plotly_chart(fig_orig, use_container_width=True)
            with col2:
                fig_inf = plot_graph(G_influenced, "LLM‑Influenced Graph", size_attr="attention", use_attention_hover=True)
                if fig_inf:
                    st.plotly_chart(fig_inf, use_container_width=True)
        else:
            fig = plot_graph(G_filtered, f"Battery Degradation Graph - {analysis_type}", size_attr="priority_score", use_attention_hover=False)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Additional Analytics (from original)
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
        fig_edge = px.bar(x=edge_type_counts.index, y=edge_type_counts.values, title="Edge Type Distribution")
        st.plotly_chart(fig_edge, use_container_width=True)

    # Export functionality (enhanced)
    with st.sidebar:
        st.markdown("### 💾 Export")
        if st.button("Export Filtered Graph as CSV"):
            nodes_exp = pd.DataFrame([{'node': n, **G_filtered.nodes[n]} for n in G_filtered.nodes()])
            edges_exp = pd.DataFrame([{'source': u, 'target': v, **G_filtered.edges[u, v]} for u, v in G_filtered.edges()])
            st.download_button("Download Nodes", nodes_exp.to_csv(index=False), "nodes.csv")
            st.download_button("Download Edges", edges_exp.to_csv(index=False), "edges.csv")
        if st.button("Export as GraphML"):
            # GraphML requires networkx write_graphml
            import networkx as nx
            buf = BytesIO()
            nx.write_graphml(G_filtered, buf)
            st.download_button("Download GraphML", buf.getvalue(), "graph.graphml")
        if st.button("Export as JSON-LD"):
            # Simple JSON-LD conversion
            jsonld = {
                "@context": {"@vocab": "http://example.org/battery#"},
                "@graph": [
                    {"@id": n, "@type": "Node", "category": G_filtered.nodes[n].get('category'), "frequency": G_filtered.nodes[n].get('frequency')}
                    for n in G_filtered.nodes()
                ] + [
                    {"@id": f"{u}-{v}", "@type": "Edge", "source": u, "target": v, "weight": G_filtered.edges[u, v].get('weight')}
                    for u, v in G_filtered.edges()
                ]
            }
            st.download_button("Download JSON-LD", json.dumps(jsonld, indent=2), "graph.jsonld")

    # Display example queries
    with st.expander("📚 Example Queries", expanded=False):
        st.markdown("""
        - "Show pathways from electrode cracking to capacity fade involving diffusion-induced stress"
        - "Analyze communities related to chemo-mechanical degradation with physics boost 0.2"
        - "Find correlations between thermal runaway and mechanical degradation"
        - "How have SEI formation and lithium plating evolved over time?"
        - "Ego network around stress concentration with radius 2"
        - "High C-rate 2C, temperature 45°C, voltage 4.2V"
        - "Analyze centrality for fracture, fatigue, and damage"
        """)

if __name__ == "__main__":
    main()
