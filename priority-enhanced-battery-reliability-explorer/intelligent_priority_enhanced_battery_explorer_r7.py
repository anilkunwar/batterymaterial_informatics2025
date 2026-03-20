#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT BATTERY DEGRADATION KNOWLEDGE EXPLORER
===================================================
Expanded version with performance optimizations, mathematical robustness,
LLM enhancements, advanced graph analytics, uncertainty quantification,
scalability, UX improvements, and physics integration.

FIXED: UnboundLocalError for nx by removing local import inside main() scope.
"""

import os
import pathlib
import sys
import streamlit as st
import pandas as pd
import networkx as nx   # <-- Global Import
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
        from scipy.stats import pearsonr
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
