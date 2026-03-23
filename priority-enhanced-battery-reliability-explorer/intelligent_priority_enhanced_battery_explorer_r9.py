#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT BATTERY DEGRADATION KNOWLEDGE EXPLORER
===================================================
Optimized for Streamlit Cloud – CPU‑safe, memory‑efficient, and fast.
"""

import os
import pathlib
import sys
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
import redis
from scipy.stats import bootstrap

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Optional dependencies flags – defined at the very top to avoid NameError
# ============================================================================

TRANSFORMERS_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False
PYVIS_AVAILABLE = False
FAISS_AVAILABLE = False

try:
    from transformers import (
        AutoModel, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel,
        AutoModelForCausalLM, BitsAndBytesConfig
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    try:
        import bitsandbytes as bnb
        BITSANDBYTES_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    AutoModel = AutoTokenizer = GPT2Tokenizer = GPT2LMHeadModel = AutoModelForCausalLM = None
    torch = None

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Fast similarity search disabled.")

# ============================================================================
# Global configuration – data directory detection
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
        "equation": r"σ = \frac{E \Omega \Delta c}{1 - ν}",
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
        "equation": r"LAM \propto \int σ \, dN",
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
        "equation": r"j = j_0 \left[ \exp\left(\frac{\alpha n F η}{RT}\right) - \exp\left(-\frac{(1-\alpha) n F η}{RT}\right) \right]",
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
# DATA CLASS FOR PARSED PARAMETERS
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
    uncertainty: Dict[str, float] = field(default_factory=dict)

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
# SCIBERT LOADER & EMBEDDING UTILITIES
# ============================================================================
scibert_tokenizer = None
scibert_model = None
KEY_TERMS_EMBEDDINGS = []
PHYSICS_TERMS_EMBEDDINGS = []
EMBEDDING_INDEX = None  # FAISS index
EMBEDDING_INDEX_NODES = []

@st.cache_resource
def load_scibert():
    """Lazy-load SciBERT – called only when semantic analysis is enabled."""
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

@st.cache_data(ttl=None)
def compute_embeddings(texts):
    return get_scibert_embedding(texts)

def build_faiss_index(embeddings, nodes):
    if not FAISS_AVAILABLE or not embeddings:
        return None, None
    valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
    if not valid_idx:
        return None, None
    valid_embeddings = np.array([embeddings[i] for i in valid_idx], dtype=np.float32)
    valid_nodes = [nodes[i] for i in valid_idx]
    index = faiss.IndexFlatIP(valid_embeddings.shape[1])
    index.add(valid_embeddings)
    return index, valid_nodes

def fast_similarity_search(query_emb, index, nodes_list, k=10):
    if query_emb is None or index is None:
        return [], []
    if len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
    scores, indices = index.search(query_emb.astype(np.float32), k)
    return [nodes_list[i] for i in indices[0]], scores[0]

# ============================================================================
# UNCERTAINTY QUANTIFICATION FUNCTIONS (unchanged)
# ============================================================================
def bootstrap_centrality(G, n_samples=100, metric='degree'):
    if G.number_of_nodes() == 0:
        return {}, {}, {}
    nodes = list(G.nodes())
    edges = list(G.edges())
    n_edges = len(edges)
    centrality_samples = []
    for _ in range(n_samples):
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
    base_priority = calculate_priority_scores(G, nodes_df, **kwargs)
    samples = []
    for _ in range(n_samples):
        nodes_df_perturbed = nodes_df.copy()
        nodes_df_perturbed['frequency'] = nodes_df_perturbed['frequency'] * (1 + np.random.normal(0, 0.05))
        op_params = kwargs.get('operational_params', {})
        pert_op = {}
        for k, v in op_params.items():
            pert_op[k] = v * (1 + np.random.normal(0, 0.1))
        priority = calculate_priority_scores(G, nodes_df_perturbed,
                                            physics_boost_weight=kwargs.get('physics_boost_weight', 0.15),
                                            operational_params=pert_op,
                                            focus_terms=kwargs.get('focus_terms'))
        samples.append(priority)
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
    if G.number_of_nodes() == 0:
        return {}
    partitions = {}
    for r in resolutions:
        try:
            partitions[r] = community_louvain.best_partition(G, weight=weight, resolution=r)
        except:
            partitions[r] = {n: 0 for n in G.nodes()}
    baseline = partitions.get(1.0, {})
    stability = {}
    for r, part in partitions.items():
        if r == 1.0:
            stability[r] = 1.0
        else:
            labels1 = [baseline.get(n, 0) for n in G.nodes()]
            labels2 = [part.get(n, 0) for n in G.nodes()]
            try:
                nmi = normalized_mutual_info_score(labels1, labels2)
            except:
                nmi = 0
            stability[r] = nmi
    return {'partitions': partitions, 'stability': stability}

def comprehensive_centrality(G):
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
    try:
        return list(nx.shortest_simple_paths(G, source, target, weight=weight))[:k]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

def trend_detection(values):
    from scipy.stats import mstats
    n = len(values)
    if n < 3:
        return {'trend': 'insufficient data', 'p': 1.0}
    try:
        from scipy.stats import kendalltau
        tau, p = kendalltau(range(n), values)
        trend = 'increasing' if tau > 0 else 'decreasing' if tau < 0 else 'no trend'
        return {'trend': trend, 'p': p, 'tau': tau}
    except:
        from scipy.stats import pearsonr
        r, p = pearsonr(range(n), values)
        trend = 'increasing' if r > 0.1 else 'decreasing' if r < -0.1 else 'no trend'
        return {'trend': trend, 'p': p, 'correlation': r}

# ============================================================================
# PHYSICS INTEGRATION (symbolic verification)
# ============================================================================
def verify_physics_equation(equation_str, variables):
    try:
        expr = sp.sympify(equation_str.replace('\\', ''))
        free_symbols = expr.free_symbols
        missing = [str(s) for s in free_symbols if str(s) not in variables]
        if missing:
            return False, f"Missing variables: {missing}"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def dimensional_analysis(equation, units):
    return True, "Dimensional analysis not implemented."

# ============================================================================
# PERFORMANCE OPTIMIZATIONS: PARALLEL EMBEDDING
# ============================================================================
def parallel_embedding_compute(texts, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()
    chunk_size = max(1, len(texts) // (n_workers * 2))
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    embeddings = []
    for i, chunk in enumerate(chunks):
        emb = get_scibert_embedding(chunk)
        embeddings.extend(emb)
    return embeddings

# ============================================================================
# CACHING UTILITIES
# ============================================================================
def get_cache():
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return r
    except:
        logger.warning("Redis connection failed. Using disk cache.")
    return None

def disk_cache_get(key):
    cache_dir = os.path.join(DB_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def disk_cache_set(key, value):
    cache_dir = os.path.join(DB_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(value, f)

# ============================================================================
# LLM LOADER – MEMORY EFFICIENT (CPU & GPU) with extended model list
# ============================================================================
LLM_MODELS = {
    "GPT-2 (124M, fastest)": "gpt2",
    "GPT-2 Medium (355M, fast)": "gpt2-medium",
    "Phi-2 (2.7B, CPU-friendly)": "microsoft/phi-2",
    "Qwen2.5-0.5B-Instruct (tiny)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct (CPU)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-2B (2B, CPU)": "google/gemma-2b",
    "Qwen2.5-3B-Instruct (may OOM)": "Qwen/Qwen2.5-3B-Instruct",
    "tinyLLAMA (1.1B)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

def get_memory_info():
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except ImportError:
        return None

@st.cache_resource(
    show_spinner="Loading LLM (cached) – this may take 30–90 seconds the first time...",
    ttl=None,
    max_entries=1
)
def load_llm_memory_efficient(model_name: str):
    if not TRANSFORMERS_AVAILABLE:
        st.error("transformers not installed. LLM features disabled.")
        return None, None, model_name

    model_id = LLM_MODELS.get(model_name, model_name)
    mem_avail = get_memory_info()
    if mem_avail is not None and mem_avail < 1.0:
        st.warning(f"⚠️ Low memory: {mem_avail:.1f} GB available. Loading a large model may fail.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            device_map = "auto"
            if BITSANDBYTES_AVAILABLE:
                try:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map=device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    st.info(f"✅ Loaded {model_name} in 8‑bit (GPU, ~{model.num_parameters()/1e9:.1f}B parameters)")
                except Exception as e:
                    st.warning(f"8‑bit loading failed: {e}. Falling back to float16.")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                st.info(f"✅ Loaded {model_name} in float16 (GPU)")
        else:
            # CPU only: float32 for stability
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            st.info(f"🖥️ Loaded {model_name} on CPU (float32)")

        model.eval()
        return tokenizer, model, model_name

    except Exception as e:
        st.error(f"Failed to load {model_name}: {str(e)}")
        return None, None, model_name

def adaptive_temperature(query):
    return max(0.05, min(0.3, 0.2 * (1 - len(query.split()) / 50)))

def parse_with_validation(text, tokenizer, model) -> ParsedQuery:
    prompt = f"""Extract battery degradation parameters from the query as JSON. Output only JSON.
Query: "{text}"
JSON:"""
    try:
        if "Qwen" in st.session_state.get('llm_backend_loaded', '') or "Phi" in st.session_state.get('llm_backend_loaded', '') or "Gemma" in st.session_state.get('llm_backend_loaded', ''):
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
# RELEVANCE SCORER
# ============================================================================
class RelevanceScorer:
    def __init__(self, use_scibert=True):
        global scibert_tokenizer, scibert_model
        self.use_scibert = use_scibert and scibert_tokenizer is not None and scibert_model is not None
    def score_query_to_nodes(self, query: str, nodes_list: List[str]) -> Tuple[float, float]:
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
                conf = 0.8 if all(e is not None for e in n_emb) else 0.5
                return float(avg_sim), conf
            except:
                return 0.5, 0.0
        else:
            words = set(query.lower().split())
            matches = sum(1 for n in nodes_list[:100] if any(w in n.lower() for w in words))
            return min(1.0, matches / 50.0), 0.3

# ============================================================================
# QUERY-DRIVEN GRAPH RECONSTRUCTION
# ============================================================================
def llm_expand_vocabulary(query: str, tokenizer, model) -> List[str]:
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
    q_emb = get_scibert_embedding(user_query)
    if q_emb is None:
        return G_orig.copy()

    seed_terms = params.get('focus_terms', []) + params.get('source_terms', []) + params.get('target_terms', [])
    seed_terms = [t for t in seed_terms if t]
    llm_terms = llm_expand_vocabulary(user_query, tokenizer, model)
    all_seed_terms = list(set(seed_terms + llm_terms))

    node_list = [n for n in G_orig.nodes() if term_embeddings_dict.get(n) is not None]
    if not node_list:
        return G_orig.copy()
    node_embs = np.stack([term_embeddings_dict[n] for n in node_list])

    tau = 0.1
    logits = np.dot(node_embs, q_emb) / tau
    exp_logits = np.exp(logits - np.max(logits))
    attention_raw = exp_logits / np.sum(exp_logits)

    seed_embs = [term_embeddings_dict.get(s) for s in all_seed_terms if term_embeddings_dict.get(s) is not None]
    if seed_embs:
        seed_embs = np.stack(seed_embs)
        sim_to_seeds = np.max(np.dot(node_embs, seed_embs.T), axis=1)
    else:
        sim_to_seeds = np.ones(len(node_list))

    phys_boost = np.array([1.3 if any(p in n.lower() for p in PHYSICS_TERMS) else 1.0 for n in node_list])
    degrees = np.array([G_orig.degree(n) for n in node_list])
    max_deg = max(degrees) if max(degrees) > 0 else 1
    deg_boost = 1 + 0.5 * (degrees / max_deg)
    attention = attention_raw * sim_to_seeds * phys_boost * deg_boost

    if len(attention) > max_nodes:
        selected_idx = np.argsort(attention)[-max_nodes:]
        selected_nodes = [node_list[i] for i in selected_idx]
    else:
        selected_nodes = node_list

    node_attention = {node_list[i]: attention[i] for i in range(len(node_list))}
    G_inf = G_orig.subgraph(selected_nodes).copy()
    for n in G_inf.nodes():
        G_inf.nodes[n]['attention'] = node_attention.get(n, 0)
    for u, v, d in G_inf.edges(data=True):
        att_u = node_attention.get(u, 0.5)
        att_v = node_attention.get(v, 0.5)
        d['weight'] = d.get('weight', 1) * (att_u + att_v) / 2 * 1.5

    if tokenizer is not None and model is not None:
        suggested = llm_suggest_missing_edges(user_query, list(G_inf.nodes()), tokenizer, model)
        for src, tgt, w in suggested:
            if src in G_inf and tgt in G_inf and not G_inf.has_edge(src, tgt):
                G_inf.add_edge(src, tgt, weight=w, type="LLM-inferred")
    return G_inf

# ============================================================================
# PRIORITY SCORE CALCULATION – OPTIMIZED (uses pre‑computed embeddings)
# ============================================================================
def robust_normalize(values, epsilon=1e-8):
    median = np.median(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25) + epsilon
    return (values - median) / iqr

def safe_normalize(values, epsilon=1e-8):
    min_v, max_v = np.min(values), np.max(values)
    range_v = max_v - min_v + epsilon
    return (values - min_v) / range_v

def calculate_priority_scores(
    G,
    nodes_df,
    term_embeddings_dict,          # pre‑computed embeddings for all nodes
    physics_boost_weight=0.15,
    operational_params=None,
    focus_terms=None
):
    global KEY_TERMS_EMBEDDINGS, PHYSICS_TERMS_EMBEDDINGS

    # 1. Frequency normalization (vectorised)
    freq_vals = nodes_df['frequency'].values
    if len(freq_vals) > 0:
        median = np.median(freq_vals)
        iqr = np.percentile(freq_vals, 75) - np.percentile(freq_vals, 25) + 1e-8
        norm_freq = (freq_vals - median) / iqr
        norm_freq = np.clip(norm_freq, 0, 1)
    else:
        norm_freq = np.array([])

    # 2. Centrality – use approximate for large graphs
    if G.number_of_nodes() > 500:
        degree_centrality = nx.degree_centrality(G)
        # Approximate betweenness (k=50 samples)
        betweenness_centrality = nx.approximate_betweenness_centrality(G, k=50)
    else:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)  # limit iterations
    except:
        eigenvector_centrality = {node: 0 for node in G.nodes()}

    # 3. Semantic & physics scores (use pre‑computed embeddings)
    semantic_scores = {}
    physics_scores = {}

    # Pre‑compute focus embeddings once
    focus_embs = []
    if focus_terms:
        focus_embs = [e for e in get_scibert_embedding(focus_terms) if e is not None]

    for node in G.nodes():
        emb = term_embeddings_dict.get(node)
        if emb is None:
            semantic_scores[node] = 0
            physics_scores[node] = 0
            continue

        # Semantic similarity
        if focus_embs:
            sem_sims = [cosine_similarity([emb], [fe])[0][0] for fe in focus_embs]
            semantic_scores[node] = max(sem_sims, default=0)
        else:
            # Use global key terms (cached)
            if KEY_TERMS_EMBEDDINGS:
                sem_sims = [cosine_similarity([emb], [kt_emb])[0][0] for kt_emb in KEY_TERMS_EMBEDDINGS if kt_emb is not None]
                semantic_scores[node] = max(sem_sims, default=0)
            else:
                semantic_scores[node] = 0

        # Physics similarity
        if PHYSICS_TERMS_EMBEDDINGS:
            phys_sims = [cosine_similarity([emb], [pt_emb])[0][0] for pt_emb in PHYSICS_TERMS_EMBEDDINGS if pt_emb is not None]
            physics_scores[node] = min(max(phys_sims, default=0), 1.0)
        else:
            physics_scores[node] = 0

        # Operational boost
        if operational_params:
            score = physics_scores[node]
            if operational_params.get('temperature', 25) > 45 and any(t in node.lower() for t in ['thermal', 'temp']):
                score *= 1.3
            if operational_params.get('c_rate', 1.0) > 2.0 and any(t in node.lower() for t in ['rate', 'power']):
                score *= 1.3
            if operational_params.get('voltage', 3.7) > 4.2 and any(t in node.lower() for t in ['voltage', 'potential']):
                score *= 1.3
            physics_scores[node] = min(score, 1.0)

    # 4. Weighted sum
    w_f, w_d, w_b, w_s, w_p = 0.35, 0.25, 0.20, 0.10, physics_boost_weight
    weights = np.exp([w_f, w_d, w_b, w_s, w_p])
    weights = weights / np.sum(weights)
    w_f, w_d, w_b, w_s, w_p = weights

    # Create mapping from node to normalized frequency index
    freq_map = {node: freq for node, freq in zip(nodes_df['node'], norm_freq)} if len(norm_freq) > 0 else {}

    priority_scores = {}
    for node in G.nodes():
        f_val = freq_map.get(node, 0)
        priority_scores[node] = (
            w_f * f_val +
            w_d * degree_centrality.get(node, 0) +
            w_b * betweenness_centrality.get(node, 0) +
            w_s * semantic_scores.get(node, 0) +
            w_p * physics_scores.get(node, 0)
        )
    return priority_scores

# ============================================================================
# FILTER GRAPH
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
    if len(Gf) > 0 and not nx.is_connected(Gf):
        largest_cc = max(nx.connected_components(Gf), key=len)
        Gf = Gf.subgraph(largest_cc).copy()
    return Gf

# ============================================================================
# STRUCTURED INSIGHT GENERATOR (unchanged)
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
# DATA LOADING
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
# BATTERY NLP PARSER
# ============================================================================
class BatteryNLParser:
    def __init__(self):
        self.defaults = ParsedParameters()
        self.patterns = {
            'min_weight': [r'min(?:imum)?\s*edge\s*weight\s*(?:of|>=|>|=)?\s*(\d+)'],
            'min_freq': [r'min(?:imum)?\s*(?:node)?\s*frequency\s*(?:of|>=|>|=)?\s*(\d+)'],
            'min_priority_score': [r'priority\s*score\s*(?:of|>=|>|=)?\s*(\d*\.?\d+)'],
            'physics_boost_weight': [r'physics\s*boost\s*(?:of|>=|>|=)?\s*(\d*\.?\d+)'],
            'c_rate': [r'c[-\s]?rate\s*(?:of|>=|>|=)?\s*(\d+\.?\d*)'],
            'voltage': [r'voltage\s*(?:of|>=|>|=)?\s*(\d+\.?\d*)'],
            'temperature': [r'temperature\s*(?:of|>=|>|=)?\s*(\d+\.?\d*)'],
            'require_physics_in_pathways': [r'require\s*physics', r'only\s*physics\s*paths?']
        }
        self.analysis_map = {
            'centrality': 'Centrality Analysis',
            'community': 'Community Detection',
            'ego': 'Ego Network Analysis',
            'pathway': 'Pathway Analysis',
            'temporal': 'Temporal Analysis',
            'correlation': 'Correlation Analysis'
        }

    def parse_regex(self, text: str) -> Dict:
        if not text:
            return asdict(self.defaults)
        params = asdict(self.defaults).copy()
        text_lower = text.lower()
        for key, pats in self.patterns.items():
            for pat in pats:
                match = re.search(pat, text_lower)
                if match:
                    try:
                        if key == 'require_physics_in_pathways':
                            params[key] = True
                        else:
                            params[key] = float(match.group(1))
                        break
                    except:
                        continue
        unit_vals = UnitParser.parse_units(text)
        if 'c_rate' in unit_vals:
            params['c_rate'] = unit_vals['c_rate']
        if 'voltage' in unit_vals:
            params['voltage'] = unit_vals['voltage']
        if 'temperature' in unit_vals:
            params['temperature'] = unit_vals['temperature']
        for key, val in self.analysis_map.items():
            if key in text_lower:
                params['analysis_type'] = val
                break
        src = re.search(r'from\s+([a-zA-Z\s\-]+?)\s+to', text_lower)
        tgt = re.search(r'to\s+([a-zA-Z\s\-]+?)(?:\s+and|\s*,|$|\.)', text_lower)
        if src:
            params['source_terms'] = [t.strip() for t in src.group(1).split(',')]
        if tgt:
            params['target_terms'] = [t.strip() for t in tgt.group(1).split(',')]
        return params

    def _extract_json_robust(self, generated: str) -> Optional[Dict]:
        match = re.search(r'\{.*\}', generated, re.DOTALL)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except:
                return None
        return None

    def parse_with_llm(self, text, tokenizer, model, regex_params=None, temperature=0.1) -> Dict:
        if not text or tokenizer is None or model is None:
            return regex_params if regex_params else asdict(self.defaults)
        system = "You are an expert in battery degradation. Output only a JSON dictionary with keys from: " + str(list(asdict(self.defaults).keys()))
        examples = """
Examples:
1. "Show pathways from electrode cracking to capacity fade" -> {"analysis_type": "Pathway Analysis", "source_terms": ["electrode cracking"], "target_terms": ["capacity fade"]}
2. "Analyze communities related to chemo-mechanical degradation, boost physics to 0.2" -> {"analysis_type": "Community Detection", "focus_terms": ["chemo-mechanical degradation"], "physics_boost_weight": 0.2}
3. "Show ego network around stress concentration" -> {"analysis_type": "Ego Network Analysis", "central_nodes": ["stress concentration"]}
4. "Find correlations between thermal runaway and mechanical degradation" -> {"analysis_type": "Correlation Analysis", "focus_terms": ["thermal runaway", "mechanical degradation"]}
5. "How have SEI formation and lithium plating evolved?" -> {"analysis_type": "Temporal Analysis", "focus_terms": ["SEI formation", "lithium plating"]}
6. "High C-rate 2C, temperature 45°C" -> {"c_rate": 2.0, "temperature": 45.0}
"""
        user = f"{examples}\nText: \"{text}\"\nPreliminary regex: {json.dumps(regex_params, default=str) if regex_params else 'None'}\nJSON:"
        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default)')
        try:
            if "Qwen" in backend or "Phi" in backend or "Gemma" in backend:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{system}\n{user}\n"
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=400, temperature=temperature, do_sample=temperature>0, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            llm_params = self._extract_json_robust(generated)
            if llm_params:
                for k in asdict(self.defaults).keys():
                    if k not in llm_params:
                        llm_params[k] = asdict(self.defaults)[k]
                for k in ['c_rate','voltage','temperature','soc','dod']:
                    if k in llm_params:
                        llm_params[k] = np.clip(llm_params[k], OPERATIONAL_CONSTRAINTS[k]['min'], OPERATIONAL_CONSTRAINTS[k]['max'])
                return llm_params
        except Exception as e:
            logger.warning(f"LLM parsing error: {e}")
        return regex_params if regex_params else asdict(self.defaults)

    def hybrid_parse(self, text, tokenizer=None, model=None, use_ensemble=False, ensemble_runs=3) -> Dict:
        regex_params = self.parse_regex(text)
        if tokenizer is None or model is None:
            return regex_params
        if use_ensemble:
            all_llm = [self.parse_with_llm(text, tokenizer, model, regex_params, temperature=0.2) for _ in range(ensemble_runs)]
            llm_params = {}
            for k in asdict(self.defaults).keys():
                vals = [p[k] for p in all_llm]
                if isinstance(vals[0], (int, float)):
                    llm_params[k] = float(np.mean(vals))
                elif isinstance(vals[0], bool):
                    llm_params[k] = max(set(vals), key=vals.count)
                elif isinstance(vals[0], list):
                    flat = [item for sublist in vals for item in (sublist if isinstance(sublist, list) else [sublist])]
                    llm_params[k] = list(set(flat))[:10]
                else:
                    llm_params[k] = vals[0]
        else:
            llm_params = self.parse_with_llm(text, tokenizer, model, regex_params)
        merged = {}
        for k in asdict(self.defaults).keys():
            if k in regex_params and regex_params[k] != getattr(self.defaults, k):
                merged[k] = regex_params[k]
            else:
                merged[k] = llm_params.get(k, getattr(self.defaults, k))
        merged['confidence_score'] = 0.8
        merged['parsing_method'] = 'hybrid'
        return merged

# ============================================================================
# UNIT PARSER
# ============================================================================
class UnitParser:
    @staticmethod
    def parse_units(text: str) -> Dict[str, float]:
        if not text:
            return {}
        text_lower = text.lower()
        extracted = {}
        patterns = {
            'c_rate': [r'(\d+\.?\d*)\s*(?:C-rate|C\s*rate|C)'],
            'voltage': [r'(\d+\.?\d*)\s*(?:V|volt|volts)'],
            'temperature': [r'(\d+\.?\d*)\s*(?:°C|C|celsius)']
        }
        for key, pats in patterns.items():
            for pat in pats:
                match = re.search(pat, text_lower)
                if match:
                    try:
                        extracted[key] = float(match.group(1))
                    except:
                        pass
        return extracted

# ============================================================================
# FAILURE ANALYSIS FUNCTIONS
# ============================================================================
def analyze_failure_centrality(G_filtered, focus_terms=None):
    if focus_terms is None:
        focus_terms = ["crack", "fracture", "degradation", "fatigue", "damage", "failure", "mechanical", "cycling", "capacity fade", "SEI"]
    degree = nx.degree_centrality(G_filtered)
    between = nx.betweenness_centrality(G_filtered)
    closeness = nx.closeness_centrality(G_filtered)
    try:
        eigen = nx.eigenvector_centrality(G_filtered, max_iter=1000)
    except:
        eigen = {n: 0 for n in G_filtered.nodes()}
    results = []
    for node in G_filtered.nodes():
        if any(term in node.lower() for term in focus_terms):
            results.append({
                'node': node,
                'degree': degree.get(node, 0),
                'betweenness': between.get(node, 0),
                'closeness': closeness.get(node, 0),
                'eigenvector': eigen.get(node, 0),
                'category': G_filtered.nodes[node].get('category', ''),
                'type': G_filtered.nodes[node].get('type', '')
            })
    return pd.DataFrame(results)

def detect_failure_communities(G_filtered):
    try:
        partition = community_louvain.best_partition(G_filtered, weight='weight', resolution=1.2)
    except:
        partition = {node: 0 for node in G_filtered.nodes()}
    communities = {}
    for node, cid in partition.items():
        if cid not in communities:
            communities[cid] = {'nodes': [], 'categories': Counter(), 'failure_keywords': Counter(), 'physics_terms': Counter()}
        communities[cid]['nodes'].append(node)
        cat = G_filtered.nodes[node].get('category', '')
        if cat:
            communities[cid]['categories'][cat] += 1
        for kw in ['crack', 'fracture', 'degrad', 'fatigue', 'damage', 'failure']:
            if kw in node.lower():
                communities[cid]['failure_keywords'][kw] += 1
        for term in PHYSICS_TERMS:
            if term in node.lower():
                communities[cid]['physics_terms'][term] += 1
    return communities, partition

def analyze_ego_networks(G_filtered, central_nodes=None):
    if central_nodes is None:
        central_nodes = ["electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress", "capacity fade", "lithium plating"]
    results = {}
    for node in central_nodes:
        if node in G_filtered.nodes():
            try:
                ego = nx.ego_graph(G_filtered, node, radius=2)
                results[node] = {
                    'node_count': ego.number_of_nodes(),
                    'edge_count': ego.number_of_edges(),
                    'density': nx.density(ego),
                    'average_degree': sum(dict(ego.degree()).values()) / ego.number_of_nodes() if ego.number_of_nodes() > 0 else 0,
                    'centrality': nx.degree_centrality(ego).get(node, 0),
                    'neighbors': list(ego.neighbors(node)),
                    'subgraph_categories': Counter([ego.nodes[n].get('category', '') for n in ego.nodes()]),
                    'physics_terms': [n for n in ego.nodes() if any(t in n.lower() for t in PHYSICS_TERMS)]
                }
            except:
                results[node] = {'node_count': 0, 'edge_count': 0, 'density': 0, 'average_degree': 0, 'centrality': 0, 'neighbors': [], 'subgraph_categories': Counter(), 'physics_terms': []}
    return results

def find_failure_pathways(G_filtered, source_terms, target_terms, require_physics=False, min_physics_similarity=0.5):
    pathways = {}
    for src in source_terms:
        for tgt in target_terms:
            if src in G_filtered.nodes() and tgt in G_filtered.nodes():
                try:
                    paths = list(nx.all_shortest_paths(G_filtered, source=src, target=tgt, weight='weight'))
                    if require_physics:
                        filtered = []
                        for p in paths:
                            has_phys = any(any(t in node.lower() for t in PHYSICS_TERMS) for node in p)
                            if has_phys:
                                filtered.append(p)
                        paths = filtered
                    if paths:
                        path = paths[0]
                        physics_scores = []
                        for node in path:
                            emb = get_scibert_embedding(node)
                            if emb is not None and PHYSICS_TERMS_EMBEDDINGS:
                                sims = [cosine_similarity([emb], [pt_emb])[0][0] for pt_emb in PHYSICS_TERMS_EMBEDDINGS if pt_emb is not None]
                                physics_scores.append(max(sims, default=0))
                        avg_phys = np.mean(physics_scores) if physics_scores else 0
                        pathways[f"{src} -> {tgt}"] = {
                            'path': path,
                            'length': len(path)-1,
                            'nodes': path,
                            'num_paths': len(paths),
                            'contains_physics': any(any(t in node.lower() for t in PHYSICS_TERMS) for node in path),
                            'avg_physics_similarity': avg_phys
                        }
                    else:
                        pathways[f"{src} -> {tgt}"] = {'path': None, 'length': float('inf'), 'nodes': [], 'contains_physics': False, 'avg_physics_similarity': 0}
                except nx.NetworkXNoPath:
                    pathways[f"{src} -> {tgt}"] = {'path': None, 'length': float('inf'), 'nodes': [], 'contains_physics': False, 'avg_physics_similarity': 0}
    return pathways

def analyze_temporal_patterns(nodes_df, edges_df, time_column='year'):
    if time_column not in nodes_df.columns:
        return {"error": "Time column not found"}
    periods = sorted(nodes_df[time_column].dropna().unique())
    analysis = {}
    for p in periods:
        period_nodes = nodes_df[nodes_df[time_column] == p]
        analysis[p] = {
            'total_concepts': len(period_nodes),
            'failure_concepts': len([n for n in period_nodes['node'] if any(kw in n.lower() for kw in ['crack','fracture','degrad','fatigue','damage'])]),
            'physics_concepts': len([n for n in period_nodes['node'] if any(term in n.lower() for term in PHYSICS_TERMS)]),
            'top_concepts': period_nodes.nlargest(5, 'frequency')['node'].tolist()
        }
    return analysis

def analyze_failure_correlations(G_filtered):
    failure_terms = [n for n in G_filtered.nodes() if any(kw in n.lower() for kw in ['crack','fracture','degrad','fatigue','damage','failure'])]
    n = len(failure_terms)
    corr = np.zeros((n, n))
    for i, t1 in enumerate(failure_terms):
        for j, t2 in enumerate(failure_terms):
            if G_filtered.has_edge(t1, t2):
                corr[i, j] = G_filtered.edges[t1, t2].get('weight', 0)
    return corr, failure_terms

def benchmark_graphs(G1: nx.Graph, G2: nx.Graph, analysis_type: str) -> Dict:
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    inter_nodes = nodes1 & nodes2

    node_jaccard = len(inter_nodes) / len(nodes1 | nodes2) if (nodes1 | nodes2) else 0
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    edge_jaccard = len(edges1 & edges2) / len(edges1 | edges2) if (edges1 | edges2) else 0
    node_delta_pct = (len(nodes2) - len(nodes1)) / len(nodes1) * 100 if len(nodes1) > 0 else 0

    deg1 = [G1.degree(n) for n in inter_nodes]
    deg2 = [G2.degree(n) for n in inter_nodes]
    avg_deg1 = np.mean(deg1) if deg1 else 0
    avg_deg2 = np.mean(deg2) if deg2 else 0
    avg_deg_delta = avg_deg2 - avg_deg1

    if len(inter_nodes) > 1:
        cent1 = [nx.degree_centrality(G1)[n] for n in inter_nodes]
        cent2 = [nx.degree_centrality(G2)[n] for n in inter_nodes]
        centrality_corr, _ = spearmanr(cent1, cent2)
    else:
        centrality_corr = 1.0 if len(inter_nodes) == 1 else 0.0

    nmi = 0.0
    if len(inter_nodes) > 0:
        try:
            part1 = community_louvain.best_partition(G1.subgraph(inter_nodes))
            part2 = community_louvain.best_partition(G2.subgraph(inter_nodes))
            labels1 = [part1[n] for n in inter_nodes]
            labels2 = [part2[n] for n in inter_nodes]
            nmi = normalized_mutual_info_score(labels1, labels2)
        except:
            nmi = 0.0

    metrics = {
        "node_jaccard": round(node_jaccard, 3),
        "edge_jaccard": round(edge_jaccard, 3),
        "node_count_delta_pct": round(node_delta_pct, 1),
        "avg_degree_delta": round(avg_deg_delta, 3),
        "centrality_correlation": round(centrality_corr, 3),
        "community_nmi": round(nmi, 3),
    }
    return metrics

# ============================================================================
# PYVIS VISUALIZATION FUNCTION
# ============================================================================
def create_pyvis_network(G, title="Battery Degradation Graph", size_attr="priority_score", use_attention_hover=False, height="600px"):
    if not PYVIS_AVAILABLE:
        return None

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black")
    net.set_options("""
    var options = {
      nodes: {
        shape: 'dot',
        scaling: {
          min: 10,
          max: 50,
          label: {
            enabled: true,
            min: 12,
            max: 24
          }
        },
        font: {
          size: 12,
          face: 'Arial'
        }
      },
      edges: {
        smooth: false,
        arrows: {
          to: { enabled: false }
        },
        scaling: {
          min: 1,
          max: 10,
          label: {
            enabled: true,
            min: 12,
            max: 20
          }
        }
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -200,
          centralGravity: 0.01,
          springLength: 150,
          springConstant: 0.08,
          damping: 0.4,
          avoidOverlap: 0.5
        },
        stabilization: {
          iterations: 200
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true
      }
    }
    """)

    categories = set(G.nodes[n].get('category', 'Unknown') for n in G.nodes())
    color_palette = px.colors.qualitative.Set3
    color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}

    if size_attr == "attention" and all('attention' in G.nodes[n] for n in G.nodes()):
        sizes_vals = [G.nodes[n].get('attention', 0) for n in G.nodes()]
    else:
        sizes_vals = [G.nodes[n].get('priority_score', 0) for n in G.nodes()]

    min_s, max_s = 10, 50
    if max(sizes_vals) > min(sizes_vals):
        sizes = [min_s + (max_s-min_s)*(s-min(sizes_vals))/(max(sizes_vals)-min(sizes_vals)) for s in sizes_vals]
    else:
        sizes = [30] * len(sizes_vals)

    for i, node in enumerate(G.nodes()):
        node_data = G.nodes[node]
        color = color_map.get(node_data.get('category', 'Unknown'), 'lightgray')
        tooltip = f"{node}<br>Category: {node_data.get('category','N/A')}<br>Type: {node_data.get('type','N/A')}<br>Frequency: {node_data.get('frequency',0)}"
        if use_attention_hover and 'attention' in node_data:
            tooltip += f"<br>Attention: {node_data['attention']:.3f}"
        else:
            tooltip += f"<br>Priority: {node_data.get('priority_score',0):.3f}"
        if 'priority_std' in node_data:
            tooltip += f"<br>Uncertainty: ±{node_data['priority_std']:.3f}"
        net.add_node(node, label=node, title=tooltip, color=color, size=sizes[i])

    edge_weights = [d.get('weight',1) for u,v,d in G.edges(data=True)]
    if edge_weights:
        ew_min, ew_max = min(edge_weights), max(edge_weights)
    else:
        ew_min = ew_max = 1

    for u, v, d in G.edges(data=True):
        w = d.get('weight',1)
        if ew_max > ew_min:
            width = 1 + 9 * (w - ew_min) / (ew_max - ew_min)
        else:
            width = 3
        title = f"{u} — {v}<br>weight: {w}<br>type: {d.get('type','')}"
        net.add_edge(u, v, value=width, title=title)

    return net.generate_html()

# ============================================================================
# MAIN APPLICATION – sidebar_filters is a regular function (no fragment)
# ============================================================================
def main():
    try:
        _ = nx
    except NameError:
        st.error("NetworkX module not available. Please install networkx.")
        st.stop()

    st.set_page_config(layout="wide", page_title="Intelligent Battery Degradation Explorer")
    st.markdown("<h1 style='text-align:center;'>🔋 Intelligent Battery Degradation Knowledge Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Optimized for Streamlit Cloud – CPU‑safe, memory‑efficient, and fast.</p>", unsafe_allow_html=True)

    # Memory warning banner
    st.warning("""
⚠️ **Memory‑efficient mode active**  
• On **GPU**: models loaded in 8‑bit (if available) → ~4× less VRAM  
• On **CPU**: models use float32 → stable but slower  
• First load may take up to 2 minutes (model is cached afterwards)  
• Only models ≤ 2 B parameters are offered to avoid OOM.
""")

    # ------------------------------------------------------------------------
    # Lazy load SciBERT – only when needed
    # ------------------------------------------------------------------------
    global scibert_tokenizer, scibert_model, KEY_TERMS_EMBEDDINGS, PHYSICS_TERMS_EMBEDDINGS
    if "scibert_loaded" not in st.session_state:
        st.session_state.scibert_loaded = False

    if not st.session_state.scibert_loaded:
        if st.checkbox("Enable Semantic Analysis (loads SciBERT)", value=True):
            with st.spinner("Loading SciBERT (once per session)..."):
                scibert_tokenizer, scibert_model = load_scibert()
                st.session_state.scibert_loaded = True
                # Pre‑compute global embeddings only if SciBERT loaded
                KEY_TERMS_EMBEDDINGS = compute_embeddings(KEY_TERMS)
                PHYSICS_TERMS_EMBEDDINGS = compute_embeddings(PHYSICS_TERMS)
                KEY_TERMS_EMBEDDINGS = [emb for emb in KEY_TERMS_EMBEDDINGS if emb is not None]
                PHYSICS_TERMS_EMBEDDINGS = [emb for emb in PHYSICS_TERMS_EMBEDDINGS if emb is not None]
        else:
            scibert_tokenizer, scibert_model = None, None
            KEY_TERMS_EMBEDDINGS = []
            PHYSICS_TERMS_EMBEDDINGS = []

    # ------------------------------------------------------------------------
    # Load data (cached)
    # ------------------------------------------------------------------------
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

    # Pre‑compute node embeddings once (if SciBERT is loaded)
    if st.session_state.scibert_loaded:
        @st.cache_data
        def compute_node_embeddings_dict(nodes_list):
            emb_list = get_scibert_embedding(nodes_list)
            return dict(zip(nodes_list, emb_list))
        node_list_all = list(G.nodes())
        term_embeddings_dict = compute_node_embeddings_dict(node_list_all)

        # Build FAISS index if available
        if FAISS_AVAILABLE:
            try:
                embeddings_list = [term_embeddings_dict[n] for n in node_list_all if term_embeddings_dict.get(n) is not None]
                if embeddings_list:
                    EMBEDDING_INDEX, EMBEDDING_INDEX_NODES = build_faiss_index(embeddings_list, node_list_all)
                else:
                    EMBEDDING_INDEX = None
                    EMBEDDING_INDEX_NODES = []
            except Exception as e:
                logger.error(f"FAISS index build failed: {e}")
                EMBEDDING_INDEX = None
                EMBEDDING_INDEX_NODES = []
    else:
        term_embeddings_dict = {}

    # ------------------------------------------------------------------------
    # Session state initialisation
    # ------------------------------------------------------------------------
    if "parser" not in st.session_state:
        st.session_state.parser = BatteryNLParser()
    if "relevance_scorer" not in st.session_state:
        st.session_state.relevance_scorer = RelevanceScorer(use_scibert=st.session_state.scibert_loaded)
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
    if "graph_cache" not in st.session_state:
        st.session_state.graph_cache = {}   # simple cache for filtered graphs

    # ------------------------------------------------------------------------
    # Sidebar filters – regular function (no fragment)
    # ------------------------------------------------------------------------
    def sidebar_filters():
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

            st.markdown("### 🎨 Visualization")
            viz_engine = st.selectbox("Visualization Engine", ["Plotly (static)", "Pyvis (interactive)"], index=0, key="viz_engine")

        return {
            "min_weight": min_weight,
            "min_freq": min_freq,
            "selected_cats": selected_cats,
            "selected_types": selected_types,
            "min_priority": min_priority,
            "selected_nodes": selected_nodes,
            "excluded_terms": excluded_terms,
            "physics_boost": physics_boost,
            "require_physics": require_physics,
            "min_phys_sim": min_phys_sim,
            "c_rate": c_rate,
            "voltage": voltage,
            "temperature": temperature,
            "soc": soc,
            "dod": dod,
            "highlight": highlight,
            "threshold": threshold,
            "suppress": suppress,
            "show_labels": show_labels,
            "label_size": label_size,
            "max_chars": max_chars,
            "edge_width": edge_width,
            "use_reconstruction": use_reconstruction,
            "max_recon_nodes": max_recon_nodes,
            "use_multi_res": use_multi_res,
            "use_bootstrap": use_bootstrap,
            "bootstrap_samples": bootstrap_samples,
            "viz_engine": viz_engine,
        }

    # Call the function to get current sidebar values
    sidebar_vals = sidebar_filters()

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
            st.markdown("### 🧠 LLM Settings (Memory Efficient)")
            model_choice = st.selectbox(
                "Choose local model",
                options=list(LLM_MODELS.keys()),
                index=0,
                key="llm_choice"
            )
            st.caption("**tinyLLAMA (1.1B)** added; Qwen2.5-1.5B still recommended.")
            use_llm = st.checkbox("Use LLM parsing & expansion", value=True)
            use_ensemble = st.checkbox("Use ensemble (slower but more stable)", value=False)
        with col3:
            st.markdown("### 🚀 Actions")
            run_button = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
            if st.button("🧹 Clear History", use_container_width=True):
                st.session_state.last_params = None
                st.session_state.last_analysis_results = None
                st.session_state.last_structured_insights = None
                st.rerun()

    # ------------------------------------------------------------------------
    # Handle query analysis (only when button clicked)
    # ------------------------------------------------------------------------
    if run_button and user_query:
        with st.spinner("🔍 Loading LLM (cached) and parsing query..."):
            if use_llm and TRANSFORMERS_AVAILABLE:
                tok, mod, loaded_name = load_llm_memory_efficient(model_choice)
                st.session_state.llm_tokenizer = tok
                st.session_state.llm_model = mod
                st.session_state.llm_backend_loaded = loaded_name
            else:
                st.session_state.llm_tokenizer = None
                st.session_state.llm_model = None

            parser = st.session_state.parser
            if use_llm and st.session_state.llm_tokenizer is not None:
                params = parser.hybrid_parse(user_query, st.session_state.llm_tokenizer, st.session_state.llm_model,
                                             use_ensemble=use_ensemble, ensemble_runs=3 if use_ensemble else 1)
            else:
                params = parser.parse_regex(user_query)
            try:
                validated = ParsedQuery(**params)
                params = validated.dict()
            except ValidationError as e:
                st.warning(f"LLM output validation failed: {e}. Using regex parsed.")
            st.session_state.last_params = ParsedParameters(**params)

            # Semantic relevance only if SciBERT loaded
            if st.session_state.scibert_loaded:
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
    params['min_weight'] = sidebar_vals['min_weight']
    params['min_freq'] = sidebar_vals['min_freq']
    params['selected_categories'] = sidebar_vals['selected_cats']
    params['selected_types'] = sidebar_vals['selected_types']
    params['min_priority_score'] = sidebar_vals['min_priority']
    params['selected_nodes'] = sidebar_vals['selected_nodes']
    params['excluded_terms'] = sidebar_vals['excluded_terms']
    params['physics_boost_weight'] = sidebar_vals['physics_boost']
    params['require_physics_in_pathways'] = sidebar_vals['require_physics']
    params['min_physics_similarity'] = sidebar_vals['min_phys_sim']
    params['c_rate'] = sidebar_vals['c_rate']
    params['voltage'] = sidebar_vals['voltage']
    params['temperature'] = sidebar_vals['temperature']
    params['soc'] = sidebar_vals['soc']
    params['dod'] = sidebar_vals['dod']
    params['highlight_priority'] = sidebar_vals['highlight']
    params['priority_threshold'] = sidebar_vals['threshold']
    params['suppress_low_priority'] = sidebar_vals['suppress']
    params['show_labels'] = sidebar_vals['show_labels']
    params['label_font_size'] = sidebar_vals['label_size']
    params['label_max_chars'] = sidebar_vals['max_chars']
    params['edge_width_factor'] = sidebar_vals['edge_width']

    # ------------------------------------------------------------------------
    # Compute priority scores (cached if possible)
    # ------------------------------------------------------------------------
    # Create a robust cache key using MD5 hash of parameters and nodes
    # Convert params to a JSON string (handles lists, dicts, etc.)
    params_json = json.dumps(params, sort_keys=True, default=str)
    params_hash = hashlib.md5(params_json.encode()).hexdigest()

    # Hash the sorted list of nodes to capture graph structure changes
    nodes_hash = hashlib.md5("".join(sorted(G.nodes())).encode()).hexdigest()

    cache_key = f"priority_{params_hash}_{nodes_hash}"

    if cache_key in st.session_state.graph_cache:
        priority_scores = st.session_state.graph_cache[cache_key]['priority_scores']
    else:
        operational = {"c_rate": params['c_rate'], "voltage": params['voltage'], "temperature": params['temperature']}
        if st.session_state.scibert_loaded and term_embeddings_dict:
            priority_scores = calculate_priority_scores(
                G, nodes_df, term_embeddings_dict,
                physics_boost_weight=params['physics_boost_weight'],
                operational_params=operational,
                focus_terms=params.get('focus_terms')
            )
        else:
            # Fallback without embeddings – use only structural metrics
            degree = nx.degree_centrality(G)
            between = nx.betweenness_centrality(G)
            priority_scores = {}
            for node in G.nodes():
                priority_scores[node] = 0.5 * degree.get(node,0) + 0.5 * between.get(node,0)
        for n in G.nodes():
            G.nodes[n]['priority_score'] = priority_scores.get(n, 0)
        nodes_df['priority_score'] = nodes_df['node'].apply(lambda x: priority_scores.get(x, 0))
        st.session_state.graph_cache[cache_key] = {'priority_scores': priority_scores}

    # ------------------------------------------------------------------------
    # Bootstrap if requested
    # ------------------------------------------------------------------------
    if sidebar_vals['use_bootstrap']:
        with st.spinner("Computing Monte Carlo uncertainty for priority scores..."):
            mean_priority, std_priority = monte_carlo_priority_score(
                G, nodes_df, n_samples=sidebar_vals['bootstrap_samples'],
                physics_boost_weight=params['physics_boost_weight'],
                operational_params=operational,
                focus_terms=params.get('focus_terms')
            )
            for n in G.nodes():
                G.nodes[n]['priority_mean'] = mean_priority.get(n, priority_scores.get(n, 0))
                G.nodes[n]['priority_std'] = std_priority.get(n, 0)
            params['uncertainty'] = {'priority_scores': {n: round(std_priority.get(n, 0), 4) for n in G.nodes() if std_priority.get(n, 0) > 0}}

    # ------------------------------------------------------------------------
    # Filter graph (cached)
    # ------------------------------------------------------------------------
    filter_key = f"filter_{params['min_weight']}_{params['min_freq']}_{params['selected_categories']}_{params['selected_types']}_{params['selected_nodes']}_{params['excluded_terms']}_{params['min_priority_score']}_{params['suppress_low_priority']}"
    if filter_key in st.session_state.graph_cache:
        G_filtered = st.session_state.graph_cache[filter_key]['graph']
    else:
        G_filtered = filter_graph(
            G,
            min_weight=params['min_weight'],
            min_freq=params['min_freq'],
            selected_categories=params['selected_categories'],
            selected_types=params['selected_types'],
            selected_nodes=params['selected_nodes'],
            excluded_terms=params['excluded_terms'],
            min_priority_score=params['min_priority_score'],
            suppress_low_priority=params['suppress_low_priority']
        )
        st.session_state.graph_cache[filter_key] = {'graph': G_filtered}

    st.sidebar.markdown(f"**Graph Stats (filtered):** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # ------------------------------------------------------------------------
    # Optional: reconstruct influenced graph
    # ------------------------------------------------------------------------
    if sidebar_vals['use_reconstruction'] and run_button and user_query:
        with st.spinner("🔄 Reconstructing graph with attention + LLM..."):
            G_influenced = reconstruct_graph_with_attention(
                G_original, user_query, params,
                st.session_state.llm_tokenizer, st.session_state.llm_model,
                term_embeddings_dict if st.session_state.scibert_loaded else {},
                max_nodes=sidebar_vals['max_recon_nodes'],
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
            if sidebar_vals['use_multi_res']:
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

    uncertainty_orig = {}
    if sidebar_vals['use_bootstrap'] and results_orig is not None and analysis_type == "Centrality Analysis":
        with st.spinner("Computing bootstrap centrality uncertainty..."):
            mean_cent, ci_lower, ci_upper = bootstrap_centrality(G_filtered, n_samples=sidebar_vals['bootstrap_samples'], metric='degree')
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
    # Quantitative Measures Tabs (Reconstruction Mode)
    # ------------------------------------------------------------------------
    if sidebar_vals['use_reconstruction'] and results_orig is not None and results_inf is not None:
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
    # Visualization – Plotly or Pyvis (with sampling for large graphs)
    # ------------------------------------------------------------------------
    def plot_graph_plotly(graph, title, size_attr="priority_score", use_attention_hover=False):
        if graph.number_of_nodes() == 0:
            return None
        # Sample if graph is too large ( > 500 nodes)
        if graph.number_of_nodes() > 500:
            st.warning("Graph too large (>500 nodes). Showing top 500 by priority.")
            nodes_to_keep = sorted(graph.nodes(), key=lambda n: graph.nodes[n].get(size_attr, 0), reverse=True)[:500]
            graph = graph.subgraph(nodes_to_keep).copy()

        cats_in_graph = list(set([graph.nodes[n].get('category','Unknown') for n in graph.nodes()]))
        color_pal = px.colors.qualitative.Set3 if len(cats_in_graph) <= 10 else px.colors.qualitative.Alphabet
        color_map = {c: color_pal[i % len(color_pal)] for i, c in enumerate(cats_in_graph)}
        node_colors = [color_map.get(graph.nodes[n].get('category','Unknown'), 'lightgray') for n in graph.nodes()]

        pos = nx.spring_layout(graph, k=1, iterations=100, seed=42, weight='weight')

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
                width = 0.5 + 4.5 * sidebar_vals['edge_width'] * (w - ew_min) / (ew_max - ew_min)
            else:
                width = 2.0 * sidebar_vals['edge_width']
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
        if sidebar_vals['use_reconstruction'] and G_influenced.number_of_nodes() > 0:
            col1, col2 = st.columns(2)
            with col1:
                if sidebar_vals['viz_engine'] == "Plotly (static)":
                    fig_orig = plot_graph_plotly(G_filtered, "Original Filtered Graph", size_attr="priority_score", use_attention_hover=False)
                    if fig_orig:
                        st.plotly_chart(fig_orig, use_container_width=True)
                else:
                    if PYVIS_AVAILABLE:
                        # Pyvis also benefits from sampling if large
                        if G_filtered.number_of_nodes() > 500:
                            nodes_to_keep = sorted(G_filtered.nodes(), key=lambda n: G_filtered.nodes[n].get('priority_score', 0), reverse=True)[:500]
                            G_viz = G_filtered.subgraph(nodes_to_keep).copy()
                        else:
                            G_viz = G_filtered
                        html_orig = create_pyvis_network(G_viz, "Original Filtered Graph", size_attr="priority_score", use_attention_hover=False)
                        if html_orig:
                            st.components.v1.html(html_orig, height=650)
                    else:
                        st.warning("Pyvis not installed. Install with `pip install pyvis` to use interactive graphs.")
            with col2:
                if sidebar_vals['viz_engine'] == "Plotly (static)":
                    fig_inf = plot_graph_plotly(G_influenced, "LLM‑Influenced Graph", size_attr="attention", use_attention_hover=True)
                    if fig_inf:
                        st.plotly_chart(fig_inf, use_container_width=True)
                else:
                    if PYVIS_AVAILABLE:
                        if G_influenced.number_of_nodes() > 500:
                            nodes_to_keep = sorted(G_influenced.nodes(), key=lambda n: G_influenced.nodes[n].get('attention', 0), reverse=True)[:500]
                            G_viz = G_influenced.subgraph(nodes_to_keep).copy()
                        else:
                            G_viz = G_influenced
                        html_inf = create_pyvis_network(G_viz, "LLM‑Influenced Graph", size_attr="attention", use_attention_hover=True)
                        if html_inf:
                            st.components.v1.html(html_inf, height=650)
                    else:
                        st.warning("Pyvis not installed. Install with `pip install pyvis` to use interactive graphs.")
        else:
            if sidebar_vals['viz_engine'] == "Plotly (static)":
                fig = plot_graph_plotly(G_filtered, f"Battery Degradation Graph - {analysis_type}", size_attr="priority_score", use_attention_hover=False)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                if PYVIS_AVAILABLE:
                    if G_filtered.number_of_nodes() > 500:
                        nodes_to_keep = sorted(G_filtered.nodes(), key=lambda n: G_filtered.nodes[n].get('priority_score', 0), reverse=True)[:500]
                        G_viz = G_filtered.subgraph(nodes_to_keep).copy()
                    else:
                        G_viz = G_filtered
                    html = create_pyvis_network(G_viz, f"Battery Degradation Graph - {analysis_type}", size_attr="priority_score", use_attention_hover=False)
                    if html:
                        st.components.v1.html(html, height=650)
                else:
                    st.warning("Pyvis not installed. Install with `pip install pyvis` to use interactive graphs.")

        # Additional Analytics (only if graph not too large)
        if G_filtered.number_of_nodes() < 500:
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
        else:
            st.info("Graph too large to show detailed analytics. Switch to reconstruction mode or reduce filters.")

    # Export functionality (unchanged)
    with st.sidebar:
        st.markdown("### 💾 Export")
        if st.button("Export Filtered Graph as CSV"):
            nodes_exp = pd.DataFrame([{'node': n, **G_filtered.nodes[n]} for n in G_filtered.nodes()])
            edges_exp = pd.DataFrame([{'source': u, 'target': v, **G_filtered.edges[u, v]} for u, v in G_filtered.edges()])
            st.download_button("Download Nodes", nodes_exp.to_csv(index=False), "nodes.csv")
            st.download_button("Download Edges", edges_exp.to_csv(index=False), "edges.csv")
        if st.button("Export as GraphML"):
            buf = BytesIO()
            nx.write_graphml(G_filtered, buf)
            st.download_button("Download GraphML", buf.getvalue(), "graph.graphml")
        if st.button("Export as JSON-LD"):
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
