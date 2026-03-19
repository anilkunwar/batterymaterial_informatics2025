#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT BATTERY DEGRADATION KNOWLEDGE EXPLORER – ENHANCED
===============================================================
Fully data‑grounded; LLM used only for parsing and inference on pre‑computed JSON.
Now includes advanced graph metrics, physics equation relevance, and operational modulation.
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
from collections import Counter, OrderedDict
import numpy as np
import traceback
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
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

# ----------------------------------------------------------------------------
# Determine if transformers are available (set TRANSFORMERS_AVAILABLE)
# ----------------------------------------------------------------------------
try:
    from transformers import AutoModel, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create dummy objects so the code can still run without transformers
    AutoModel = AutoTokenizer = GPT2Tokenizer = GPT2LMHeadModel = AutoModelForCausalLM = None
    torch = None

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL CONFIGURATION – ROBUST DATA DIRECTORY DETECTION
# ============================================================================
def get_data_dir() -> str:
    """
    Determine the most likely directory containing knowledge_graph_edges.csv and knowledge_graph_nodes.csv.
    Checks, in order:
      1. Environment variable BATTERY_DATA_DIR
      2. Script directory (if __file__ exists) and its 'data' subfolder
      3. Current working directory and its 'data' subfolder
    Returns a string path (guaranteed to exist or be a reasonable default).
    """
    # 1. Environment variable override
    env_dir = os.environ.get("BATTERY_DATA_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
        if p.is_dir():
            return str(p.resolve())

    # Helper to check if a directory contains either of the required files
    def has_data_files(dir_path: pathlib.Path) -> bool:
        return (dir_path / "knowledge_graph_nodes.csv").exists() or (dir_path / "knowledge_graph_edges.csv").exists()

    # 2. Script directory (if we are in a script)
    if '__file__' in globals():
        script_dir = pathlib.Path(__file__).resolve().parent
        if has_data_files(script_dir):
            return str(script_dir)
        # Check data/ subfolder
        data_sub = script_dir / "data"
        if data_sub.is_dir() and has_data_files(data_sub):
            return str(data_sub)

    # 3. Current working directory
    cwd = pathlib.Path.cwd()
    if has_data_files(cwd):
        return str(cwd)
    cwd_data = cwd / "data"
    if cwd_data.is_dir() and has_data_files(cwd_data):
        return str(cwd_data)

    # 4. Fallback – script dir if available, else cwd (no guarantee files exist)
    if '__file__' in globals():
        return str(pathlib.Path(__file__).resolve().parent)
    return str(cwd)

DB_DIR = get_data_dir()
logger.info(f"Data directory set to: {DB_DIR}")

APP_VERSION = "6.0.0"   # Enhanced version

# ----------------------------------------------------------------------------
# 25+ PHYSICS EQUATIONS (expanded, now with full descriptions for embedding matching)
# ----------------------------------------------------------------------------
PHYSICS_EQUATIONS = {
    "diffusion_stress": {
        "equation": r"σ = \frac{E \Omega \Delta c}{1 - \nu}",
        "description": "Diffusion-induced stress in active particles. Relates stress to concentration gradient.",
        "variables": {"E": "Young's modulus (GPa)", "Ω": "Partial molar volume (m³/mol)",
                      "Δc": "Concentration gradient (mol/m³)", "ν": "Poisson's ratio"},
        "unit": "MPa"
    },
    "sei_growth": {
        "equation": r"δ_{SEI} \propto \sqrt{t}",
        "description": "SEI layer growth kinetics (parabolic). Thickness increases with square root of time.",
        "variables": {"δ_SEI": "SEI thickness (nm)", "t": "Time (hours)"},
        "unit": "nm"
    },
    "chemo_mechanical": {
        "equation": r"LAM \propto \int \sigma \, dN",
        "description": "Loss of Active Material from stress cycling. Accumulated stress over cycles.",
        "variables": {"LAM": "Loss of Active Material (%)", "σ": "Stress (MPa)", "N": "Cycle number"},
        "unit": "%"
    },
    "lithium_plating": {
        "equation": r"Risk \uparrow \text{ when } V < V_{plate} \text{ or } T < 0°C",
        "description": "Lithium plating risk conditions. Plating occurs at low voltage or low temperature.",
        "variables": {"V": "Cell voltage (V)", "V_plate": "Plating potential (~0.1V vs Li/Li+)", "T": "Temperature (°C)"},
        "unit": "V/°C"
    },
    "capacity_fade": {
        "equation": r"Q_{loss} = Q_0 - \int I(t) \, dt",
        "description": "Integrated current loss over time. Capacity loss as ampere-hour throughput.",
        "variables": {"Q_loss": "Capacity loss (mAh)", "Q_0": "Initial capacity (mAh)", "I": "Current (A)"},
        "unit": "%"
    },
    "crack_propagation": {
        "equation": r"\frac{da}{dN} = C(\Delta K)^m",
        "description": "Paris law for fatigue crack growth. Crack growth per cycle as function of stress intensity.",
        "variables": {"a": "Crack length (μm)", "N": "Cycle number", "ΔK": "Stress intensity factor (MPa·m^½)", "C, m": "Material constants"},
        "unit": "μm"
    },
    "nernst_equation": {
        "equation": r"E = E^0 - \frac{RT}{nF} \ln Q",
        "description": "Electrode potential under non-standard conditions. Relates potential to reaction quotient.",
        "variables": {"E": "Cell potential (V)", "E^0": "Standard potential (V)", "R": "Gas constant", "T": "Temperature (K)", "n": "Number of electrons", "F": "Faraday constant", "Q": "Reaction quotient"},
        "unit": "V"
    },
    "butler_volmer": {
        "equation": r"j = j_0 \left[ \exp\left(\frac{\alpha n F \eta}{RT}\right) - \exp\left(-\frac{(1-\alpha) n F \eta}{RT}\right) \right]",
        "description": "Electrode kinetics (current density). Relates current to overpotential.",
        "variables": {"j": "Current density (A/m²)", "j_0": "Exchange current density", "α": "Charge transfer coefficient", "η": "Overpotential (V)"},
        "unit": "A/m²"
    },
    "heat_generation": {
        "equation": r"Q_{gen} = I^2 R + I T \frac{dU}{dT}",
        "description": "Total heat generation (Joule + reversible).",
        "variables": {"Q_gen": "Heat generation rate (W)", "I": "Current (A)", "R": "Internal resistance (Ω)", "dU/dT": "Entropy coefficient (V/K)"},
        "unit": "W"
    },
    "thermal_runaway": {
        "equation": r"\frac{dT}{dt} = \frac{Q_{gen} - Q_{diss}}{m C_p}",
        "description": "Temperature rise rate. Balance of heat generation and dissipation.",
        "variables": {"T": "Temperature (°C)", "Q_gen": "Heat generation (W)", "Q_diss": "Heat dissipation (W)", "m": "Mass (kg)", "C_p": "Specific heat (J/kg·K)"},
        "unit": "°C"
    },
    "calendar_aging": {
        "equation": r"Q_{loss} \propto \sqrt{t} \cdot \exp\left(-\frac{E_a}{RT}\right)",
        "description": "Calendar aging with Arrhenius temperature dependence.",
        "variables": {"Q_loss": "Capacity loss (%)", "t": "Time (months)", "E_a": "Activation energy (kJ/mol)", "T": "Temperature (K)"},
        "unit": "%/year"
    },
    "cycle_aging": {
        "equation": r"Q_{loss} = A \cdot N^z \cdot \text{DOD}^b",
        "description": "Cycle aging model. Capacity loss as function of cycles and depth of discharge.",
        "variables": {"N": "Cycle number", "DOD": "Depth of Discharge (%)", "A, z, b": "Fitting parameters"},
        "unit": "cycles"
    },
    "diffusion_coefficient": {
        "equation": r"D = D_0 \exp\left(-\frac{E_a}{RT}\right)",
        "description": "Temperature-dependent diffusion coefficient. Arrhenius behavior.",
        "variables": {"D": "Diffusion coefficient (cm²/s)", "D_0": "Pre-exponential factor", "E_a": "Activation energy (kJ/mol)"},
        "unit": "cm²/s"
    },
    "migration_flux": {
        "equation": r"J = -D \nabla c + \frac{z F D c}{RT} \nabla \phi",
        "description": "Nernst-Planck ion flux (diffusion + migration).",
        "variables": {"J": "Ion flux (mol/m²·s)", "c": "Concentration (mol/m³)", "φ": "Electric potential (V)"},
        "unit": "mol/m²·s"
    },
    "gas_evolution": {
        "equation": r"V_{gas} \propto \int I_{parasitic} \, dt",
        "description": "Gas volume from parasitic reactions.",
        "variables": {"V_gas": "Gas volume (mL)", "I_parasitic": "Parasitic current (mA)"},
        "unit": "mL"
    },
    "stress_intensity": {
        "equation": r"K_I = Y \sigma \sqrt{\pi a}",
        "description": "Stress intensity factor for mode I fracture. Depends on crack length and stress.",
        "variables": {"K_I": "Stress intensity (MPa·m^½)", "Y": "Geometric factor", "σ": "Applied stress (MPa)", "a": "Crack length (m)"},
        "unit": "MPa·m^½"
    },
    "young_modulus": {
        "equation": r"E = \frac{\sigma}{\epsilon}",
        "description": "Young's modulus, ratio of stress to strain in elastic region.",
        "variables": {"E": "Young's modulus (GPa)", "σ": "Stress (MPa)", "ε": "Strain"},
        "unit": "GPa"
    },
    "poisson_ratio": {
        "equation": r"\nu = -\frac{\epsilon_{lateral}}{\epsilon_{axial}}",
        "description": "Poisson's ratio, negative ratio of transverse to axial strain.",
        "variables": {"ν": "Poisson's ratio", "ε_lateral": "Lateral strain", "ε_axial": "Axial strain"},
        "unit": "dimensionless"
    },
    "fracture_toughness": {
        "equation": r"K_{IC} = Y \sigma_f \sqrt{\pi a_c}",
        "description": "Fracture toughness, critical stress intensity for crack propagation.",
        "variables": {"K_IC": "Fracture toughness (MPa·m^½)", "σ_f": "Fracture stress (MPa)", "a_c": "Critical crack length (m)"},
        "unit": "MPa·m^½"
    },
    "electrochemical_strain": {
        "equation": r"\epsilon_{chem} = \beta \Delta c",
        "description": "Chemical strain due to concentration change.",
        "variables": {"ε_chem": "Chemical strain", "β": "Expansion coefficient (m³/mol)", "Δc": "Concentration change (mol/m³)"},
        "unit": "dimensionless"
    },
    "sei_ionic_resistance": {
        "equation": r"R_{SEI} = \frac{\delta_{SEI}}{\kappa_{SEI}}",
        "description": "Ionic resistance of SEI layer.",
        "variables": {"R_SEI": "Resistance (Ω·m²)", "δ_SEI": "Thickness (m)", "κ_SEI": "Ionic conductivity (S/m)"},
        "unit": "Ω·m²"
    },
    "exchange_current": {
        "equation": r"j_0 = F k^0 (c_{ox})^{1-\alpha} (c_{red})^{\alpha}",
        "description": "Exchange current density from rate constant and concentrations.",
        "variables": {"j_0": "Exchange current density (A/m²)", "k^0": "Standard rate constant (m/s)", "c_ox", "c_red": "Concentrations (mol/m³)", "α": "Transfer coefficient"},
        "unit": "A/m²"
    },
    "porosity_evolution": {
        "equation": r"\frac{\partial \epsilon}{\partial t} = -\frac{M_{SEI}}{\rho_{SEI}} j_{SEI}",
        "description": "Porosity change due to SEI growth.",
        "variables": {"ε": "Porosity", "M_SEI": "Molar mass (kg/mol)", "ρ_SEI": "Density (kg/m³)", "j_SEI": "SEI current (A/m²)"},
        "unit": "s⁻¹"
    },
    "li_diffusion_time": {
        "equation": r"\tau = \frac{L^2}{D}",
        "description": "Characteristic diffusion time in electrode particle.",
        "variables": {"τ": "Diffusion time (s)", "L": "Diffusion length (m)", "D": "Diffusion coefficient (m²/s)"},
        "unit": "s"
    }
}

# Extended physics terms for semantic boosting (now derived from equation descriptions)
PHYSICS_TERMS = list(set([
    term.lower() for eq in PHYSICS_EQUATIONS.values()
    for term in eq["description"].split()
    if len(term) > 3
])) + [
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

# Original key terms (from the snippet)
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

# Operational constraint bounds (all floats for type safety)
OPERATIONAL_CONSTRAINTS = {
    "c_rate": {"min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1, "unit": "C"},
    "voltage": {"min": 2.5, "max": 4.5, "default": 3.7, "step": 0.1, "unit": "V"},
    "temperature": {"min": -20.0, "max": 80.0, "default": 25.0, "step": 1.0, "unit": "°C"},
    "soc": {"min": 0.0, "max": 100.0, "default": 50.0, "step": 5.0, "unit": "%"},
    "dod": {"min": 0.0, "max": 100.0, "default": 80.0, "step": 5.0, "unit": "%"}
}

# ============================================================================
# DATA CLASSES FOR TYPE-SAFE PARAMETER HANDLING
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
    selected_categories: List[str] = field(default_factory=lambda: ["Crack and Fracture", "Deformation", "Degradation", "Fatigue"])
    selected_types: List[str] = field(default_factory=lambda: ["category", "term"])
    selected_nodes: List[str] = field(default_factory=lambda: ["electrode cracking", "SEI formation", "capacity fade"])
    excluded_terms: List[str] = field(default_factory=lambda: ["battery", "material"])
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

    def to_dict(self) -> Dict:
        result = {}
        for key, value in asdict(self).items():
            result[key] = value
        return result

# ============================================================================
# SCIBERT LOADER & EMBEDDING UTILITIES (no Streamlit calls at global scope)
# ============================================================================
# These globals will be set inside main() after page config
scibert_tokenizer = None
scibert_model = None
KEY_TERMS_EMBEDDINGS = []
PHYSICS_TERMS_EMBEDDINGS = []
EQUATION_DESCRIPTION_EMBEDDINGS = {}  # mapping equation name -> embedding

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
    """Uses the globally set scibert_tokenizer and scibert_model."""
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
    """Helper to compute and cache embeddings for a list of texts."""
    return get_scibert_embedding(texts)

def compute_equation_embeddings():
    """Pre‑compute embeddings for each physics equation description."""
    global scibert_tokenizer, scibert_model
    eq_emb = {}
    for name, data in PHYSICS_EQUATIONS.items():
        desc = data["description"]
        emb = get_scibert_embedding(desc)
        eq_emb[name] = emb
    return eq_emb

# ============================================================================
# UNIT-AWARE PARSING (simple version)
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
# ADVANCED GRAPH METRICS AND PHYSICS RELEVANCE
# ============================================================================
def compute_advanced_metrics(G):
    """Compute PageRank, clustering coefficient, and k‑core decomposition for all nodes."""
    pagerank = nx.pagerank(G, weight='weight')
    clustering = nx.clustering(G, weight='weight')
    # k‑core: core number for each node
    core_numbers = nx.core_number(G)
    return pagerank, clustering, core_numbers

def compute_physics_relevance_for_nodes(G, equation_embeddings):
    """
    For each node, compute its maximum cosine similarity to any physics equation description.
    Returns a dict node -> physics_relevance (0..1).
    """
    if not equation_embeddings:
        return {n: 0.0 for n in G.nodes()}
    node_emb_dict = {}
    nodes = list(G.nodes())
    embs = get_scibert_embedding(nodes)
    for n, emb in zip(nodes, embs):
        if emb is not None:
            node_emb_dict[n] = emb
    relevance = {}
    for n in G.nodes():
        n_emb = node_emb_dict.get(n)
        if n_emb is None:
            relevance[n] = 0.0
        else:
            max_sim = 0.0
            for eq_emb in equation_embeddings.values():
                if eq_emb is not None:
                    sim = cosine_similarity([n_emb], [eq_emb])[0][0]
                    if sim > max_sim:
                        max_sim = sim
            relevance[n] = max_sim
    return relevance

# ============================================================================
# ENHANCED PRIORITY SCORE CALCULATION (with advanced metrics and operational modulation)
# ============================================================================
def calculate_priority_scores(G, nodes_df, physics_boost_weight=0.15, operational_params=None,
                               physics_relevance=None, pagerank=None, clustering=None, core_numbers=None):
    """
    Enhanced priority scoring with:
      - Frequency (0.25)
      - Degree centrality (0.15)
      - Betweenness centrality (0.15)
      - PageRank (0.10)
      - Clustering coefficient (0.05)
      - Core number (0.05)
      - Semantic relevance to KEY_TERMS (0.10)
      - Physics relevance (physics_boost_weight, default 0.15)
    Operational constraints (C-rate, voltage, temperature) modulate physics relevance.
    """
    max_freq = nodes_df['frequency'].max() if nodes_df['frequency'].max() > 0 else 1
    nodes_df['norm_frequency'] = nodes_df['frequency'] / max_freq

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # If advanced metrics not provided, compute them
    if pagerank is None:
        pagerank = nx.pagerank(G, weight='weight')
    if clustering is None:
        clustering = nx.clustering(G, weight='weight')
    if core_numbers is None:
        core_numbers = nx.core_number(G)

    # Normalize core numbers to [0,1]
    max_core = max(core_numbers.values()) if core_numbers else 1
    core_norm = {n: v / max_core for n, v in core_numbers.items()}

    # Semantic relevance to KEY_TERMS (using pre‑computed embeddings)
    node_terms = nodes_df['node'].tolist()
    term_embeddings = get_scibert_embedding(node_terms)
    term_embeddings_dict = dict(zip(node_terms, term_embeddings))

    semantic_scores = {}
    for node in G.nodes():
        emb = term_embeddings_dict.get(node)
        if emb is None:
            semantic_scores[node] = 0
        else:
            sims = [cosine_similarity([emb], [kt_emb])[0][0] for kt_emb in KEY_TERMS_EMBEDDINGS if kt_emb is not None]
            semantic_scores[node] = max(sims, default=0)

    # Physics relevance (if not provided, compute from equation embeddings)
    if physics_relevance is None:
        physics_relevance = compute_physics_relevance_for_nodes(G, EQUATION_DESCRIPTION_EMBEDDINGS)

    # Operational constraint modulation: boost physics relevance if node matches extreme conditions
    def operational_modulation(node, base_phys):
        mod = 1.0
        if operational_params:
            if operational_params.get('temperature', 25) > 45 and any(t in node.lower() for t in ['thermal', 'temp', 'heat']):
                mod *= 1.3
            if operational_params.get('c_rate', 1.0) > 2.0 and any(t in node.lower() for t in ['rate', 'power', 'current']):
                mod *= 1.3
            if operational_params.get('voltage', 3.7) > 4.2 and any(t in node.lower() for t in ['voltage', 'potential']):
                mod *= 1.3
            if operational_params.get('dod', 80) > 80 and any(t in node.lower() for t in ['depth', 'discharge']):
                mod *= 1.2
            if operational_params.get('soc', 50) < 20 and any(t in node.lower() for t in ['low soc', 'overdischarge']):
                mod *= 1.2
        return min(base_phys * mod, 1.0)

    physics_modulated = {n: operational_modulation(n, physics_relevance.get(n, 0)) for n in G.nodes()}

    # Weight definitions
    w_f, w_d, w_b, w_pr, w_cl, w_core, w_s, w_p = 0.25, 0.15, 0.15, 0.10, 0.05, 0.05, 0.10, physics_boost_weight
    total = w_f + w_d + w_b + w_pr + w_cl + w_core + w_s + w_p
    w_f, w_d, w_b, w_pr, w_cl, w_core, w_s, w_p = [w/total for w in [w_f, w_d, w_b, w_pr, w_cl, w_core, w_s, w_p]]

    priority_scores = {}
    for node in G.nodes():
        freq = nodes_df[nodes_df['node'] == node]['norm_frequency'].iloc[0] if len(nodes_df[nodes_df['node'] == node]) > 0 else 0
        priority_scores[node] = (
            w_f * freq +
            w_d * degree_centrality.get(node, 0) +
            w_b * betweenness_centrality.get(node, 0) +
            w_pr * pagerank.get(node, 0) +
            w_cl * clustering.get(node, 0) +
            w_core * core_norm.get(node, 0) +
            w_s * semantic_scores.get(node, 0) +
            w_p * physics_modulated.get(node, 0)
        )
    return priority_scores

# ============================================================================
# FAILURE ANALYSIS FUNCTIONS (enhanced to include new metrics)
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
    pagerank = nx.pagerank(G_filtered, weight='weight')
    clustering = nx.clustering(G_filtered, weight='weight')
    core_numbers = nx.core_number(G_filtered)
    max_core = max(core_numbers.values()) if core_numbers else 1
    core_norm = {n: v/max_core for n,v in core_numbers.items()}

    results = []
    for node in G_filtered.nodes():
        if any(term in node.lower() for term in focus_terms):
            results.append({
                'node': node,
                'degree': degree.get(node, 0),
                'betweenness': between.get(node, 0),
                'closeness': closeness.get(node, 0),
                'eigenvector': eigen.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'clustering': clustering.get(node, 0),
                'core_number': core_numbers.get(node, 0),
                'core_norm': core_norm.get(node, 0),
                'category': G_filtered.nodes[node].get('category', ''),
                'type': G_filtered.nodes[node].get('type', '')
            })
    return pd.DataFrame(results)

def find_failure_pathways(G_filtered, source_terms, target_terms, require_physics=False,
                          min_physics_similarity=0.5, physics_relevance=None):
    """
    Enhanced pathway analysis: if physics_relevance dict is given, compute avg physics along path.
    Also add physics_weighted_paths that maximize sum of physics relevance.
    """
    pathways = {}
    physics_weighted_paths = []  # for new analysis type

    for src in source_terms:
        for tgt in target_terms:
            if src in G_filtered.nodes() and tgt in G_filtered.nodes():
                try:
                    # Get all shortest paths by weight (or unweighted if weight not used)
                    paths = list(nx.all_shortest_paths(G_filtered, source=src, target=tgt, weight='weight'))
                    if require_physics and physics_relevance is not None:
                        # Filter paths that have at least one node with physics relevance > min_physics_similarity
                        filtered = []
                        for p in paths:
                            max_rel = max(physics_relevance.get(n, 0) for n in p)
                            if max_rel >= min_physics_similarity:
                                filtered.append(p)
                        paths = filtered
                    if paths:
                        # Use the first path for simplicity
                        path = paths[0]
                        # Compute average physics relevance along path
                        if physics_relevance is not None:
                            phys_scores = [physics_relevance.get(n, 0) for n in path]
                            avg_phys = np.mean(phys_scores)
                            sum_phys = np.sum(phys_scores)
                        else:
                            avg_phys = 0
                            sum_phys = 0
                        pathways[f"{src} -> {tgt}"] = {
                            'path': path,
                            'length': len(path)-1,
                            'nodes': path,
                            'num_paths': len(paths),
                            'contains_physics': any(physics_relevance.get(n, 0) > 0 for n in path) if physics_relevance else False,
                            'avg_physics_similarity': avg_phys,
                            'sum_physics': sum_phys
                        }
                        # Also store for physics-weighted ranking
                        physics_weighted_paths.append({
                            'path': path,
                            'source': src,
                            'target': tgt,
                            'sum_physics': sum_phys,
                            'avg_physics': avg_phys,
                            'length': len(path)-1
                        })
                    else:
                        pathways[f"{src} -> {tgt}"] = {'path': None, 'length': float('inf'), 'nodes': [], 'contains_physics': False, 'avg_physics_similarity': 0, 'sum_physics': 0}
                except nx.NetworkXNoPath:
                    pathways[f"{src} -> {tgt}"] = {'path': None, 'length': float('inf'), 'nodes': [], 'contains_physics': False, 'avg_physics_similarity': 0, 'sum_physics': 0}
    # Sort physics_weighted_paths by sum_physics descending
    physics_weighted_paths.sort(key=lambda x: x['sum_physics'], reverse=True)
    return pathways, physics_weighted_paths

# ============================================================================
# STRUCTURED INSIGHT GENERATOR (produces countable JSON, now enriched)
# ============================================================================
class DegradationInsightGenerator:
    @staticmethod
    def generate_structured_insights(
        analysis_results,
        analysis_type: str,
        parsed_params: Dict,
        graph_stats: Dict,
        user_query: str = "",
        physics_relevance: Optional[Dict] = None,
        pagerank: Optional[Dict] = None,
        clustering: Optional[Dict] = None,
        core_numbers: Optional[Dict] = None
    ) -> Dict:
        """
        Returns a fully numerical, countable JSON object.
        Every entry has a composite_weight (0–1) that can be sorted/ranked.
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
            "physics_weighted_pathways": [],
            "communities": [],
            "strong_correlations": [],
            "temporal_trend": {},
            "operational_constraints": {
                "c_rate": parsed_params.get("c_rate", 1.0),
                "voltage": parsed_params.get("voltage", 3.7),
                "temperature": parsed_params.get("temperature", 25.0),
                "soc": parsed_params.get("soc", 50.0),
                "dod": parsed_params.get("dod", 80.0)
            }
        }

        # ------------------------------------------------------------------
        # 1. Ranked Mechanisms (Centrality / Community)
        # ------------------------------------------------------------------
        if analysis_type == "Centrality Analysis" and isinstance(analysis_results, pd.DataFrame) and not analysis_results.empty:
            top = analysis_results.nlargest(15, "degree")  # take more, we'll re-weight
            for _, row in top.iterrows():
                node = row["node"]
                # Physics match from pre‑computed physics_relevance
                phys_match = physics_relevance.get(node, 0.0) if physics_relevance else 0.0

                # Operational boost (already factored into physics_relevance via modulation, but we keep it separate for transparency)
                operational_boost = 1.0
                if parsed_params.get("temperature", 25) > 45 and any(t in node.lower() for t in ["thermal", "temp"]):
                    operational_boost = 1.3
                if parsed_params.get("c_rate", 1.0) > 2.0 and any(t in node.lower() for t in ["rate", "power"]):
                    operational_boost = 1.3
                if parsed_params.get("voltage", 3.7) > 4.2 and any(t in node.lower() for t in ["voltage", "potential"]):
                    operational_boost = 1.3

                # Get advanced metrics
                pr_val = pagerank.get(node, 0) if pagerank else 0
                clust_val = clustering.get(node, 0) if clustering else 0
                core_val = core_numbers.get(node, 0) if core_numbers else 0
                max_core = max(core_numbers.values()) if core_numbers else 1
                core_norm = core_val / max_core if max_core > 0 else 0

                # Composite weight: combine degree, betweenness, physics, pagerank, clustering, core
                comp = (
                    0.3 * row["degree"] +
                    0.15 * row.get("betweenness", 0) +
                    0.2 * phys_match +
                    0.1 * pr_val +
                    0.1 * clust_val +
                    0.1 * core_norm +
                    0.05 * operational_boost
                )
                # Clamp to 0-1
                comp = min(max(comp, 0), 1)

                # Find matching physics equation (closest by embedding)
                equation = ""
                if phys_match > 0.3 and EQUATION_DESCRIPTION_EMBEDDINGS:
                    # find equation with highest similarity to node embedding
                    node_emb = get_scibert_embedding(node)
                    if node_emb is not None:
                        best_eq = None
                        best_sim = 0
                        for eq_name, eq_emb in EQUATION_DESCRIPTION_EMBEDDINGS.items():
                            if eq_emb is not None:
                                sim = cosine_similarity([node_emb], [eq_emb])[0][0]
                                if sim > best_sim:
                                    best_sim = sim
                                    best_eq = eq_name
                        if best_eq:
                            equation = PHYSICS_EQUATIONS[best_eq]["equation"]

                entry = {
                    "name": node,
                    "degree": round(row["degree"], 3),
                    "betweenness": round(row.get("betweenness", 0), 3),
                    "pagerank": round(pr_val, 3),
                    "clustering": round(clust_val, 3),
                    "core_number": core_val,
                    "physics_match": round(phys_match, 3),
                    "operational_boost": round(operational_boost, 2),
                    "composite_weight": round(comp, 3),
                    "equation": equation
                }
                data["ranked_mechanisms"].append(entry)

        # ------------------------------------------------------------------
        # 2. Pathways (standard)
        # ------------------------------------------------------------------
        if analysis_type == "Pathway Analysis" and isinstance(analysis_results, tuple) and len(analysis_results)==2:
            pathways_dict, phys_weighted = analysis_results
            for name, p in pathways_dict.items():
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
            # Also add physics-weighted pathways
            for pw in phys_weighted[:10]:
                data["physics_weighted_pathways"].append({
                    "source": pw["source"],
                    "target": pw["target"],
                    "path": " → ".join(pw["path"]),
                    "length": pw["length"],
                    "sum_physics": round(pw["sum_physics"], 3),
                    "avg_physics": round(pw["avg_physics"], 3)
                })

        # ------------------------------------------------------------------
        # 3. Communities (enhanced with physics count)
        # ------------------------------------------------------------------
        if analysis_type == "Community Detection" and isinstance(analysis_results, dict):
            for cid, c in list(analysis_results.items())[:5]:
                phys_count = sum(c.get("physics_terms", Counter()).values())
                size = len(c["nodes"])
                # Also compute average physics relevance in community if physics_relevance available
                if physics_relevance:
                    avg_phys = np.mean([physics_relevance.get(n, 0) for n in c["nodes"]])
                else:
                    avg_phys = 0
                data["communities"].append({
                    "community_id": cid,
                    "size": size,
                    "physics_terms_count": phys_count,
                    "avg_physics_relevance": round(avg_phys, 3),
                    "composite_weight": round(phys_count / max(size, 1) * avg_phys, 3)
                })

        # ------------------------------------------------------------------
        # 4. Correlations (top 3)
        # ------------------------------------------------------------------
        if analysis_type == "Correlation Analysis" and isinstance(analysis_results, tuple) and len(analysis_results)==2:
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

        # ------------------------------------------------------------------
        # 5. Temporal trend (simplified) – FIXED to handle non-dict values
        # ------------------------------------------------------------------
        if analysis_type == "Temporal Analysis" and isinstance(analysis_results, dict):
            periods = sorted(analysis_results.keys())
            if periods:
                first, last = periods[0], periods[-1]
                first_val = analysis_results[first]
                last_val = analysis_results[last]
                if isinstance(first_val, dict):
                    fc_first = first_val.get('failure_concepts', 0)
                else:
                    fc_first = 0
                if isinstance(last_val, dict):
                    fc_last = last_val.get('failure_concepts', 0)
                else:
                    fc_last = 0
                data["temporal_trend"] = {
                    "first_period": first,
                    "last_period": last,
                    "failure_concepts_change": fc_last - fc_first,
                    "direction": "increasing" if fc_last > fc_first else "decreasing"
                }

        # ------------------------------------------------------------------
        # Query-Aware Gating (SciBERT focus boost)
        # ------------------------------------------------------------------
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

        # Sort everything by composite_weight
        data["ranked_mechanisms"].sort(key=lambda x: x["composite_weight"], reverse=True)
        data["pathways"].sort(key=lambda x: x["composite_weight"], reverse=True)
        data["physics_weighted_pathways"].sort(key=lambda x: x["sum_physics"], reverse=True)

        return data

# ============================================================================
# INFERENCE-ONLY LLM CALL (sees only the JSON)
# ============================================================================
def llm_infer_on_insights(structured_json: Dict, user_query: str, tokenizer, model) -> str:
    """LLM sees ONLY the JSON data. Never invents. Pure inference."""
    if tokenizer is None or model is None:
        return "LLM not available."

    prompt = f"""You are a factual battery analyst. Use ONLY the JSON data below.
Answer the user query in 1-3 short bullets, referencing specific numbers from the JSON.
Never add new mechanisms or numbers.
User query: "{user_query}"

JSON data:
{json.dumps(structured_json, indent=2)}

Answer (bullets, using numbers from JSON):"""

    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=300, temperature=0.0, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0], skip_special_tokens=True)
        # extract only the part after "Answer:"
        if "Answer:" in answer:
            return answer.split("Answer:")[-1].strip()
        return answer.strip()
    except Exception as e:
        logger.error(f"LLM inference error: {e}")
        return f"Error generating inference: {e}"

# ============================================================================
# RELEVANCE SCORER (unchanged)
# ============================================================================
class RelevanceScorer:
    def __init__(self, use_scibert=True):
        global scibert_tokenizer, scibert_model
        self.use_scibert = use_scibert and scibert_tokenizer is not None and scibert_model is not None
    def score_query_to_nodes(self, query: str, nodes_list: List[str]) -> float:
        if not query or not nodes_list:
            return 0.5
        if self.use_scibert:
            try:
                q_emb = get_scibert_embedding(query)
                sample = nodes_list[:100]
                n_emb = get_scibert_embedding(sample)
                valid = [i for i, e in enumerate(n_emb) if e is not None]
                if not valid or q_emb is None:
                    return 0.5
                sims = [cosine_similarity([q_emb], [n_emb[i]])[0][0] for i in valid]
                return float(np.mean(sims)) if sims else 0.5
            except:
                return 0.5
        else:
            words = set(query.lower().split())
            matches = sum(1 for n in nodes_list[:100] if any(w in n.lower() for w in words))
            return min(1.0, matches / 50.0)

# ============================================================================
# DATA LOADING FUNCTION – unchanged
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
        st.error(f"❌ Required data file(s) not found in {DB_DIR}: {', '.join(missing)}. "
                 f"Please upload the files to the correct location or set the BATTERY_DATA_DIR environment variable.")
        st.stop()

    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    # Ensure required columns exist (add default values if missing)
    required_node_cols = ['node', 'type', 'category', 'frequency']
    for col in required_node_cols:
        if col not in nodes_df.columns:
            nodes_df[col] = '' if col != 'frequency' else 0

    required_edge_cols = ['source', 'target', 'weight']
    for col in required_edge_cols:
        if col not in edges_df.columns:
            edges_df[col] = '' if col != 'weight' else 0

    return edges_df, nodes_df

# ============================================================================
# UNIFIED LLM LOADER (unchanged)
# ============================================================================
@st.cache_resource(show_spinner="Loading LLM for intelligent parsing...")
def load_llm(backend: str):
    if not TRANSFORMERS_AVAILABLE:
        return None, None, backend
    try:
        if "GPT-2" in backend:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            if "Qwen2-0.5B" in backend:
                model_name = "Qwen/Qwen2-0.5B-Instruct"
            else:  # Qwen2.5-0.5B
                model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
            model.eval()
        return tokenizer, model, backend
    except Exception as e:
        st.warning(f"⚠️ Failed to load {backend}: {str(e)}")
        return None, None, backend

# ============================================================================
# NLP PARSER (unchanged from original, but uses the same class)
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
            'correlation': 'Correlation Analysis',
            'physics pathway': 'Physics‑Weighted Pathway Analysis'  # new
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
        # Unit parsing
        unit_vals = UnitParser.parse_units(text)
        if 'c_rate' in unit_vals:
            params['c_rate'] = unit_vals['c_rate']
        if 'voltage' in unit_vals:
            params['voltage'] = unit_vals['voltage']
        if 'temperature' in unit_vals:
            params['temperature'] = unit_vals['temperature']
        # Analysis type
        for key, val in self.analysis_map.items():
            if key in text_lower:
                params['analysis_type'] = val
                break
        # Source/target terms
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
7. "Find physics‑rich pathways from SEI growth to capacity fade" -> {"analysis_type": "Physics‑Weighted Pathway Analysis", "source_terms": ["SEI growth"], "target_terms": ["capacity fade"], "require_physics_in_pathways": true}
"""
        user = f"{examples}\nText: \"{text}\"\nPreliminary regex: {json.dumps(regex_params, default=str) if regex_params else 'None'}\nJSON:"
        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default)')
        try:
            if "Qwen" in backend:
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
                # Merge with defaults
                for k in asdict(self.defaults).keys():
                    if k not in llm_params:
                        llm_params[k] = asdict(self.defaults)[k]
                # Clip
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
        # Simple confidence: if regex changed a value, trust it more
        regex_conf = {}
        for k, v in regex_params.items():
            if v != getattr(self.defaults, k, None):
                regex_conf[k] = 1.0
            else:
                regex_conf[k] = 0.0
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
        llm_conf = {k: 0.7 if llm_params.get(k) != getattr(self.defaults, k, None) else 0.3 for k in asdict(self.defaults).keys()}
        merged = {}
        for k in asdict(self.defaults).keys():
            merged[k] = regex_params.get(k, getattr(self.defaults, k)) if regex_conf.get(k,0) >= llm_conf.get(k,0) else llm_params.get(k, getattr(self.defaults, k))
        merged['confidence_score'] = (sum(regex_conf.values()) + sum(llm_conf.values())) / (2 * len(asdict(self.defaults)))
        merged['parsing_method'] = 'hybrid'
        return merged

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # --- Page config must be the very first Streamlit command ---
    st.set_page_config(layout="wide", page_title="Intelligent Battery Degradation Explorer (Enhanced)")

    # Now it's safe to call other Streamlit functions
    st.markdown(f"<h1 style='text-align:center;'>🔋 Intelligent Battery Degradation Knowledge Explorer – Enhanced</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>Version {APP_VERSION} — LLM used ONLY for parsing and inference on structured JSON.</p>", unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # Initialize models and embeddings (after page config)
    # ------------------------------------------------------------------------
    global scibert_tokenizer, scibert_model, KEY_TERMS_EMBEDDINGS, PHYSICS_TERMS_EMBEDDINGS, EQUATION_DESCRIPTION_EMBEDDINGS
    scibert_tokenizer, scibert_model = load_scibert()
    # Compute embeddings for key terms and physics terms (cached)
    KEY_TERMS_EMBEDDINGS = compute_embeddings(KEY_TERMS)
    PHYSICS_TERMS_EMBEDDINGS = compute_embeddings(PHYSICS_TERMS)
    # Remove None entries
    KEY_TERMS_EMBEDDINGS = [emb for emb in KEY_TERMS_EMBEDDINGS if emb is not None]
    PHYSICS_TERMS_EMBEDDINGS = [emb for emb in PHYSICS_TERMS_EMBEDDINGS if emb is not None]
    # Compute equation embeddings
    EQUATION_DESCRIPTION_EMBEDDINGS = compute_equation_embeddings()

    # Load data – will stop if files missing
    try:
        edges_df, nodes_df = load_data()
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    # Normalize terms
    def norm(t): return t.lower().strip() if isinstance(t, str) else ""
    nodes_df["node"] = nodes_df["node"].apply(norm)
    edges_df["source"] = edges_df["source"].apply(norm)
    edges_df["target"] = edges_df["target"].apply(norm)

    # Build graph
    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["node"], type=r["type"], category=r["category"], frequency=r["frequency"],
                   unit=r.get("unit","None"), similarity_score=r.get("similarity_score",0))
    for _, r in edges_df.iterrows():
        G.add_edge(r["source"], r["target"], weight=r["weight"], type=r["type"], label=r["label"],
                   relationship=r.get("relationship",""), strength=r.get("strength",0))

    # Initialize session state
    if "parser" not in st.session_state:
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

    # Sidebar for filters
    with st.sidebar:
        st.markdown("## ⚙️ Filters")
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

        # New: colour by physics relevance
        colour_by = st.radio("Colour nodes by", ["Community", "Physics Relevance"], index=0, key="colour_by")

    # Calculate advanced metrics
    pagerank, clustering, core_numbers = compute_advanced_metrics(G)
    # Compute physics relevance for all nodes (using equation descriptions)
    physics_relevance = compute_physics_relevance_for_nodes(G, EQUATION_DESCRIPTION_EMBEDDINGS)

    # Calculate priority scores with current settings
    operational = {"c_rate": c_rate, "voltage": voltage, "temperature": temperature, "soc": soc, "dod": dod}
    priority_scores = calculate_priority_scores(G, nodes_df, physics_boost_weight=physics_boost,
                                                 operational_params=operational,
                                                 physics_relevance=physics_relevance,
                                                 pagerank=pagerank, clustering=clustering,
                                                 core_numbers=core_numbers)
    for n in G.nodes():
        G.nodes[n]['priority_score'] = priority_scores.get(n, 0)
        G.nodes[n]['physics_relevance'] = physics_relevance.get(n, 0)
        G.nodes[n]['pagerank'] = pagerank.get(n, 0)
        G.nodes[n]['clustering'] = clustering.get(n, 0)
        G.nodes[n]['core_number'] = core_numbers.get(n, 0)
    nodes_df['priority_score'] = nodes_df['node'].apply(lambda x: priority_scores.get(x, 0))
    nodes_df['physics_relevance'] = nodes_df['node'].apply(lambda x: physics_relevance.get(x, 0))

    # Filter graph
    G_filtered = filter_graph(G, min_weight, min_freq, selected_cats, selected_types,
                               selected_nodes, excluded_terms, min_priority, suppress)

    st.sidebar.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # ------------------------------------------------------------------------
    # QUERY INTERFACE (LLM parsing)
    # ------------------------------------------------------------------------
    with st.expander("🤖 AI-Powered Query Interface", expanded=True):
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            user_query = st.text_area("Ask about battery degradation in natural language:", height=100,
                                      placeholder="e.g., 'Show physics‑rich pathways from electrode cracking to capacity fade involving diffusion-induced stress'",
                                      key="user_query")
        with col2:
            st.markdown("### 🧠 LLM Settings")
            model_choice = st.selectbox("Model", ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"], index=0, key="llm_choice")
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

    # Process query
    if run_button and user_query:
        with st.spinner("🔍 Parsing query..."):
            parser = st.session_state.parser
            if use_llm and st.session_state.llm_tokenizer is not None:
                params = parser.hybrid_parse(user_query, st.session_state.llm_tokenizer, st.session_state.llm_model,
                                             use_ensemble=use_ensemble, ensemble_runs=ensemble_runs)
            else:
                params = parser.parse_regex(user_query)

            st.session_state.last_params = params

            # Compute relevance
            all_nodes = list(G.nodes())
            relevance = st.session_state.relevance_scorer.score_query_to_nodes(user_query, all_nodes[:100])
            st.info(f"**Semantic Relevance:** {relevance:.3f}")

            # Show parsed parameters
            st.markdown("### 📋 Parsed Parameters")
            cols = st.columns(3)
            for i, (k, v) in enumerate(params.items()):
                if k in ['confidence_score','parsing_method','timestamp']:
                    continue
                with cols[i % 3]:
                    st.metric(k.replace('_',' ').title(), str(v)[:50])

            analysis_type = params.get('analysis_type', 'Centrality Analysis')
    else:
        # If no query or button not pressed, still set a default analysis type
        analysis_type = "Centrality Analysis"

    # Always run analysis with current filter settings (if graph nonempty)
    if G_filtered.number_of_nodes() > 0:
        at = st.session_state.last_params.get('analysis_type', 'Centrality Analysis') if st.session_state.last_params else "Centrality Analysis"

        # Execute analysis
        if at == "Centrality Analysis":
            focus = st.session_state.last_params.get('focus_terms', ['crack','fracture','degradation']) if st.session_state.last_params else None
            results = analyze_failure_centrality(G_filtered, focus)
        elif at == "Community Detection":
            results, _ = detect_failure_communities(G_filtered)
        elif at == "Ego Network Analysis":
            central = st.session_state.last_params.get('central_nodes', ["electrode cracking","SEI formation","capacity fade"]) if st.session_state.last_params else None
            results = analyze_ego_networks(G_filtered, central)
        elif at == "Pathway Analysis" or at == "Physics‑Weighted Pathway Analysis":
            src = st.session_state.last_params.get('source_terms', ['electrode cracking']) if st.session_state.last_params else ['electrode cracking']
            tgt = st.session_state.last_params.get('target_terms', ['capacity fade']) if st.session_state.last_params else ['capacity fade']
            req = st.session_state.last_params.get('require_physics_in_pathways', False) if st.session_state.last_params else False
            min_phys = st.session_state.last_params.get('min_physics_similarity', 0.5) if st.session_state.last_params else 0.5
            results = find_failure_pathways(G_filtered, src, tgt, require_physics=req, min_physics_similarity=min_phys,
                                            physics_relevance=physics_relevance)
        elif at == "Temporal Analysis":
            time_col = st.session_state.last_params.get('time_column', 'year') if st.session_state.last_params else 'year'
            results = analyze_temporal_patterns(nodes_df, edges_df, time_col)
        elif at == "Correlation Analysis":
            results = analyze_failure_correlations(G_filtered)
        else:
            results = analyze_failure_centrality(G_filtered)

        st.session_state.last_analysis_results = results

        # Generate structured insights (JSON) – pass advanced metrics
        graph_stats = {'nodes': G_filtered.number_of_nodes(), 'edges': G_filtered.number_of_edges()}
        params_for_insights = st.session_state.last_params if st.session_state.last_params else {}
        # For the filtered graph, recompute advanced metrics on the filtered graph for accurate ranking
        pagerank_f = nx.pagerank(G_filtered, weight='weight')
        clustering_f = nx.clustering(G_filtered, weight='weight')
        core_f = nx.core_number(G_filtered)
        # Physics relevance for filtered nodes
        phys_rel_f = {n: physics_relevance.get(n, 0) for n in G_filtered.nodes()}

        structured = st.session_state.insight_generator.generate_structured_insights(
            results, at, params_for_insights, graph_stats, user_query,
            physics_relevance=phys_rel_f,
            pagerank=pagerank_f,
            clustering=clustering_f,
            core_numbers=core_f
        )
        st.session_state.last_structured_insights = structured

        # Display JSON (optional, for transparency)
        with st.expander("📊 Structured Numerical Insights (JSON)"):
            st.json(structured)

        # Optional: Show ranked mechanisms as a table
        if structured["ranked_mechanisms"]:
            st.subheader("🏆 Ranked Failure Mechanisms (Enhanced)")
            df_rank = pd.DataFrame(structured["ranked_mechanisms"])
            st.dataframe(df_rank[["name", "composite_weight", "degree", "physics_match", "pagerank", "clustering", "core_number", "equation"]])

        if structured["physics_weighted_pathways"]:
            st.subheader("🧪 Physics‑Weighted Pathways (sorted by total physics relevance)")
            st.dataframe(pd.DataFrame(structured["physics_weighted_pathways"]))

        # Second LLM inference (only if user wants)
        if st.checkbox("🤖 Ask LLM to summarise the numerical insights (inference only)", value=False):
            with st.spinner("LLM inferring on numerical data..."):
                answer = llm_infer_on_insights(structured, user_query,
                                               st.session_state.llm_tokenizer,
                                               st.session_state.llm_model)
                st.markdown("### 🤖 LLM Inference")
                st.write(answer)

        # Visualization
        if G_filtered.number_of_nodes() > 0:
            if colour_by == "Community":
                # community detection on filtered graph
                try:
                    partition = community_louvain.best_partition(G_filtered, weight='weight')
                    comm_map = partition
                except:
                    comm_map = {n: 0 for n in G_filtered.nodes()}
                unique_comms = sorted(set(comm_map.values()))
                color_pal = px.colors.qualitative.Set3 if len(unique_comms) <= 10 else px.colors.qualitative.Alphabet
                node_colors = [color_pal[comm_map[n] % len(color_pal)] for n in G_filtered.nodes()]
            else:  # Physics Relevance
                # colour by physics_relevance value (continuous)
                phys_vals = [G_filtered.nodes[n].get('physics_relevance', 0) for n in G_filtered.nodes()]
                # use a colour scale: low=blue, high=red
                node_colors = phys_vals  # for plotly, we'll use colorscale

            pos = nx.spring_layout(G_filtered, k=1, iterations=100, seed=42, weight='weight')
            scores = [G_filtered.nodes[n].get('priority_score',0) for n in G_filtered.nodes()]
            min_s, max_s = 15, 60
            if max(scores) > min(scores):
                sizes = [min_s + (max_s-min_s)*(s-min(scores))/(max(scores)-min(scores)) for s in scores]
            else:
                sizes = [30]*len(scores)
            edge_x, edge_y = [], []
            for u,v in G_filtered.edges():
                x0,y0 = pos[u]; x1,y1 = pos[v]
                edge_x.extend([x0,x1,None]); edge_y.extend([y0,y1,None])
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=edge_width, color='#888'), hoverinfo='none', mode='lines')
            node_x, node_y, node_text, node_labels = [],[],[],[]
            for n in G_filtered.nodes():
                x,y = pos[n]
                node_x.append(x); node_y.append(y)
                d = G_filtered.nodes[n]
                node_text.append(
                    f"{n}<br>Category: {d.get('category','N/A')}<br>Freq: {d.get('frequency',0)}<br>"
                    f"Priority: {d.get('priority_score',0):.3f}<br>Physics rel: {d.get('physics_relevance',0):.3f}<br>"
                    f"PageRank: {d.get('pagerank',0):.3f}<br>Clustering: {d.get('clustering',0):.3f}<br>"
                    f"Core: {d.get('core_number',0)}"
                )
                node_labels.append(n[:max_chars]+'...' if len(n)>max_chars else n)

            if colour_by == "Community":
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text' if show_labels else 'markers',
                    text=node_labels if show_labels else None,
                    textfont=dict(size=label_size, color='black'),
                    textposition='middle center',
                    hoverinfo='text', hovertext=node_text,
                    marker=dict(color=node_colors, size=sizes, line=dict(width=1, color='darkgray'))
                )
                # Add legend for communities
                fig = go.Figure(data=[edge_trace, node_trace])
                for i, col in enumerate(color_pal[:len(unique_comms)]):
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=col), name=f"Comm {i}", showlegend=True))
            else:
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text' if show_labels else 'markers',
                    text=node_labels if show_labels else None,
                    textfont=dict(size=label_size, color='black'),
                    textposition='middle center',
                    hoverinfo='text', hovertext=node_text,
                    marker=dict(
                        color=node_colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Physics Relevance"),
                        size=sizes,
                        line=dict(width=1, color='darkgray')
                    )
                )
                fig = go.Figure(data=[edge_trace, node_trace])

            fig.update_layout(title=f"Battery Degradation Graph - {at}", showlegend=(colour_by=="Community"), hovermode='closest',
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)

    # Export functionality
    with st.sidebar:
        st.markdown("### 💾 Export")
        if st.button("Export Filtered Graph as CSV"):
            nodes_exp = pd.DataFrame([{'node':n, **G_filtered.nodes[n]} for n in G_filtered.nodes()])
            edges_exp = pd.DataFrame([{'source':u, 'target':v, **G_filtered.edges[u,v]} for u,v in G_filtered.edges()])
            st.download_button("Download Nodes", nodes_exp.to_csv(index=False), "nodes.csv")
            st.download_button("Download Edges", edges_exp.to_csv(index=False), "edges.csv")

if __name__ == "__main__":
    main()
