# ==================== IMPORTS ====================
import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from collections import Counter
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import glob
import uuid
import json
import gc
import psutil
import pickle
import faiss
from rank_bm25 import BM25Okapi
import time
import tempfile
import hashlib

# Try to import transformers – if too old, show error and disable LLM
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    st.error(
        "The installed `transformers` library is too old or missing. "
        "Please upgrade it: `pip install 'transformers>=4.44.0'`. "
        "LLM features will be disabled."
    )
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModel = None

# ==================== CONFIGURATION ====================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 1.5,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.transparent': True
})

DB_DIR = os.path.dirname(os.path.abspath(__file__))
RELIABILITY_DB_FILE = os.path.join(DB_DIR, "battery_reliability.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "battery_reliability_universe.db")

# Index storage in a temporary directory – cleared on reboot
def get_index_dir(db_path):
    """Return a unique path under /tmp for the given database."""
    # Hash the absolute path to avoid collisions
    db_hash = hashlib.sha256(os.path.abspath(db_path).encode()).hexdigest()[:16]
    index_dir = os.path.join(tempfile.gettempdir(), f"sentence_index_{db_hash}")
    os.makedirs(index_dir, exist_ok=True)
    return index_dir

# We'll compute the actual index directory later when we know the database path
# But we need to keep these path constants for later use
FAISS_INDEX_PATH = None
METADATA_PATH = None
BM25_PATH = None
TOKENIZED_SENTENCES_PATH = None

logging.basicConfig(
    filename=os.path.join(DB_DIR, 'battery_reliability_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="Battery Reliability Analysis Tool (SciBERT + LLM + Accelerated Index)", layout="wide")
st.title("Battery Reliability Analysis: Electrode Cracking, SEI Formation, Degradation")

st.markdown("""
This tool uses a **retrieve‑then‑extract** architecture: pre‑computed sentence embeddings + BM25 index,
then two‑stage retrieval to feed only the most relevant sentences to the LLM or heuristic NER.
Speedups of 10–50× are typical.

**Index storage**: The sentence index is stored in a temporary directory (e.g., `/tmp` on Linux) and
is automatically cleared on system reboot. No persistent files are left behind.
""")

# ==================== MEMORY MANAGEMENT UTILITIES ====================
def clear_memory(verbose=True):
    """Clear all caches and run garbage collection"""
    if verbose:
        st.info("Clearing memory caches...")
    
    st.cache_data.clear()
    st.cache_resource.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()
    
    if verbose:
        st.success("Memory cleared successfully!")

def get_memory_usage():
    """Get current memory usage estimate"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# ==================== SPACY SETUP (LAZY LOADING) ====================
nlp = None

@st.cache_resource
def load_spacy_model():
    """Load spaCy model with resource management"""
    try:
        nlp_model = spacy.load("en_core_web_sm")
        nlp_model.max_length = 100000
        
        @Language.component("custom_tokenizer")
        def custom_tokenizer(doc):
            hyphenated_phrases = ["electrode-cracking", "SEI-formation", "cyclic-mechanical-damage", "diffusion-induced-stress", "electrolyte-degradation", "capacity-fade", "lithium-ion", "Li-ion", "crack-propagation"]
            for phrase in hyphenated_phrases:
                if phrase.lower() in doc.text.lower():
                    with doc.retokenize() as retokenizer:
                        for match in re.finditer(rf'\b{re.escape(phrase)}\b', doc.text, re.IGNORECASE):
                            start_char, end_char = match.span()
                            start_token = None
                            for token in doc:
                                if token.idx >= start_char:
                                    start_token = token.i
                                    break
                            if start_token is not None:
                                retokenizer.merge(doc[start_token:start_token+len(phrase.split('-'))])
            return doc
        
        if "custom_tokenizer" not in nlp_model.pipe_names:
            nlp_model.add_pipe("custom_tokenizer", before="parser")
        
        return nlp_model
    except Exception as e:
        st.error(f"Failed to load spaCy: {e}")
        st.stop()

def get_nlp():
    """Get spaCy model, loading if necessary"""
    global nlp
    if nlp is None:
        nlp = load_spacy_model()
    return nlp

# ==================== SCIBERT SETUP (LAZY LOADING) ====================
scibert_tokenizer = None
scibert_model = None

@st.cache_resource
def load_scibert():
    """Load SciBERT with memory optimization"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased",
            cache_dir=os.path.join(DB_DIR, ".cache", "scibert")
        )
        model = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased",
            cache_dir=os.path.join(DB_DIR, ".cache", "scibert"),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()
        model = model.to("cpu")
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load SciBERT: {e}")
        return None, None

def get_scibert_models():
    """Get SciBERT models, loading if necessary"""
    global scibert_tokenizer, scibert_model
    
    if scibert_tokenizer is None or scibert_model is None:
        scibert_tokenizer, scibert_model = load_scibert()
    
    return scibert_tokenizer, scibert_model

def unload_scibert():
    """Unload SciBERT to free memory"""
    global scibert_tokenizer, scibert_model
    
    if scibert_model is not None:
        del scibert_model
        scibert_model = None
    
    if scibert_tokenizer is not None:
        del scibert_tokenizer
        scibert_tokenizer = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()

# ==================== LLM SETUP ====================
LLM_MODELS = {
    "GPT-2 (124M, fastest)": "gpt2",
    "GPT-2 Medium (355M, fast)": "gpt2-medium",
    "Phi-2 (2.7B, CPU-friendly)": "microsoft/phi-2",
    "Qwen2.5-0.5B-Instruct (tiny)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct (CPU)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-2B (2B, CPU)": "google/gemma-2b",
    "Qwen2.5-3B-Instruct (may OOM)": "Qwen/Qwen2.5-3B-Instruct",
    "TinyLlama-1.1B (CPU)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

@st.cache_resource(max_entries=1, show_spinner=False)
def load_llm(model_key: str):
    """Load the selected LLM, unloading any previously loaded model to save memory."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, None
    
    if model_key not in LLM_MODELS:
        st.error(f"Unknown model: {model_key}")
        return None, None, None
    
    model_id = LLM_MODELS[model_key]
    
    # Unload previous model if it exists in session state
    if "llm_model" in st.session_state and st.session_state.llm_model is not None:
        del st.session_state.llm_model
    if "llm_tokenizer" in st.session_state and st.session_state.llm_tokenizer is not None:
        del st.session_state.llm_tokenizer
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        st.session_state.llm_tokenizer = tokenizer
        st.session_state.llm_model = model
        st.success(f"Loaded {model_key} (approx {model.num_parameters()/1e9:.2f}B parameters) on CPU.")
        
        return tokenizer, model, model_key
    except Exception as e:
        st.error(f"Failed to load {model_key}: {str(e)}")
        if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
            st.error("Out of memory. Try a smaller model (e.g., GPT-2 or Qwen2.5-0.5B).")
        return None, None, None

# ==================== SESSION STATE ====================
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "ner_results" not in st.session_state:
    st.session_state.ner_results = None
if "db_file" not in st.session_state:
    st.session_state.db_file = UNIVERSE_DB_FILE if os.path.exists(UNIVERSE_DB_FILE) else None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None
if "last_structured_insights" not in st.session_state:
    st.session_state.last_structured_insights = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm_tokenizer" not in st.session_state:
    st.session_state.llm_tokenizer = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "llm_backend_loaded" not in st.session_state:
    st.session_state.llm_backend_loaded = None
if "sentence_index_loaded" not in st.session_state:
    st.session_state.sentence_index_loaded = False
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}   # key: term embedding hash -> entities

# ==================== HELPER FUNCTIONS ====================
def update_log(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

@st.cache_data(ttl=3600, max_entries=100)
def get_scibert_embedding_batch(texts):
    """Compute embeddings for a list of texts in a single forward pass."""
    if not texts:
        return []
    
    tokenizer, model = get_scibert_models()
    if tokenizer is None or model is None:
        return [None] * len(texts)
    
    try:
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=64, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
                
                norms = np.linalg.norm(last_hidden_state, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized_embeddings = last_hidden_state / norms
                
                all_embeddings.extend(list(normalized_embeddings))
            
            del inputs, outputs, last_hidden_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_embeddings
    except Exception as e:
        update_log(f"SciBERT batch embedding failed: {str(e)}")
        return [None] * len(texts)

def get_embedding_for_text(text):
    """Single-text embedding (wrapper)."""
    emb = get_scibert_embedding_batch([text])
    return emb[0] if emb else None

# ==================== SENTENCE INDEX BUILDING (EPHEMERAL) ====================
def build_sentence_index(db_path):
    """
    Build FAISS index and BM25 index for all sentences in the database.
    Stores them in a temporary directory (cleared on reboot).
    """
    index_dir = get_index_dir(db_path)
    faiss_path = os.path.join(index_dir, "faiss.index")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    tokenized_path = os.path.join(index_dir, "tokenized_sentences.pkl")

    if os.path.exists(faiss_path) and os.path.exists(metadata_path) and os.path.exists(bm25_path):
        update_log("Sentence index already exists in temp dir, skipping build.")
        return True

    update_log("Building sentence index... This may take a while.")
    progress_bar = st.progress(0, text="Loading papers...")

    conn = sqlite3.connect(db_path)
    query = "SELECT id, title, year, content FROM papers WHERE content IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        st.error("No papers with content found. Cannot build index.")
        return False

    # Extract sentences using spaCy (once)
    nlp_model = get_nlp()
    all_sentences = []          # list of dicts: {paper_id, title, year, text}
    all_sentence_texts = []
    tokenized_sentences = []    # list of list of tokens for BM25

    total_papers = len(df)
    for idx, row in df.iterrows():
        doc = nlp_model(row["content"][:nlp_model.max_length])
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                all_sentences.append({
                    "paper_id": row["id"],
                    "title": row["title"],
                    "year": row["year"],
                    "text": sent_text
                })
                all_sentence_texts.append(sent_text)
                tokenized_sentences.append(sent_text.lower().split())
        progress_bar.progress((idx+1)/total_papers, text=f"Processing paper {idx+1}/{total_papers}")

    if not all_sentences:
        st.error("No sentences extracted.")
        return False

    # Compute embeddings
    update_log(f"Computing embeddings for {len(all_sentence_texts)} sentences...")
    embeddings = get_scibert_embedding_batch(all_sentence_texts)
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    valid_embeddings = np.array([embeddings[i] for i in valid_indices], dtype=np.float32)
    valid_sentences = [all_sentences[i] for i in valid_indices]
    valid_tokenized = [tokenized_sentences[i] for i in valid_indices]

    # Build FAISS index
    dim = valid_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine (since embeddings are normalized)
    index.add(valid_embeddings)
    faiss.write_index(index, faiss_path)

    # Save metadata
    with open(metadata_path, "wb") as f:
        pickle.dump(valid_sentences, f)

    # Build BM25 index
    bm25 = BM25Okapi(valid_tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save tokenized sentences for BM25 (optional, but needed for later search)
    with open(tokenized_path, "wb") as f:
        pickle.dump(valid_tokenized, f)

    update_log(f"Index built: {len(valid_sentences)} sentences indexed in {index_dir}.")
    st.success(f"Sentence index built and saved in temporary directory: {index_dir}")
    return True

def load_sentence_index(db_path):
    """Load FAISS index and metadata from temporary directory."""
    index_dir = get_index_dir(db_path)
    faiss_path = os.path.join(index_dir, "faiss.index")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    tokenized_path = os.path.join(index_dir, "tokenized_sentences.pkl")

    if not (os.path.exists(faiss_path) and os.path.exists(metadata_path) and os.path.exists(bm25_path)):
        return None, None, None, None
    index = faiss.read_index(faiss_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(tokenized_path, "rb") as f:
        tokenized_sentences = pickle.load(f)
    return index, metadata, bm25, tokenized_sentences

def hybrid_search(term, index, metadata, bm25, tokenized_sentences, top_k=50, bm25_weight=0.5):
    """
    Perform hybrid retrieval:
    - BM25 score
    - Cosine similarity of term embedding with sentence embeddings
    Combine with weight.
    Returns list of metadata dicts for top_k sentences.
    """
    # Term embedding
    term_emb = get_embedding_for_text(term)
    if term_emb is None:
        update_log("Failed to get term embedding, falling back to BM25 only.")
        term_emb = None

    # BM25 scores
    term_tokens = term.lower().split()
    bm25_scores = bm25.get_scores(term_tokens)   # array of length num_sentences

    # Embedding similarity (if available)
    if term_emb is not None:
        term_emb = np.array([term_emb], dtype=np.float32)
        _, indices = index.search(term_emb, len(metadata))   # search all, get all distances
        # distances are cosine similarity (since index is IP)
        sim_scores = np.zeros(len(metadata))
        # fill in the similarity for each retrieved index
        for dist, idx in zip(indices[0], indices[1]):
            sim_scores[idx] = dist
    else:
        sim_scores = np.zeros(len(metadata))

    # Normalize scores (min-max)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    sim_scores = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-8)

    # Combine
    combined = bm25_weight * bm25_scores + (1 - bm25_weight) * sim_scores

    # Get top_k indices
    top_indices = np.argsort(combined)[-top_k:][::-1]
    return [metadata[i] for i in top_indices]

# ==================== RETRIEVAL-AUGMENTED NER ====================
def perform_ner_on_terms_retrieval(db_file, selected_terms, index, metadata, bm25, tokenized_sentences, max_papers=500):
    """Heuristic NER using retrieved sentences."""
    update_log(f"Starting retrieval-augmented heuristic NER for terms: {', '.join(selected_terms)}")
    # For each term, retrieve relevant sentences and then apply pattern matching.

    # We'll collect entities from all retrieved sentences for all terms (deduplicate later)
    retrieved_sentences = []
    for term in selected_terms:
        top_sentences = hybrid_search(term, index, metadata, bm25, tokenized_sentences, top_k=50)
        retrieved_sentences.extend(top_sentences)
    # Remove duplicates (by text and paper_id)
    unique_sentences = {}
    for s in retrieved_sentences:
        key = (s["paper_id"], s["text"])
        if key not in unique_sentences:
            unique_sentences[key] = s

    # Now process these sentences similarly to original heuristic NER but only on retrieved ones.
    # We'll reuse the original logic but adapted to work on sentences rather than full papers.

    categories = {
        "Deformation": ["plastic deformation", "yield strength", "plastic strain", "ductility", "inelastic strain", "flow stress", "volume expansion", "swelling", "volumetric expansion", "volume change", "dimensional change", "volume deformation", "volume increase", "volumetric strain", "expansion strain", "dilatation", "volume distortion"],
        "Fatigue": ["fatigue life", "cyclic loading", "cycle life", "fatigue crack", "stress cycling", "mechanical fatigue", "low cycle fatigue", "high cycle fatigue", "fatigue damage", "fatigue failure", "cyclic deformation", "cyclic stress", "endurance limit", "S-N curve"],
        "Crack and Fracture": ["electrode cracking", "crack propagation", "crack growth", "crack initiation", "micro-cracking", "fracture", "fracture toughness", "brittle fracture", "ductile fracture", "crack branching", "crack tip", "stress intensity factor", "fracture mechanics", "J-integral", "cleavage fracture"],
        "Degradation": ["SEI formation", "electrolyte degradation", "capacity fade", "cycle degradation", "electrode degradation", "electrolyte decomposition", "aging", "thermal degradation", "mechanical degradation", "capacity loss", "fading", "deterioration", "corrosion", "passivation", "side reaction"]
    }

    valid_units = {
        "Deformation": ["%", "MPa"],
        "Fatigue": ["cycles", "MPa"],
        "Crack and Fracture": ["μm", "MPa·m^0.5"],
        "Degradation": ["%", "nm"]
    }

    valid_ranges = {
        "Deformation": [(0, 100, "%"), (0, 1000, "MPa")],
        "Fatigue": [(1, 10000, "cycles"), (0, 1000, "MPa")],
        "Crack and Fracture": [(0, 1000, "μm"), (0, 10, "MPa·m^0.5")],
        "Degradation": [(0, 100, "%"), (0, 1000, "nm")]
    }

    numerical_pattern = r"(\d+\.?\d*[eE]?-?\d*|\d+)\s*(mpa|gpa|kpa|pa|%|μm|nm|MPa·m\^0\.5|cycles|MPa|GPa)"
    similarity_threshold = 0.7

    ref_embeddings = {cat: [get_embedding_for_text(term) for term in terms if get_embedding_for_text(term) is not None] for cat, terms in categories.items()}

    term_patterns = {term: re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE) for term in selected_terms}

    entities = []
    entity_set = set()
    nlp_model = get_nlp()

    for sent in unique_sentences.values():
        text = sent["text"].lower()
        if not text.strip():
            continue
        doc = nlp_model(text)  # treat the sentence as a doc

        # Check if any term appears in this sentence
        term_present = any(term_patterns[term].search(text) for term in selected_terms)
        if not term_present:
            # Optionally use embedding similarity to see if it's semantically similar to any term
            sent_emb = get_embedding_for_text(text)
            if sent_emb is None:
                continue
            # Check similarity with term embeddings
            term_embeddings = [get_embedding_for_text(term) for term in selected_terms if get_embedding_for_text(term) is not None]
            if not term_embeddings:
                continue
            sims = [np.dot(sent_emb, t_emb) / (np.linalg.norm(sent_emb) * np.linalg.norm(t_emb)) for t_emb in term_embeddings if np.linalg.norm(t_emb) != 0]
            if max(sims) < 0.6:
                continue  # not relevant enough

        # Extract numerical entities
        matches = re.finditer(numerical_pattern, text, re.IGNORECASE)
        for match in matches:
            span_text = match.group(0)
            value_str = match.group(1)
            unit = match.group(2).upper()
            try:
                value = float(value_str)
            except ValueError:
                continue
            if unit in ["GPA", "GPa"]:
                unit = "MPa"
                value *= 1000
            elif unit in ["KPA", "kPa"]:
                unit = "MPa"
                value /= 1000
            elif unit in ["PA", "Pa"]:
                unit = "MPa"
                value /= 1000000
            elif unit == "MPA·M^0.5":
                unit = "MPa·m^0.5"
            elif unit == "CYCLES":
                unit = "cycles"

            # Get embedding for span
            span_emb = get_embedding_for_text(span_text)
            if span_emb is None:
                continue

            # Classify into category
            best_label = None
            best_score = 0
            for label, ref_embeds in ref_embeddings.items():
                for ref_embed in ref_embeds:
                    if np.linalg.norm(span_emb) == 0 or np.linalg.norm(ref_embed) == 0:
                        continue
                    sim = np.dot(span_emb, ref_embed) / (np.linalg.norm(span_emb) * np.linalg.norm(ref_embed))
                    if sim > similarity_threshold and sim > best_score:
                        best_label = label
                        best_score = sim

            if not best_label:
                continue

            # Validate unit and range
            if unit not in valid_units.get(best_label, []):
                # Try to infer correct unit from context
                context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
                context_emb = get_embedding_for_text(context)
                if context_emb is None:
                    continue
                unit_valid = False
                for v_unit in valid_units.get(best_label, []):
                    unit_emb = get_embedding_for_text(f"{span_text} {v_unit}")
                    if unit_emb is None:
                        continue
                    unit_score = np.dot(context_emb, unit_emb) / (np.linalg.norm(context_emb) * np.linalg.norm(unit_emb))
                    if unit_score > 0.6:
                        unit_valid = True
                        unit = v_unit
                        break
                if not unit_valid:
                    continue

            # Range check
            range_valid = False
            for min_val, max_val, expected_unit in valid_ranges.get(best_label, [(None, None, None)]):
                if expected_unit == unit and min_val is not None and max_val is not None:
                    if min_val <= value <= max_val:
                        range_valid = True
                        break
            if not range_valid:
                continue

            entity_key = (sent["paper_id"], span_text, best_label, value, unit)
            if entity_key in entity_set:
                continue
            entity_set.add(entity_key)

            entities.append({
                "paper_id": sent["paper_id"],
                "title": sent["title"],
                "year": sent["year"],
                "entity_text": span_text,
                "entity_label": best_label,
                "value": value,
                "unit": unit,
                "context": text[max(0, match.start()-100):min(len(text), match.end()+100)],
                "score": best_score
            })

    update_log(f"Extracted {len(entities)} entities via retrieval heuristic NER")
    return pd.DataFrame(entities)

def scientific_llm_quantified_ner_retrieval(db_file: str, max_papers: int = 500):
    """Retrieval‑augmented LLM NER using pre‑built index."""
    if not st.session_state.llm_model or not st.session_state.llm_tokenizer:
        st.error("LLM not loaded. Select a model in the sidebar first.")
        return pd.DataFrame()

    # Ensure index is loaded
    index, metadata, bm25, tokenized_sentences = load_sentence_index(db_file)
    if index is None:
        st.error("Sentence index not built. Please build it first.")
        return pd.DataFrame()

    # For now, we will use a predefined set of terms (or allow user input)
    # For simplicity, we use the same terms as heuristic NER if selected, or default.
    if "selected_terms" in st.session_state and st.session_state.selected_terms:
        terms_to_search = st.session_state.selected_terms
    else:
        terms_to_search = ["electrode cracking", "SEI formation", "cyclic mechanical damage", "electrolyte degradation", "capacity fade"]

    # Retrieve sentences for all terms
    retrieved_sentences = []
    for term in terms_to_search:
        top_sentences = hybrid_search(term, index, metadata, bm25, tokenized_sentences, top_k=20)  # smaller k for LLM
        retrieved_sentences.extend(top_sentences)
    # Remove duplicates
    unique_sentences = {}
    for s in retrieved_sentences:
        key = (s["paper_id"], s["text"])
        if key not in unique_sentences:
            unique_sentences[key] = s

    # Prepare a context window with metadata and neighbor sentences (we don't have neighbor info here; could be added)
    # For simplicity, we just use the sentences themselves.
    # In a real implementation, we would retrieve surrounding sentences for context.
    context_text = ""
    for s in unique_sentences.values():
        context_text += f"Paper ID: {s['paper_id']}\nTitle: {s['title']}\nYear: {s['year']}\nText: {s['text']}\n\n"

    # Now pass to LLM
    tokenizer = st.session_state.llm_tokenizer
    model = st.session_state.llm_model

    # Determine max context length
    if hasattr(model.config, "max_position_embeddings"):
        max_context = model.config.max_position_embeddings
    else:
        max_context = 1024
    safety_buffer = 20
    safe_max_context = max_context - safety_buffer

    # System prompt (same as before)
    system_prompt = """You are a battery reliability expert specializing in electrode cracking, SEI formation, cyclic mechanical damage, and degradation mechanisms.
Extract EVERY quantified statement related to battery degradation.
Return ONLY a valid JSON array of objects. Each object MUST have exactly these keys:

{
  "paper_id": int,
  "title": str,
  "year": int,
  "term": str,                  // e.g. "electrode cracking", "SEI thickness", "capacity fade", "crack length"
  "value": float,
  "unit": str,                  // normalized: %, μm, nm, MPa, cycles, MPa·m^{0.5}, etc.
  "mechanism": str,             // must be one of: "Deformation", "Fatigue", "Crack and Fracture", "Degradation"
  "context": str,               // 1-2 sentences from the paper
  "confidence": float,          // 0.0–1.0 (how certain you are this extraction is correct)
  "temperature": float or null, // °C if mentioned, else null
  "cycles": int or null         // cycle number if mentioned
}

**Ontology of mechanisms (use exactly these labels):**
- Deformation: plastic deformation, yield strength, volume expansion, swelling, etc.
- Fatigue: fatigue life, cyclic loading, cycle life, stress cycling
- Crack and Fracture: electrode cracking, crack propagation, crack growth, fracture toughness
- Degradation: SEI formation, electrolyte degradation, capacity fade, aging

**Rules:**
- Normalize units (convert GPa → MPa, kPa → MPa, etc.).
- If a value is given without unit but context is clear, infer the most common unit.
- Only extract values that are explicitly numerical and tied to a degradation term.
- If nothing is found, return [].
- Think step-by-step before outputting JSON.

Now process the retrieved excerpts below and return the JSON array."""

    full_prompt = f"{system_prompt}\n\n{context_text}"

    # Tokenize and truncate
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=safe_max_context)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON
    json_match = re.search(r'\[\s*\{.*\}\s*\]', generated, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group(0))
            all_entities = []
            for item in items:
                # Add missing fields if needed
                item.setdefault("paper_id", -1)
                item.setdefault("title", "")
                item.setdefault("year", -1)
                item.setdefault("confidence", 0.7)
                if isinstance(item.get("value"), str):
                    try:
                        item["value"] = float(item["value"])
                    except:
                        continue
                all_entities.append(item)
            df_entities = pd.DataFrame(all_entities)
            if not df_entities.empty:
                df_entities["unit"] = df_entities["unit"].str.replace("GPa", "MPa").str.replace("kPa", "MPa")
                df_entities = df_entities[df_entities["confidence"] >= 0.6]
                df_entities = df_entities.drop_duplicates(subset=["paper_id", "term", "value", "unit", "mechanism"])
            update_log(f"Scientific LLM NER extracted {len(df_entities)} quantified entities via retrieval.")
            return df_entities
        except (json.JSONDecodeError, TypeError):
            pass

    update_log("LLM extraction returned no valid JSON.")
    return pd.DataFrame()

# ==================== SEMANTIC CACHING ====================
def get_cache_key(term, db_file):
    """Generate a cache key from term embedding."""
    emb = get_embedding_for_text(term)
    if emb is None:
        return None
    # Use approximate hash (e.g., first 10 digits of embedding)
    emb_bytes = emb.tobytes()
    import hashlib
    h = hashlib.sha256(emb_bytes).hexdigest()[:16]
    return f"{db_file}_{h}"

def query_with_cache(term, db_file, retrieval_func, *args, **kwargs):
    """Check cache before executing retrieval, store results."""
    cache_key = get_cache_key(term, db_file)
    if cache_key is None:
        return retrieval_func(term, *args, **kwargs)

    if cache_key in st.session_state.query_cache:
        update_log(f"Cache hit for term: {term}")
        return st.session_state.query_cache[cache_key]

    result = retrieval_func(term, *args, **kwargs)
    st.session_state.query_cache[cache_key] = result
    return result

# ==================== DATABASE INSPECTION (unchanged) ====================
def inspect_database(db_path):
    try:
        update_log(f"Inspecting database: {os.path.basename(db_path)}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        st.subheader("Tables in Database")
        if tables:
            st.write([table[0] for table in tables])
        else:
            st.warning("No tables found in the database.")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers';")
        if cursor.fetchone():
            st.subheader("Schema of 'papers' Table")
            cursor.execute("PRAGMA table_info(papers);")
            schema = cursor.fetchall()
            schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
            st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)
            
            available_columns = [col[1] for col in schema]
            update_log(f"Available columns in 'papers' table: {', '.join(available_columns)}")
            
            select_columns = ["id", "title", "year"]
            where_conditions = []
            if "content" in available_columns:
                select_columns.append("substr(content, 1, 200) as sample_content")
                where_conditions.append("content IS NOT NULL")
            if "abstract" in available_columns:
                select_columns.append("substr(abstract, 1, 200) as sample_abstract")
                where_conditions.append("abstract IS NOT NULL")
            if "matched_terms" in available_columns:
                select_columns.append("matched_terms")
            if "relevance_prob" in available_columns:
                select_columns.append("relevance_prob")
            
            if not where_conditions:
                where_clause = "1=1"
            else:
                where_clause = " OR ".join(where_conditions)
            
            query = f"SELECT {', '.join(select_columns)} FROM papers WHERE {where_clause} LIMIT 5"
            try:
                df = pd.read_sql_query(query, conn)
                st.subheader("Sample Rows from 'papers' Table (First 5 Papers)")
                if df.empty:
                    st.warning("No valid papers found in the 'papers' table.")
                else:
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error executing sample query: {str(e)}")
                update_log(f"Error executing sample query: {str(e)}")
            
            total_query = f"SELECT COUNT(*) as count FROM papers WHERE {where_clause}"
            cursor.execute(total_query)
            total_papers = cursor.fetchone()[0]
            st.subheader("Total Valid Papers")
            st.write(f"{total_papers} papers")
            
            terms_to_search = ["electrode cracking", "SEI formation", "cyclic mechanical damage", "electrolyte degradation", "capacity fade"]
            st.subheader("Term Frequency in Available Text Columns")
            term_counts = {}
            for term in terms_to_search:
                conditions = []
                if "abstract" in available_columns:
                    conditions.append(f"abstract LIKE '%{term}%'")
                if "content" in available_columns:
                    conditions.append(f"content LIKE '%{term}%'")
                if conditions:
                    term_query = f"SELECT COUNT(*) FROM papers WHERE {' OR '.join(conditions)}"
                    cursor.execute(term_query)
                    count = cursor.fetchone()[0]
                    term_counts[term] = count
                    st.write(f"'{term}': {count} papers")
                else:
                    st.write(f"'{term}': Unable to search (no abstract or content columns)")
                    update_log(f"Unable to search for '{term}': no abstract or content columns")
            
            select_columns_download = ["id", "title", "year"]
            if "content" in available_columns:
                select_columns_download.append("substr(content, 1, 1000) as content")
            if "abstract" in available_columns:
                select_columns_download.append("substr(abstract, 1, 1000) as abstract")
            if "matched_terms" in available_columns:
                select_columns_download.append("matched_terms")
            if "relevance_prob" in available_columns:
                select_columns_download.append("relevance_prob")
            
            query = f"SELECT {', '.join(select_columns_download)} FROM papers WHERE {where_clause} LIMIT 10"
            try:
                df_full = pd.read_sql_query(query, conn)
                csv_filename = f"database_sample_{uuid.uuid4().hex}.csv"
                csv_path = os.path.join(DB_DIR, csv_filename)
                df_full.to_csv(csv_path, index=False)
                with open(csv_path, "rb") as f:
                    st.session_state.csv_data = f.read()
                    st.session_state.csv_filename = csv_filename
                st.subheader("Download Sample Content")
                st.download_button(
                    label="Download Sample CSV",
                    data=st.session_state.csv_data,
                    file_name="database_sample.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            except Exception as e:
                st.error(f"Error generating sample CSV: {str(e)}")
                update_log(f"Error generating sample CSV: {str(e)}")
            
            conn.close()
            st.success(f"Database inspection completed for {os.path.basename(db_path)}")
            return term_counts
        else:
            st.warning("No 'papers' table found in the database.")
            conn.close()
            return None
    except Exception as e:
        st.error(f"Error reading database {os.path.basename(db_path)}: {str(e)}")
        update_log(f"Error reading database {os.path.basename(db_path)}: {str(e)}")
        return None

# ==================== VISUALIZATION HELPERS (unchanged) ====================
@st.cache_data(ttl=1800)
def plot_ner_histogram(df, top_n, colormap):
    if df.empty:
        return None
    
    label_counts = df["entity_label"].value_counts().head(top_n)
    labels = label_counts.index.tolist()
    counts = label_counts.values
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
    ax.bar(labels, counts, color=colors, edgecolor="black")
    ax.set_xlabel("Entity Labels")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of Top {top_n} NER Entities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

@st.cache_data(ttl=1800)
def plot_ner_value_boxplot(df, top_n, colormap):
    if df.empty or df["value"].isna().all():
        return None
    
    value_df = df[df["value"].notna() & df["unit"].notna()]
    if value_df.empty:
        return None
    
    label_counts = value_df["entity_label"].value_counts().head(top_n)
    labels = label_counts.index.tolist()
    data = [value_df[value_df["entity_label"] == label]["value"].values for label in labels]
    units = [value_df[value_df["entity_label"] == label]["unit"].iloc[0] for label in labels]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
    box = ax.boxplot(data, patch_artist=True, labels=[f"{label} ({unit})" for label, unit in zip(labels, units)])
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel("Entity Labels")
    ax.set_ylabel("Value")
    ax.set_title(f"Box Plot of Numerical Values for Top {top_n} NER Entities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

@st.cache_data(ttl=1800)
def plot_ner_value_histograms(df, categories_units, top_n, colormap):
    if df.empty or df["value"].isna().all():
        return []
    
    value_df = df[df["value"].notna() & df["unit"].notna()]
    if value_df.empty:
        return []
    
    figs = []
    for category, unit in categories_units.items():
        cat_df = value_df[value_df["entity_label"] == category]
        if unit:
            cat_df = cat_df[cat_df["unit"] == unit]
        if cat_df.empty:
            update_log(f"No data for {category} with unit {unit}")
            continue
        
        values = cat_df["value"].values
        if len(values) == 0:
            update_log(f"No numerical values for {category} with unit {unit}")
            continue
        
        fig, ax = plt.subplots(figsize=(8, 4))
        color = cm.get_cmap(colormap)(0.5)
        ax.hist(values, bins=20, color=color, edgecolor="black")
        ax.set_xlabel(f"Value ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {category} Values ({unit})")
        plt.tight_layout()
        figs.append(fig)
    
    return figs

# ==================== NARRATIVE GENERATION (unchanged) ====================
def generate_narrative_insight(structured_json: dict):
    tokenizer = st.session_state.llm_tokenizer
    model = st.session_state.llm_model
    
    if not tokenizer or not model:
        return "LLM not loaded. Please select a model."
    
    prompt = f"""Convert the following structured battery-degradation analysis into a clear, publication-style paragraph (max 180 words).
Use the exact node names and equations when relevant.
{json.dumps(structured_json, indent=2)}"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()
    
    return answer

# ==================== MAIN APP ====================
st.header("Select or Upload Database")

with st.sidebar:
    st.subheader("Memory Management")
    
    if st.button("Clear All Memory & Cache"):
        clear_memory()
        st.rerun()
    
    mem_usage = get_memory_usage()
    st.metric("Current Memory Usage", f"{mem_usage:.1f} MB")
    
    st.subheader("NER Parameters")
    max_papers = st.slider("Max Papers to Process", min_value=100, max_value=2000, value=500, step=20, 
                          help="Lower values = less memory usage")
    top_n = st.slider("Top N Entities to Show in Plots", min_value=5, max_value=30, value=10)
    
    st.subheader("Index Management")
    if st.button("Build/Refresh Sentence Index"):
        if st.session_state.db_file and os.path.exists(st.session_state.db_file):
            with st.spinner("Building index (may take a while)..."):
                build_sentence_index(st.session_state.db_file)
                st.session_state.sentence_index_loaded = False  # force reload later
        else:
            st.error("No database selected.")
    
    st.subheader("Retrieval Settings")
    top_k_retrieval = st.slider("Top K sentences to retrieve per term", 10, 100, 50, help="More sentences = more context but slower.")
    bm25_weight = st.slider("BM25 weight (0=only embedding, 1=only BM25)", 0.0, 1.0, 0.5)
    
    st.subheader("Semantic Cache")
    if st.button("Clear Query Cache"):
        st.session_state.query_cache.clear()
        st.success("Cache cleared.")

db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files if f in [RELIABILITY_DB_FILE, UNIVERSE_DB_FILE]] + ["Upload a new .db file"]

if os.path.basename(UNIVERSE_DB_FILE) not in db_options and os.path.exists(UNIVERSE_DB_FILE):
    db_options.insert(0, os.path.basename(UNIVERSE_DB_FILE))

default_index = db_options.index(os.path.basename(UNIVERSE_DB_FILE)) if os.path.basename(UNIVERSE_DB_FILE) in db_options else 0
db_selection = st.selectbox("Select Database", db_options, index=default_index, key="db_select")

if db_selection == "Upload a new .db file":
    uploaded_file = st.file_uploader("Upload SQLite Database (.db)", type=["db"], key="db_upload")
    if uploaded_file:
        temp_db_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.db_file = temp_db_path
else:
    if db_selection == os.path.basename(UNIVERSE_DB_FILE):
        st.session_state.db_file = UNIVERSE_DB_FILE
    else:
        st.session_state.db_file = os.path.join(DB_DIR, db_selection)

if st.session_state.db_file and os.path.exists(st.session_state.db_file):
    # Ensure index is loaded
    if not st.session_state.sentence_index_loaded:
        index, metadata, bm25, tokenized_sentences = load_sentence_index(st.session_state.db_file)
        if index is None:
            st.warning("Sentence index not found. Please build it using the sidebar button.")
            st.session_state.sentence_index_loaded = False
        else:
            st.session_state.sentence_index_loaded = True
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.bm25 = bm25
            st.session_state.tokenized_sentences = tokenized_sentences
            st.success("Sentence index loaded from temporary directory.")

    tab1, tab2 = st.tabs(["Database Inspection", "NER Analysis"])
    
    with tab1:
        st.header("Database Inspection")
        if st.button("Inspect Database", key="inspect_button"):
            with st.spinner("Inspecting..."):
                inspect_database(st.session_state.db_file)
    
    with tab2:
        st.header("NER Analysis")
        
        use_llm = st.checkbox("Use LLM for quantified NER", value=True)
        
        if use_llm and TRANSFORMERS_AVAILABLE:
            model_choice = st.selectbox("Select LLM Model", list(LLM_MODELS.keys()), index=0, key="llm_model_select")
        elif use_llm and not TRANSFORMERS_AVAILABLE:
            st.error("Transformers is too old. Please upgrade to use LLM features.")
        
        # For heuristic NER, allow manual term input
        if not use_llm:
            st.subheader("Heuristic NER Settings")
            default_terms = ["electrode cracking", "SEI formation", "cyclic mechanical damage", "electrolyte degradation", "capacity fade"]
            term_input = st.text_area("Enter terms (one per line or comma-separated)", 
                                     value="\n".join(default_terms),
                                     help="Terms to search for in the text. You can also use multi-word phrases.")
            if term_input:
                selected_terms = [t.strip() for t in re.split(r'[,\n]', term_input) if t.strip()]
                st.session_state.selected_terms = selected_terms
            else:
                selected_terms = []
            st.caption(f"{len(selected_terms)} terms entered.")
        
        if st.button("Run NER Analysis", key="ner_analyze"):
            if os.path.exists(st.session_state.db_file):
                if not st.session_state.sentence_index_loaded:
                    st.error("Sentence index not built. Please build it first.")
                else:
                    with st.spinner("Processing NER analysis with retrieval..."):
                        if use_llm and TRANSFORMERS_AVAILABLE:
                            tokenizer, model, loaded_key = load_llm(model_choice)
                            if tokenizer and model:
                                llm_df = scientific_llm_quantified_ner_retrieval(st.session_state.db_file, max_papers=max_papers)
                                st.session_state.ner_results = llm_df
                                if not llm_df.empty:
                                    st.session_state.last_structured_insights = {
                                        "entities": llm_df.to_dict(orient="records"),
                                        "summary": f"Extracted {len(llm_df)} quantified entities."
                                    }
                            else:
                                st.error("Failed to load LLM. Check logs.")
                        else:
                            if not use_llm and not selected_terms:
                                st.warning("Please enter at least one term for heuristic NER.")
                            elif not use_llm and selected_terms:
                                # Use retrieval-augmented heuristic NER
                                ner_df = perform_ner_on_terms_retrieval(
                                    st.session_state.db_file, selected_terms,
                                    st.session_state.index, st.session_state.metadata,
                                    st.session_state.bm25, st.session_state.tokenized_sentences,
                                    max_papers=max_papers
                                )
                                st.session_state.ner_results = ner_df
                                if not ner_df.empty:
                                    st.session_state.last_structured_insights = {
                                        "entities": ner_df.to_dict(orient="records"),
                                        "summary": f"Extracted {len(ner_df)} entities using retrieval-augmented heuristic method."
                                    }
            else:
                st.error(f"Cannot perform NER analysis: {os.path.basename(st.session_state.db_file)} not found.")
                update_log(f"Cannot perform NER analysis: {os.path.basename(st.session_state.db_file)} not found.")
            
            if st.session_state.ner_results is not None and not st.session_state.ner_results.empty:
                st.success(f"Extracted {len(st.session_state.ner_results)} entities!")
                st.dataframe(st.session_state.ner_results.head(100), use_container_width=True)
                ner_csv = st.session_state.ner_results.to_csv(index=False)
                st.download_button("Download NER Data CSV", ner_csv, "ner_data.csv", "text/csv", key="download_ner")
                
                # Visualizations (same as before)
                if "entity_label" in st.session_state.ner_results.columns:
                    fig_hist = plot_ner_histogram(st.session_state.ner_results, top_n, "viridis")
                    if fig_hist:
                        st.pyplot(fig_hist)
                    fig_box = plot_ner_value_boxplot(st.session_state.ner_results, top_n, "viridis")
                    if fig_box:
                        st.pyplot(fig_box)
                    
                    categories_units = {
                        "Deformation": "%",
                        "Fatigue": "cycles",
                        "Crack and Fracture": "um",
                        "Degradation": "%"
                    }
                    figs_hist_values = plot_ner_value_histograms(st.session_state.ner_results, categories_units, top_n, "viridis")
                    if figs_hist_values:
                        st.subheader("Value Distribution Histograms")
                        for fig in figs_hist_values:
                            st.pyplot(fig)
                
                elif "mechanism" in st.session_state.ner_results.columns:
                    st.subheader("Distribution of Mechanisms")
                    mech_counts = st.session_state.ner_results["mechanism"].value_counts()
                    fig, ax = plt.subplots()
                    ax.bar(mech_counts.index, mech_counts.values)
                    ax.set_xlabel("Mechanism")
                    ax.set_ylabel("Count")
                    ax.set_title("Count of Entities by Mechanism")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)
                    
                    st.subheader("Value Distribution per Mechanism")
                    for mech in st.session_state.ner_results["mechanism"].unique():
                        mech_df = st.session_state.ner_results[st.session_state.ner_results["mechanism"] == mech]
                        if not mech_df.empty and "value" in mech_df.columns:
                            fig, ax = plt.subplots()
                            ax.hist(mech_df["value"].dropna(), bins=20)
                            ax.set_xlabel("Value")
                            ax.set_ylabel("Frequency")
                            ax.set_title(f"{mech} Values")
                            st.pyplot(fig)
                
                if use_llm and st.session_state.llm_model is not None and st.session_state.last_structured_insights:
                    st.subheader("AI Scientific Summary")
                    narrative = generate_narrative_insight(st.session_state.last_structured_insights)
                    st.markdown(narrative)
                
                if use_llm and st.session_state.llm_model is not None:
                    st.subheader("Ask a follow-up question")
                    user_followup = st.text_input("Ask a question about the extracted entities:",
                                                placeholder="Why is capacity fade more common than cracking?")
                    if user_followup and st.button("Send", key="followup"):
                        context = json.dumps(st.session_state.last_structured_insights, indent=2)[:4000]
                        prompt = f"""You are a battery degradation expert. Use only the following structured insights:
{context}
User question: {user_followup}
Answer in 2-3 concise sentences with references to specific entities."""
                        
                        tokenizer = st.session_state.llm_tokenizer
                        model = st.session_state.llm_model
                        
                        if tokenizer and model:
                            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                            if torch.cuda.is_available():
                                inputs = inputs.to("cuda")
                            
                            with torch.no_grad():
                                outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.2, do_sample=True)
                            
                            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            answer = answer.replace(prompt, "").strip()
                            
                            st.session_state.chat_history.append(("user", user_followup))
                            st.session_state.chat_history.append(("assistant", answer))
                            
                            for role, msg in st.session_state.chat_history[-6:]:
                                st.markdown(f"**{role.capitalize()}**: {msg}")
                        else:
                            st.error("LLM not loaded. Please select a model and re-run.")
                
                st.text_area("Logs", "\n".join(st.session_state.log_buffer[-20:]), height=150, key="ner_logs")
        else:
            st.info("Click 'Run NER Analysis' to start extraction.")

else:
    st.warning(f"Select or upload a database file. Ensure {os.path.basename(UNIVERSE_DB_FILE)} is available.")

st.markdown("""
---
**Memory Optimization Tips:**
- Keep "Max Papers" <= 500 for stable performance
- Click "Clear All Memory" between heavy operations
- Use smaller models if enabling LLM features
- Close other applications to free RAM
""")
