# ==================== IMPORTS ====================
import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from collections import Counter, defaultdict
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import networkx as nx
from wordcloud import WordCloud
from nltk import ngrams
from itertools import chain, combinations
import glob
import uuid
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain
import json
import gc
import psutil
import weakref

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

logging.basicConfig(
    filename=os.path.join(DB_DIR, 'battery_reliability_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="Battery Reliability Analysis Tool (SciBERT + LLM)", layout="wide")
st.title("Battery Reliability Analysis: Electrode Cracking, SEI Formation, Degradation")

st.markdown("""
This tool inspects SQLite databases, categorizes terms related to battery reliability challenges, builds a knowledge graph, and performs NER analysis using SciBERT and optional LLM‑based quantified extraction.
Select a database, then use the tabs to inspect, categorize, visualise relationships, or extract entities with numerical values.
""")

# ==================== MEMORY MANAGEMENT UTILITIES ====================
def clear_memory(verbose=True):
    """Clear all caches and run garbage collection"""
    if verbose:
        st.info("Clearing memory caches...")
    
    # Clear Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
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
if "categorized_terms" not in st.session_state:
    st.session_state.categorized_terms = None
if "db_file" not in st.session_state:
    st.session_state.db_file = UNIVERSE_DB_FILE if os.path.exists(UNIVERSE_DB_FILE) else None
if "term_counts" not in st.session_state:
    st.session_state.term_counts = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None
if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = None
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

# ==================== DATABASE INSPECTION ====================
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

# ==================== OPTIMIZED TERM CATEGORIZATION ====================
@st.cache_data(ttl=3600, max_entries=10)
def categorize_terms(db_file, similarity_threshold=0.7, min_freq=5, max_papers=500):
    """Memory-optimized term categorization with chunked processing"""
    global nlp
    
    try:
        update_log(f"Starting term categorization (max {max_papers} papers)")
        
        if nlp is None:
            nlp = load_spacy_model()
        
        tokenizer, model = get_scibert_models()
        if tokenizer is None:
            st.error("SciBERT not available")
            return {}, Counter()
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        if max_papers:
            query += f" LIMIT {max_papers}"
        
        categories = {
            "Deformation": ["plastic deformation", "yield strength", "plastic strain", "ductility", "inelastic strain", "flow stress", "volume expansion", "swelling", "volumetric expansion", "volume change", "dimensional change", "volume deformation", "volume increase", "volumetric strain", "expansion strain", "dilatation", "volume distortion"],
            "Fatigue": ["fatigue life", "cyclic loading", "cycle life", "fatigue crack", "stress cycling", "mechanical fatigue", "low cycle fatigue", "high cycle fatigue", "fatigue damage", "fatigue failure", "cyclic deformation", "cyclic stress", "endurance limit", "S-N curve"],
            "Crack and Fracture": ["electrode cracking", "crack propagation", "crack growth", "crack initiation", "micro-cracking", "fracture", "fracture toughness", "brittle fracture", "ductile fracture", "crack branching", "crack tip", "stress intensity factor", "fracture mechanics", "J-integral", "cleavage fracture"],
            "Degradation": ["SEI formation", "electrolyte degradation", "capacity fade", "cycle degradation", "electrode degradation", "electrolyte decomposition", "aging", "thermal degradation", "mechanical degradation", "capacity loss", "fading", "deterioration", "corrosion", "passivation", "side reaction"]
        }
        
        component_terms = ["electrode", "electrolyte", "phase", "cathode", "anode", "solid electrolyte", "separator", "current collector"]
        exclude_words = ["et al", "electrochem soc", "phys", "model", "based", "high", "field"]
        
        term_freqs = Counter()
        processed_count = 0
        chunk_size = 50
        
        progress_bar = st.progress(0)
        
        while True:
            cursor.execute(query)
            rows = cursor.fetchmany(chunk_size)
            
            if not rows:
                break
            
            for (content,) in rows:
                if not content:
                    continue
                
                if len(content) > nlp.max_length:
                    content = content[:nlp.max_length]
                
                try:
                    doc = nlp(content.lower())
                    
                    phrases = [span.text.strip() for span in doc.noun_chunks if 1 < len(span.text.split()) <= 3]
                    words = [token.text for token in doc if token.text.isalpha() and not token.is_stop and len(token.text) > 3]
                    
                    n_grams = list(chain(ngrams(words, 2), ngrams(words, 3)))
                    n_gram_phrases = [' '.join(gram) for gram in n_grams if 1 < len(gram) <= 3]
                    
                    term_freqs.update(phrases + n_gram_phrases + words)
                    
                except Exception as e:
                    update_log(f"Error processing content: {str(e)}")
                    continue
            
            processed_count += len(rows)
            progress_bar.progress(min(1.0, processed_count / max_papers))
            
            del rows, doc
            if processed_count % 200 == 0:
                gc.collect()
        
        conn.close()
        
        if not term_freqs:
            return {}, Counter()
        
        filtered_terms = {
            term: freq for term, freq in term_freqs.items() 
            if freq >= min_freq and not any(w in term.lower() for w in exclude_words)
        }
        
        del term_freqs
        gc.collect()
        
        unique_terms = list(filtered_terms.keys())
        term_embeddings = {}
        
        batch_size = 16
        for i in range(0, len(unique_terms), batch_size):
            batch = unique_terms[i:i+batch_size]
            embeddings = get_scibert_embedding_batch(batch)
            
            for term, emb in zip(batch, embeddings):
                if emb is not None:
                    term_embeddings[term] = emb
            
            progress_bar.progress(min(1.0, 0.5 + (i + len(batch)) / len(unique_terms) / 2))
            
            del embeddings, batch
            if i % 64 == 0:
                gc.collect()
        
        seed_embeddings = {}
        for cat, terms in categories.items():
            emb_list = get_scibert_embedding_batch(terms)
            seed_embeddings[cat] = [e for e in emb_list if e is not None]
        
        component_embeddings = get_scibert_embedding_batch(component_terms)
        component_embeddings = [e for e in component_embeddings if e is not None]
        
        categorized_terms = {cat: [] for cat in categories}
        categorized_terms["Component"] = []
        other_terms = []
        
        for term, freq in filtered_terms.items():
            term_emb = term_embeddings.get(term)
            if term_emb is None:
                continue
            
            best_cat = None
            best_score = 0
            
            for cat, embeddings in seed_embeddings.items():
                for seed_emb in embeddings:
                    if np.linalg.norm(term_emb) == 0 or np.linalg.norm(seed_emb) == 0:
                        continue
                    score = np.dot(term_emb, seed_emb) / (np.linalg.norm(term_emb) * np.linalg.norm(seed_emb))
                    if score > similarity_threshold and score > best_score:
                        best_cat = cat
                        best_score = score
            
            if best_cat:
                categorized_terms[best_cat].append((term, freq, best_score))
            elif any(np.dot(term_emb, comp_emb) / (np.linalg.norm(term_emb) * np.linalg.norm(comp_emb)) > similarity_threshold 
                     for comp_emb in component_embeddings if np.linalg.norm(term_emb) != 0 and np.linalg.norm(comp_emb) != 0):
                categorized_terms["Component"].append((term, freq, best_score))
            else:
                other_terms.append((term, freq, best_score))
        
        for cat in categorized_terms:
            categorized_terms[cat] = sorted(categorized_terms[cat], key=lambda x: x[1], reverse=True)[:100]
        categorized_terms["Other"] = sorted(other_terms, key=lambda x: x[1], reverse=True)[:50]
        
        del term_embeddings, seed_embeddings, component_embeddings, filtered_terms
        gc.collect()
        
        update_log(f"Categorized {sum(len(terms) for terms in categorized_terms.values())} terms")
        return categorized_terms, Counter()
        
    except Exception as e:
        update_log(f"Error categorizing terms: {str(e)}")
        st.error(f"Error: {str(e)}")
        return {}, Counter()

# ==================== KNOWLEDGE GRAPH (OPTIMIZED) ====================
@st.cache_data(ttl=3600, max_entries=5)
def build_knowledge_graph_data(categorized_terms, db_file, min_co_occurrence=2, top_n=10, max_papers=500):
    """Build knowledge graph with memory limits"""
    try:
        update_log(f"Building knowledge graph (max {max_papers} papers)")
        
        conn = sqlite3.connect(db_file)
        
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        if max_papers:
            query += f" LIMIT {max_papers}"
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        G = nx.Graph()
        
        categories = list(categorized_terms.keys())
        for cat in categories:
            G.add_node(cat, type="category", freq=0, size=2000, color="skyblue")
        
        term_freqs = {}
        term_to_category = {}
        
        for cat, terms in categorized_terms.items():
            for term, freq, score in terms[:top_n]:
                G.add_node(term, type="term" if cat != "Component" else "component",
                          freq=freq, category=cat, size=500 + 2000 * (freq / max([f for _, f, _ in terms], default=1)),
                          color="salmon" if cat != "Component" else "lightgreen", score=score)
                G.add_edge(cat, term, weight=1.0, type="category-term")
                term_freqs[term] = freq
                term_to_category[term] = cat
        
        co_occurrence_counts = defaultdict(lambda: defaultdict(int))
        chunk_size = 50
        processed = 0
        
        nlp_model = get_nlp()
        
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            
            for (content,) in rows:
                if not content:
                    continue
                
                if len(content) > nlp_model.max_length:
                    content = content[:nlp_model.max_length]
                
                doc = nlp_model(content.lower())
                
                for sent in doc.sents:
                    sent_terms = [term for term in term_freqs if re.search(rf'\b{re.escape(term)}\b', sent.text, re.IGNORECASE)]
                    for term1, term2 in combinations(sent_terms, 2):
                        if term1 != term2:
                            co_occurrence_counts[term1][term2] += 1
                            co_occurrence_counts[term2][term1] += 1
            
            processed += len(rows)
            del rows, doc
            if processed % 200 == 0:
                gc.collect()
        
        conn.close()
        
        for term1, related_terms in co_occurrence_counts.items():
            for term2, count in related_terms.items():
                if count >= min_co_occurrence and term1 in G.nodes and term2 in G.nodes:
                    G.add_edge(term1, term2, weight=count, type="term-term")
        
        nodes_df = pd.DataFrame([(n, d["type"], d.get("category", ""), d.get("freq", 0))
                                for n, d in G.nodes(data=True)],
                               columns=["node", "type", "category", "frequency"])
        edges_df = pd.DataFrame([(u, v, d["weight"], d["type"])
                                for u, v, d in G.edges(data=True)],
                               columns=["source", "target", "weight", "type"])
        
        st.session_state.knowledge_graph = G
        
        del co_occurrence_counts
        gc.collect()
        
        return G, (nodes_df.to_csv(index=False), edges_df.to_csv(index=False))
        
    except Exception as e:
        update_log(f"Error building knowledge graph: {str(e)}")
        return None, None

# ==================== VISUALIZATION HELPERS ====================
@st.cache_data(ttl=1800)
def plot_word_cloud(terms, top_n, font_size, colormap):
    """Generate word cloud with memory limits"""
    if not terms:
        return None
    
    term_dict = {term: freq for term, freq, _ in terms[:min(top_n, len(terms))]}
    
    try:
        wordcloud = WordCloud(
            width=800, height=400, background_color="white",
            min_font_size=8, max_font_size=font_size,
            colormap=colormap, max_words=top_n
        ).generate_from_frequencies(term_dict)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud of Top {top_n} Terms")
        plt.tight_layout()
        
        return fig
    except Exception as e:
        update_log(f"Word cloud error: {str(e)}")
        return None

def visualize_knowledge_graph_communities(G):
    """Visualize the knowledge graph with community detection"""
    if G is None or not G.edges():
        return None
    
    partition = community_louvain.best_partition(G)
    communities = set(partition.values())
    color_map = cm.get_cmap('tab20', len(communities))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    node_colors = [color_map(partition[node]) for node in G.nodes()]
    node_sizes = [G.nodes[node].get('size', 500) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    
    edge_widths = [0.5 + 2 * (d['weight'] / max([d2['weight'] for _, _, d2 in G.edges(data=True)], default=1))
                   for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', ax=ax)
    
    important_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'category' or
                       G.nodes[node].get('freq', 0) > 10]
    labels = {node: node for node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    community_labels = {}
    for node, comm_id in partition.items():
        if G.nodes[node].get('type') == 'category':
            community_labels[comm_id] = node
    
    legend_elements = []
    for comm_id, label in community_labels.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color_map(comm_id),
                                          markersize=10, label=label))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Knowledge Graph with Community Detection")
    plt.tight_layout()
    
    return fig

# ==================== NER WITH SCIBERT ====================
def perform_ner_on_terms(db_file, selected_terms):
    try:
        update_log(f"Starting NER analysis for terms: {', '.join(selected_terms)}")
        conn = sqlite3.connect(db_file)
        query = "SELECT id as paper_id, title, year, content FROM papers WHERE content IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            update_log(f"No valid content found in {os.path.basename(db_file)}")
            st.error("No valid content found.")
            return pd.DataFrame()
        
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
        
        ref_embeddings = {cat: [get_scibert_embedding_batch([term])[0] if get_scibert_embedding_batch([term])[0] is not None else None for term in terms] for cat, terms in categories.items()}
        ref_embeddings = {cat: [e for e in embs if e is not None] for cat, embs in ref_embeddings.items()}
        
        term_patterns = {term: re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE) for term in selected_terms}
        
        entities = []
        entity_set = set()
        progress_bar = st.progress(0)
        
        nlp_model = get_nlp()
        
        for i, row in df.iterrows():
            try:
                text = row["content"].lower()
                text = re.sub(r"young's modulus|youngs modulus", "young's modulus", text)
                
                if len(text) > nlp_model.max_length:
                    text = text[:nlp_model.max_length]
                
                if not text.strip() or len(text) < 10:
                    continue
                
                doc = nlp_model(text)
                spans = []
                
                for sent_idx, sent in enumerate(doc.sents):
                    if any(term_patterns[term].search(sent.text) for term in selected_terms):
                        start_sent_idx = max(0, sent_idx - 2)
                        end_sent_idx = min(len(list(doc.sents)), sent_idx + 3)
                        for nearby_sent in list(doc.sents)[start_sent_idx:end_sent_idx]:
                            matches = re.finditer(numerical_pattern, nearby_sent.text, re.IGNORECASE)
                            for match in matches:
                                start_char = nearby_sent.start_char + match.start()
                                end_char = nearby_sent.start_char + match.end()
                                span = doc.char_span(start_char, end_char, alignment_mode="expand")
                                if span:
                                    spans.append((span, sent.text, nearby_sent.text))
                
                if not spans:
                    continue
                
                for span, orig_sent, nearby_sent in spans:
                    span_text = span.text.lower().strip()
                    if not span_text:
                        continue
                    
                    term_matched = False
                    for term in selected_terms:
                        if term_patterns[term].search(span_text) or term_patterns[term].search(orig_sent) or term_patterns[term].search(nearby_sent):
                            term_matched = True
                            break
                    
                    if not term_matched:
                        span_embedding = get_scibert_embedding_batch([span_text])[0]
                        if span_embedding is None:
                            continue
                        
                        term_embeddings = [get_scibert_embedding_batch([term])[0] for term in selected_terms if get_scibert_embedding_batch([term])[0] is not None]
                        similarities = [
                            np.dot(span_embedding, t_emb) / (np.linalg.norm(span_embedding) * np.linalg.norm(t_emb))
                            for t_emb in term_embeddings if np.linalg.norm(span_embedding) != 0 and np.linalg.norm(t_emb) != 0
                        ]
                        if any(s > 0.5 for s in similarities):
                            term_matched = True
                    
                    if not term_matched:
                        continue
                    
                    value_match = re.match(numerical_pattern, span_text, re.IGNORECASE)
                    value = None
                    unit = None
                    
                    if value_match:
                        try:
                            value = float(value_match.group(1))
                            unit = value_match.group(2).upper()
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
                        except ValueError:
                            continue
                    
                    span_embedding = get_scibert_embedding_batch([span_text])[0]
                    if span_embedding is None:
                        continue
                    
                    best_label = None
                    best_score = 0
                    
                    for label, ref_embeds in ref_embeddings.items():
                        for ref_embed in ref_embeds:
                            if np.linalg.norm(span_embedding) == 0 or np.linalg.norm(ref_embed) == 0:
                                continue
                            similarity = np.dot(span_embedding, ref_embed) / (np.linalg.norm(span_embedding) * np.linalg.norm(ref_embed))
                            if similarity > similarity_threshold and similarity > best_score:
                                best_label = label
                                best_score = similarity
                    
                    if not best_label:
                        continue
                    
                    if value is not None and unit is not None:
                        if unit not in valid_units.get(best_label, []):
                            context = text[max(0, span.start_char - 100):min(len(text), span.end_char + 100)]
                            context_embedding = get_scibert_embedding_batch([context])[0]
                            if context_embedding is None:
                                continue
                            
                            unit_valid = False
                            for v_unit in valid_units.get(best_label, []):
                                unit_embedding = get_scibert_embedding_batch([f"{span_text} {v_unit}"])[0]
                                if unit_embedding is None:
                                    continue
                                unit_score = np.dot(context_embedding, unit_embedding) / (np.linalg.norm(context_embedding) * np.linalg.norm(unit_embedding))
                                if unit_score > 0.6:
                                    unit_valid = True
                                    unit = v_unit
                                    break
                            
                            if not unit_valid:
                                continue
                            
                            range_valid = False
                            for min_val, max_val, expected_unit in valid_ranges.get(best_label, [(None, None, None)]):
                                if expected_unit == unit and min_val is not None and max_val is not None:
                                    if min_val <= value <= max_val:
                                        range_valid = True
                                        break
                            
                            if not range_valid:
                                continue
                        elif any(v is None for v in valid_units.get(best_label, [])):
                            pass
                        else:
                            continue
                    
                    entity_key = (row["paper_id"], span_text, best_label, value if value is not None else "", unit if unit is not None else "")
                    if entity_key in entity_set:
                        continue
                    entity_set.add(entity_key)
                    
                    context_start = max(0, span.start_char - 100)
                    context_end = min(len(text), span.end_char + 100)
                    context_text = text[context_start:context_end].replace("\n", " ")
                    
                    entities.append({
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "year": row["year"],
                        "entity_text": span.text,
                        "entity_label": best_label,
                        "value": value,
                        "unit": unit,
                        "context": context_text,
                        "score": best_score
                    })
                
                progress_bar.progress((i + 1) / len(df))
                
            except Exception as e:
                update_log(f"Error processing entry {row['paper_id']}: {str(e)}")
        
        update_log(f"Extracted {len(entities)} entities")
        
        if st.session_state.knowledge_graph is not None:
            entities_df = pd.DataFrame(entities)
        
        return pd.DataFrame(entities)
        
    except Exception as e:
        update_log(f"NER analysis failed: {str(e)}")
        st.error(f"NER analysis failed: {str(e)}")
        return pd.DataFrame()

# ==================== LLM QUANTIFIED NER ====================
@st.cache_data(ttl=3600)
def llm_quantified_ner(db_file: str):
    """Use the currently loaded LLM to extract quantified terms."""
    tokenizer = st.session_state.llm_tokenizer
    model = st.session_state.llm_model
    
    if not tokenizer or not model:
        st.error("No LLM loaded. Please select a model in the sidebar first.")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_file)
    query = "SELECT id as paper_id, title, year, content FROM papers WHERE content IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    system_prompt = (
        "You are a battery reliability expert. Extract EVERY quantified degradation term from the text. "
        "Output ONLY a valid JSON array of objects. Each object must have fields: "
        "term (string), value (float), unit (string), mechanism (string), context (string). "
        "Possible mechanisms: Deformation, Fatigue, Crack and Fracture, Degradation. "
        "If a numerical value is given without an explicit unit, infer the most likely unit from context. "
        "If no relevant quantification is found, return an empty array. "
        "Examples: "
        "[{\"term\": \"capacity fade\", \"value\": 15.2, \"unit\": \"%\", \"mechanism\": \"Degradation\", \"context\": \"after 500 cycles at 45C\"}] "
        "[{\"term\": \"crack length\", \"value\": 12.5, \"unit\": \"um\", \"mechanism\": \"Crack and Fracture\", \"context\": \"measured at the electrode surface\"}] "
        "Now process the following paper content:"
    )
    
    entities = []
    progress = st.progress(0)
    
    for i, row in df.iterrows():
        text = row["content"]
        
        if len(text) > 1200:
            text = text[:1200] + " ... [truncated]"
        
        full_prompt = f"{system_prompt}\nPaper ID: {row['paper_id']}\nTitle: {row['title']}\n{text}"
        
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        json_match = re.search(r'\[.*\]', generated, re.DOTALL)
        if json_match:
            try:
                items = json.loads(json_match.group(0))
                for item in items:
                    if all(k in item for k in ("term", "value", "unit", "mechanism")):
                        item["paper_id"] = row["paper_id"]
                        item["title"] = row["title"]
                        item["year"] = row["year"]
                        entities.append(item)
            except json.JSONDecodeError:
                pass
        
        progress.progress((i + 1) / len(df))
    
    return pd.DataFrame(entities)

# ==================== NARRATIVE GENERATION ====================
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

# ==================== VISUALIZATION HELPERS ====================
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

# ==================== MAIN APP ====================
st.header("Select or Upload Database")

with st.sidebar:
    st.subheader("Memory Management")
    
    if st.button("Clear All Memory & Cache"):
        clear_memory()
        st.session_state.categorized_terms = None
        st.session_state.knowledge_graph = None
        st.rerun()
    
    mem_usage = get_memory_usage()
    st.metric("Current Memory Usage", f"{mem_usage:.1f} MB")
    
    st.subheader("Analysis Parameters")
    max_papers = st.slider("Max Papers to Process", min_value=100, max_value=2000, value=500, step=100, 
                          help="Lower values = less memory usage")
    min_freq = st.slider("Minimum Term Frequency", min_value=1, max_value=20, value=5)
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    top_n = st.slider("Top N Terms", min_value=5, max_value=30, value=10)
    
    st.info(f"Recommended: Keep max papers <= {max_papers} to avoid crashes")

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
    tab1, tab2, tab3, tab4 = st.tabs(["Database Inspection", "Term Categorization", "Knowledge Graph", "NER Analysis"])
    
    with tab1:
        st.header("Database Inspection")
        if st.button("Inspect Database", key="inspect_button"):
            with st.spinner("Inspecting..."):
                inspect_database(st.session_state.db_file)
    
    with tab2:
        st.header("Term Categorization")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_terms_button = st.button("Start Categorization", key="categorize_terms", type="primary")
        with col2:
            if st.button("Clear Results", key="clear_results"):
                st.session_state.categorized_terms = None
                st.cache_data.clear()
                gc.collect()
                st.rerun()
        
        if analyze_terms_button:
            with st.spinner(f"Categorizing terms (max {max_papers} papers)..."):
                try:
                    st.session_state.categorized_terms, _ = categorize_terms(
                        st.session_state.db_file,
                        similarity_threshold=similarity_threshold,
                        min_freq=min_freq,
                        max_papers=max_papers
                    )
                    
                    if st.session_state.categorized_terms:
                        st.success(f"Categorized {sum(len(terms) for terms in st.session_state.categorized_terms.values())} terms!")
                        
                        for cat, terms in st.session_state.categorized_terms.items():
                            if terms:
                                with st.expander(f"{cat} ({len(terms)} terms)", expanded=False):
                                    term_df = pd.DataFrame(terms[:top_n], columns=["Term", "Frequency", "Score"])
                                    st.dataframe(term_df, use_container_width=True)
                                    
                                    fig = plot_word_cloud(terms, top_n, 40, "viridis")
                                    if fig:
                                        st.pyplot(fig)
                                    
                                    csv_data = pd.DataFrame(terms).to_csv(index=False)
                                    st.download_button(f"Download {cat} CSV", csv_data, f"{cat.lower()}_terms.csv", "text/csv")
                        
                        gc.collect()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    update_log(f"Categorization error: {str(e)}")
                    clear_memory(verbose=False)
    
    with tab3:
        st.header("Knowledge Graph")
        
        if st.button("Build Knowledge Graph", key="build_graph"):
            if st.session_state.categorized_terms:
                with st.spinner("Building graph..."):
                    try:
                        G, csv_data = build_knowledge_graph_data(
                            st.session_state.categorized_terms,
                            st.session_state.db_file,
                            min_co_occurrence=min_freq,
                            top_n=top_n,
                            max_papers=max_papers
                        )
                        
                        if G and G.edges():
                            fig, ax = plt.subplots(figsize=(12, 10))
                            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
                            
                            node_colors = []
                            node_sizes = []
                            for node in G.nodes:
                                if G.nodes[node]["type"] == "category":
                                    node_colors.append("skyblue")
                                    node_sizes.append(2000)
                                elif G.nodes[node]["type"] == "component":
                                    node_colors.append("lightgreen")
                                    node_sizes.append(1500)
                                else:
                                    node_colors.append("salmon")
                                    node_sizes.append(G.nodes[node].get("size", 500))
                            
                            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
                            
                            term_term_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "term-term"]
                            category_term_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "category-term"]
                            
                            nx.draw_networkx_edges(G, pos, edgelist=term_term_edges, width=1.0, alpha=0.5, edge_color="gray", ax=ax)
                            nx.draw_networkx_edges(G, pos, edgelist=category_term_edges, width=2.0, alpha=0.7, edge_color="blue", ax=ax)
                            
                            important_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'category' or G.nodes[node].get('freq', 0) > 10]
                            labels = {node: node for node in important_nodes}
                            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
                            
                            ax.set_title(f"Knowledge Graph (Top {top_n} Terms per Category)")
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            if csv_data:
                                st.download_button("Download Nodes CSV", csv_data[0], "graph_nodes.csv", "text/csv")
                                st.download_button("Download Edges CSV", csv_data[1], "graph_edges.csv", "text/csv")
                            
                            del G
                            gc.collect()
                            
                        else:
                            st.warning("No graph generated. Check logs.")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        update_log(f"Graph error: {str(e)}")
            else:
                st.warning("Run term categorization first!")
    
    with tab4:
        st.header("NER Analysis")
        
        if st.session_state.categorized_terms or st.session_state.term_counts:
            available_terms = []
            if st.session_state.term_counts:
                available_terms += [term for term, count in st.session_state.term_counts.items() if count > 0]
            if st.session_state.categorized_terms:
                for terms in st.session_state.categorized_terms.values():
                    available_terms += [term for term, _, _ in terms]
            available_terms = sorted(list(set(available_terms)))
            
            default_terms = [term for term in ["electrode cracking", "SEI formation", "cyclic mechanical damage", "electrolyte degradation", "capacity fade"] if term in available_terms]
            selected_terms = st.multiselect("Select Terms for NER", available_terms, default_terms, key="select_terms")
            
            use_llm = st.checkbox("Use LLM for quantified NER", value=True)
            
            if use_llm and TRANSFORMERS_AVAILABLE:
                model_choice = st.selectbox("Select LLM Model", list(LLM_MODELS.keys()), index=0, key="llm_model_select")
            elif use_llm and not TRANSFORMERS_AVAILABLE:
                st.error("Transformers is too old. Please upgrade to use LLM features.")
            
            if st.button("Run NER Analysis", key="ner_analyze"):
                if os.path.exists(UNIVERSE_DB_FILE):
                    with st.spinner("Processing NER analysis..."):
                        if use_llm and TRANSFORMERS_AVAILABLE:
                            tokenizer, model, loaded_key = load_llm(model_choice)
                            if tokenizer and model:
                                llm_df = llm_quantified_ner(UNIVERSE_DB_FILE)
                                st.session_state.ner_results = llm_df
                                if not llm_df.empty:
                                    st.session_state.last_structured_insights = {
                                        "entities": llm_df.to_dict(orient="records"),
                                        "summary": f"Extracted {len(llm_df)} quantified entities."
                                    }
                            else:
                                st.error("Failed to load LLM. Check logs.")
                        else:
                            if not selected_terms:
                                st.warning("Select at least one term for heuristic NER.")
                            else:
                                ner_df = perform_ner_on_terms(UNIVERSE_DB_FILE, selected_terms)
                                st.session_state.ner_results = ner_df
                                if not ner_df.empty:
                                    st.session_state.last_structured_insights = {
                                        "entities": ner_df.to_dict(orient="records"),
                                        "summary": f"Extracted {len(ner_df)} entities using heuristic method."
                                    }
                else:
                    st.error(f"Cannot perform NER analysis: {os.path.basename(UNIVERSE_DB_FILE)} not found.")
                    update_log(f"Cannot perform NER analysis: {os.path.basename(UNIVERSE_DB_FILE)} not found.")
                
                if st.session_state.ner_results is not None and not st.session_state.ner_results.empty:
                    st.success(f"Extracted {len(st.session_state.ner_results)} entities!")
                    st.dataframe(st.session_state.ner_results.head(100), use_container_width=True)
                    ner_csv = st.session_state.ner_results.to_csv(index=False)
                    st.download_button("Download NER Data CSV", ner_csv, "ner_data.csv", "text/csv", key="download_ner")
                    
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
            st.warning("Run term categorization first!")

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
