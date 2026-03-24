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
import gc  # for garbage collection
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
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# ==================== SPACY SETUP (LAZY LOADING) ====================
@st.cache_resource
def load_spacy_model():
    """Load spaCy model with resource management"""
    try:
        nlp = spacy.load("en_core_web_sm")  # Use smaller model
        nlp.max_length = 100_000  # Reduced from 500k
        
        @Language.component("custom_tokenizer")
        def custom_tokenizer(doc):
            hyphenated_phrases = ["electrode-cracking", "SEI-formation", "capacity-fade"]
            for phrase in hyphenated_phrases:
                if phrase.lower() in doc.text.lower():
                    with doc.retokenize() as retokenizer:
                        for match in re.finditer(rf'\b{re.escape(phrase)}\b', doc.text, re.IGNORECASE):
                            start_char, end_char = match.span()
                            for token in doc:
                                if token.idx >= start_char:
                                    retokenizer.merge(doc[token.i:token.i+1])
                                    break
            return doc
        
        if "custom_tokenizer" not in nlp.pipe_names:
            nlp.add_pipe("custom_tokenizer", before="parser")
        
        return nlp
    except Exception as e:
        st.error(f"Failed to load spaCy: {e}")
        st.stop()

# Initialize lazily
nlp = None

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
        
        # Move to CPU explicitly
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
    "Qwen2.5-0.5B-Instruct (tiny)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B (CPU)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

@st.cache_resource(max_entries=1)
def load_llm(model_key: str):
    """Load LLM with proper cleanup"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, None
    
    if model_key not in LLM_MODELS:
        return None, None, None
    
    model_id = LLM_MODELS[model_key]
    
    # Clear previous model
    if "llm_model" in st.session_state:
        del st.session_state.llm_model
    if "llm_tokenizer" in st.session_state:
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
        
        return tokenizer, model, model_key
    except Exception as e:
        st.error(f"Failed to load {model_key}: {str(e)}")
        return None, None, None

# ==================== SESSION STATE ====================
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "db_file" not in st.session_state:
    st.session_state.db_file = UNIVERSE_DB_FILE if os.path.exists(UNIVERSE_DB_FILE) else None
if "llm_tokenizer" not in st.session_state:
    st.session_state.llm_tokenizer = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None

# ==================== HELPER FUNCTIONS ====================
def update_log(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

@st.cache_data(ttl=1800, max_entries=100)
def get_scibert_embedding_batch(texts):
    """Compute embeddings in batches with cache limits"""
    if not texts:
        return []
    
    tokenizer, model = get_scibert_models()
    if tokenizer is None or model is None:
        return [None] * len(texts)
    
    try:
        # Process in sub-batches to prevent memory spikes
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
            
            # Clear intermediate tensors
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
            cursor.execute("PRAGMA table_info(papers);")
            schema = cursor.fetchall()
            schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
            st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)
            
            available_columns = [col[1] for col in schema]
            
            # Count papers without loading content
            cursor.execute("SELECT COUNT(*) FROM papers WHERE content IS NOT NULL")
            total_papers = cursor.fetchone()[0]
            st.subheader("Total Valid Papers")
            st.write(f"{total_papers} papers")
            
            # Sample query (LIMIT 5)
            select_columns = ["id", "title", "year"]
            if "content" in available_columns:
                select_columns.append("substr(content, 1, 200) as sample_content")
            
            query = f"SELECT {', '.join(select_columns)} FROM papers WHERE content IS NOT NULL LIMIT 5"
            df = pd.read_sql_query(query, conn)
            st.subheader("Sample Rows (First 5 Papers)")
            st.dataframe(df, use_container_width=True)
        
        conn.close()
        return None
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        update_log(f"Error: {str(e)}")
        return None

# ==================== OPTIMIZED TERM CATEGORIZATION ====================
@st.cache_data(ttl=3600, max_entries=10)
def categorize_terms(db_file, similarity_threshold=0.7, min_freq=5, max_papers=500):
    """
    Memory-optimized term categorization with chunked processing
    """
    global nlp
    
    try:
        update_log(f"Starting term categorization (max {max_papers} papers)")
        
        # Load spaCy lazily
        if nlp is None:
            nlp = load_spacy_model()
        
        # Load SciBERT lazily
        tokenizer, model = get_scibert_models()
        if tokenizer is None:
            st.error("SciBERT not available")
            return {}, Counter()
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # CRITICAL: Use LIMIT to prevent OOM
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        if max_papers:
            query += f" LIMIT {max_papers}"
        
        categories = {
            "Deformation": ["plastic deformation", "yield strength", "volume expansion"],
            "Fatigue": ["fatigue life", "cyclic loading", "cycle life"],
            "Crack and Fracture": ["electrode cracking", "crack propagation", "fracture"],
            "Degradation": ["SEI formation", "electrolyte degradation", "capacity fade"]
        }
        
        component_terms = ["electrode", "electrolyte", "cathode", "anode"]
        exclude_words = ["et al", "electrochem soc", "phys", "model", "based"]
        
        term_freqs = Counter()
        processed_count = 0
        chunk_size = 50  # Process 50 papers at a time
        
        progress_bar = st.progress(0)
        
        # CRITICAL: Process in chunks with explicit cleanup
        while True:
            cursor.execute(query)
            rows = cursor.fetchmany(chunk_size)
            
            if not rows:
                break
            
            for (content,) in rows:
                if not content:
                    continue
                
                # Truncate long content
                if len(content) > nlp.max_length:
                    content = content[:nlp.max_length]
                
                try:
                    doc = nlp(content.lower())
                    
                    # Extract terms directly to Counter (no intermediate list)
                    phrases = [span.text.strip() for span in doc.noun_chunks 
                              if 1 < len(span.text.split()) <= 3]
                    words = [token.text for token in doc 
                            if token.text.isalpha() and not token.is_stop and len(token.text) > 3]
                    
                    n_grams = list(chain(ngrams(words, 2), ngrams(words, 3)))
                    n_gram_phrases = [' '.join(gram) for gram in n_grams if 1 < len(gram) <= 3]
                    
                    term_freqs.update(phrases + n_gram_phrases + words)
                    
                except Exception as e:
                    update_log(f"Error processing content: {str(e)}")
                    continue
            
            processed_count += len(rows)
            progress_bar.progress(min(1.0, processed_count / max_papers))
            
            # CRITICAL: Clear memory after each chunk
            del rows, doc
            if processed_count % 200 == 0:
                gc.collect()
        
        conn.close()
        
        if not term_freqs:
            return {}, Counter()
        
        # Filter rare terms BEFORE embedding
        filtered_terms = {
            term: freq for term, freq in term_freqs.items() 
            if freq >= min_freq and not any(w in term.lower() for w in exclude_words)
        }
        
        # Clear term_freqs to free memory
        del term_freqs
        gc.collect()
        
        unique_terms = list(filtered_terms.keys())
        term_embeddings = {}
        
        # Batch embedding with progress
        batch_size = 16
        for i in range(0, len(unique_terms), batch_size):
            batch = unique_terms[i:i+batch_size]
            embeddings = get_scibert_embedding_batch(batch)
            
            for term, emb in zip(batch, embeddings):
                if emb is not None:
                    term_embeddings[term] = emb
            
            progress_bar.progress(min(1.0, 0.5 + (i + len(batch)) / len(unique_terms) / 2))
            
            # Clear batch memory
            del embeddings, batch
            if i % 64 == 0:
                gc.collect()
        
        # Pre-compute seed embeddings
        seed_embeddings = {}
        for cat, terms in categories.items():
            emb_list = get_scibert_embedding_batch(terms)
            seed_embeddings[cat] = [e for e in emb_list if e is not None]
        
        component_embeddings = get_scibert_embedding_batch(component_terms)
        component_embeddings = [e for e in component_embeddings if e is not None]
        
        # Categorize terms
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
        
        # Sort and limit
        for cat in categorized_terms:
            categorized_terms[cat] = sorted(categorized_terms[cat], key=lambda x: x[1], reverse=True)[:100]
        categorized_terms["Other"] = sorted(other_terms, key=lambda x: x[1], reverse=True)[:50]
        
        # Clear large objects
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
        
        # CRITICAL: LIMIT papers loaded
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        if max_papers:
            query += f" LIMIT {max_papers}"
        
        # Process in chunks instead of loading all
        cursor = conn.cursor()
        cursor.execute(query)
        
        G = nx.Graph()
        
        # Add category nodes
        categories = list(categorized_terms.keys())
        for cat in categories:
            G.add_node(cat, type="category", freq=0, size=2000, color="skyblue")
        
        # Add term nodes (limited to top_n per category)
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
        
        # Compute co-occurrences in chunks
        co_occurrence_counts = defaultdict(lambda: defaultdict(int))
        chunk_size = 50
        processed = 0
        
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            
            for (content,) in rows:
                if not content:
                    continue
                
                global nlp
                if nlp is None:
                    nlp = load_spacy_model()
                
                if len(content) > nlp.max_length:
                    content = content[:nlp.max_length]
                
                doc = nlp(content.lower())
                
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
        
        # Add co-occurrence edges
        for term1, related_terms in co_occurrence_counts.items():
            for term2, count in related_terms.items():
                if count >= min_co_occurrence and term1 in G.nodes and term2 in G.nodes:
                    G.add_edge(term1, term2, weight=count, type="term-term")
        
        # Generate DataFrames
        nodes_df = pd.DataFrame([(n, d["type"], d.get("category", ""), d.get("freq", 0))
                                for n, d in G.nodes(data=True)],
                               columns=["node", "type", "category", "frequency"])
        edges_df = pd.DataFrame([(u, v, d["weight"], d["type"])
                                for u, v, d in G.edges(data=True)],
                               columns=["source", "target", "weight", "type"])
        
        # Store graph
        st.session_state.knowledge_graph = G
        
        # Clear memory
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

# ==================== MAIN APP ====================
st.header("Select or Upload Database")

# Memory management sidebar
with st.sidebar:
    st.subheader("⚙️ Memory Management")
    
    if st.button("🗑️ Clear All Memory & Cache"):
        clear_memory()
        st.session_state.categorized_terms = None
        st.session_state.knowledge_graph = None
        st.rerun()
    
    mem_usage = get_memory_usage()
    st.metric("Current Memory Usage", f"{mem_usage:.1f} MB")
    
    st.subheader("📊 Analysis Parameters")
    max_papers = st.slider("Max Papers to Process", min_value=100, max_value=2000, value=500, step=100, 
                          help="Lower values = less memory usage")
    min_freq = st.slider("Minimum Term Frequency", min_value=1, max_value=20, value=5)
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    top_n = st.slider("Top N Terms", min_value=5, max_value=30, value=10)
    
    st.info(f"⚠️ Recommended: Keep max papers ≤ {max_papers} to avoid crashes")

# Database selection
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

# Main app tabs
if st.session_state.db_file and os.path.exists(st.session_state.db_file):
    tab1, tab2, tab3 = st.tabs(["Database Inspection", "Term Categorization", "Knowledge Graph"])
    
    with tab1:
        st.header("Database Inspection")
        if st.button("Inspect Database", key="inspect_button"):
            with st.spinner("Inspecting..."):
                inspect_database(st.session_state.db_file)
    
    with tab2:
        st.header("Term Categorization")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_terms_button = st.button("🚀 Start Categorization", key="categorize_terms", type="primary")
        with col2:
            if st.button("🗑️ Clear Results", key="clear_results"):
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
                        st.success(f"✅ Categorized {sum(len(terms) for terms in st.session_state.categorized_terms.values())} terms!")
                        
                        for cat, terms in st.session_state.categorized_terms.items():
                            if terms:
                                with st.expander(f"{cat} ({len(terms)} terms)", expanded=False):
                                    term_df = pd.DataFrame(terms[:top_n], columns=["Term", "Frequency", "Score"])
                                    st.dataframe(term_df, use_container_width=True)
                                    
                                    # Word cloud
                                    fig = plot_word_cloud(terms, top_n, 40, "viridis")
                                    if fig:
                                        st.pyplot(fig)
                                    
                                    # Download
                                    csv_data = pd.DataFrame(terms).to_csv(index=False)
                                    st.download_button(f"Download {cat} CSV", csv_data, f"{cat.lower()}_terms.csv", "text/csv")
                        
                        # Clear cache after completion
                        gc.collect()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    update_log(f"Categorization error: {str(e)}")
                    clear_memory(verbose=False)
    
    with tab3:
        st.header("Knowledge Graph")
        
        if st.button("🕸️ Build Knowledge Graph", key="build_graph"):
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
                            
                            # Download
                            if csv_data:
                                st.download_button("Download Nodes CSV", csv_data[0], "graph_nodes.csv", "text/csv")
                                st.download_button("Download Edges CSV", csv_data[1], "graph_edges.csv", "text/csv")
                            
                            # Clear graph from memory after display
                            del G
                            gc.collect()
                            
                        else:
                            st.warning("No graph generated. Check logs.")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        update_log(f"Graph error: {str(e)}")
            else:
                st.warning("⚠️ Run term categorization first!")
    
    # Logs
    with st.expander("📋 Activity Logs"):
        st.text_area("Logs", "\n".join(st.session_state.log_buffer[-20:]), height=150)

else:
    st.warning(f"⚠️ Select or upload a database file. Ensure {os.path.basename(UNIVERSE_DB_FILE)} is available.")

# Footer
st.markdown("""
---
**💡 Memory Optimization Tips:**
- Keep "Max Papers" ≤ 500 for stable performance
- Click "Clear All Memory" between heavy operations
- Use smaller models if enabling LLM features
- Close other applications to free RAM
""")
