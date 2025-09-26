import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from collections import Counter, defaultdict
import re
from transformers import AutoModel, AutoTokenizer
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from community import community_louvain
from multiprocessing import Pool
import time
from matplotlib.patches import Patch

# Matplotlib configuration
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

# Directory setup
DB_DIR = os.path.dirname(os.path.abspath(__file__))
RELIABILITY_DB_FILE = os.path.join(DB_DIR, "battery_reliability.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "battery_reliability_universe.db")

# Logging setup
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'battery_reliability_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit configuration
st.set_page_config(page_title="PREKNOWLEDGE: Priority-Enhanced Knowledge Graph for Battery Reliability Analysis", layout="wide")
st.title("PREKNOWLEDGE: Priority-Enhanced Knowledge Graph for Battery Reliability Analysis")
st.markdown("""
PREKNOWLEDGE (Priority-Enhanced Knowledge Graph for Battery Reliability Analysis) inspects SQLite databases, categorizes terms related to battery reliability challenges (e.g., electrode cracking, SEI formation) into Deformation, Fatigue, Crack and Fracture, and Degradation, builds a priority-enhanced knowledge graph emphasizing key terms, and performs NER analysis using SciBERT. The default database is `battery_reliability_universe.db` for full-text analysis of arXiv papers.
Select a database, then use the tabs to inspect the database, categorize terms, visualize relationships, or extract entities with numerical values.
""")

# Load spaCy model with fallback
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Using 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e2:
        st.error(f"Failed to load spaCy: {e2}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()

# Custom spaCy tokenizer for hyphenated phrases
@Language.component("custom_tokenizer")
def custom_tokenizer(doc):
    hyphenated_phrases = ["electrode-cracking", "SEI-formation", "cyclic-mechanical-damage", "diffusion-induced-stress", 
                          "electrolyte-degradation", "capacity-fade", "lithium-ion", "Li-ion", "crack-propagation", 
                          "crack-damage", "fracture-damage"]
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
                        retokenizer.merge(doc[start_token:start_token+len(phrase.split('-')+phrase.split())])
    return doc

nlp.add_pipe("custom_tokenizer", before="parser")
nlp.max_length = 500_000

# Load SciBERT model
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# Initialize session state
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

def update_log(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

@st.cache_data
def get_scibert_embedding(texts):
    try:
        if isinstance(texts, str):
            texts = [texts]
        if not texts or all(not t.strip() for t in texts):
            update_log("Skipping empty text list for SciBERT embedding")
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
        update_log(f"SciBERT embedding failed: {str(e)}")
        return [None] * len(texts) if isinstance(texts, list) else None

# Predefined key terms for battery reliability
KEY_TERMS = [
    "electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress",
    "micro-cracking", "electrolyte degradation", "capacity fade", "lithium plating", "thermal runaway",
    "mechanical degradation", "cycle life", "crack", "fracture", "cycles", "degradation", "capacity",
    "cycling", "stress", "diffusion", "solid electrolyte interphase", "degrades the battery capacity",
    "cycling degradation", "calendar degradation", "complex cycling damage", "chemo-mechanical degradation mechanisms",
    "microcrack formation", "differential degradation mechanisms", "lithiation", "electrochemical performance",
    "mechanical integrity", "battery safety", "Coupled mechanical-chemical degradation", "physics-based models",
    "predict degradation mechanisms", "Electrode Side Reactions", "Capacity Loss", "Mechanical Degradation",
    "Particle Versus SEI Cracking", "degradation models", "predict degradation", "crack damage", "fracture damage",
    "fatigue damage"
]

# Precompute embeddings for key terms
KEY_TERMS_EMBEDDINGS = get_scibert_embedding(KEY_TERMS)
KEY_TERMS_EMBEDDINGS = [emb for emb in KEY_TERMS_EMBEDDINGS if emb is not None]

# Define term units for categories
term_units = {
    "Deformation": "%",
    "Fatigue": "cycles",
    "Crack and Fracture": "μm",
    "Degradation": "%"
}

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
            conn.close()
            return None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('papers', 'pages');")
        available_tables = [t[0] for t in cursor.fetchall()]
        
        if "papers" in available_tables:
            st.subheader("Schema of 'papers' Table")
            cursor.execute("PRAGMA table_info(papers);")
            schema = cursor.fetchall()
            schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
            st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)

            query = "SELECT id AS paper_id, title, substr(content, 1, 200) AS sample_content FROM papers WHERE content IS NOT NULL LIMIT 5"
            df = pd.read_sql_query(query, conn)
            st.subheader("Sample Rows from 'papers' Table (First 5 Entries)")
            if df.empty:
                st.warning("No valid entries found in the 'papers' table.")
            else:
                st.dataframe(df, use_container_width=True)

            cursor.execute("SELECT COUNT(*) as count FROM papers WHERE content IS NOT NULL")
            total_entries = cursor.fetchone()[0]
            st.subheader("Total Valid Entries")
            st.write(f"{total_entries} entries")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='page_fts';")
            if cursor.fetchone():
                search_query = st.text_input("Search papers (FTS5 enabled):", "")
                if search_query:
                    df_search = pd.read_sql_query(
                        "SELECT id AS paper_id, title, content, rank FROM page_fts WHERE page_fts MATCH ? ORDER BY rank LIMIT 50",
                        conn, params=(search_query,)
                    )
                    st.subheader("Search Results")
                    st.dataframe(df_search, use_container_width=True)

            terms_to_search = ["electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress", "micro-cracking"]
            st.subheader("Key Term Frequency in 'content' Column")
            term_counts = {}
            for term in terms_to_search:
                cursor.execute(f"SELECT COUNT(*) FROM papers WHERE content LIKE '%{term}%' AND content IS NOT NULL")
                count = cursor.fetchone()[0]
                term_counts[term] = count
                st.write(f"'{term}': {count} entries")

            query = "SELECT id AS paper_id, title, substr(content, 1, 1000) AS content FROM papers WHERE content IS NOT NULL LIMIT 10"
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
        elif "pages" in available_tables:
            st.subheader("Schema of 'pages' Table")
            cursor.execute("PRAGMA table_info(pages);")
            schema = cursor.fetchall()
            schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
            st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)

            query = "SELECT filename, page_num, substr(text, 1, 200) AS sample_content FROM pages WHERE text IS NOT NULL LIMIT 5"
            df = pd.read_sql_query(query, conn)
            st.subheader("Sample Rows from 'pages' Table (First 5 Pages)")
            if df.empty:
                st.warning("No valid pages found in the 'pages' table.")
            else:
                st.dataframe(df, use_container_width=True)

            cursor.execute("SELECT COUNT(*) as count FROM pages WHERE text IS NOT NULL")
            total_pages = cursor.fetchone()[0]
            st.subheader("Total Valid Pages")
            st.write(f"{total_pages} pages")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='page_fts';")
            if cursor.fetchone():
                search_query = st.text_input("Search pages (FTS5 enabled):", "")
                if search_query:
                    df_search = pd.read_sql_query(
                        "SELECT filename, page_num, text, rank FROM page_fts WHERE page_fts MATCH ? ORDER BY rank LIMIT 50",
                        conn, params=(search_query,)
                    )
                    st.subheader("Search Results")
                    st.dataframe(df_search, use_container_width=True)

            terms_to_search = ["electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress", "micro-cracking"]
            st.subheader("Key Term Frequency in 'text' Column")
            term_counts = {}
            for term in terms_to_search:
                cursor.execute(f"SELECT COUNT(*) FROM pages WHERE text LIKE '%{term}%' AND text IS NOT NULL")
                count = cursor.fetchone()[0]
                term_counts[term] = count
                st.write(f"'{term}': {count} pages")

            query = "SELECT filename, page_num, substr(text, 1, 1000) AS text FROM pages WHERE text IS NOT NULL LIMIT 10"
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
        else:
            st.warning("No 'papers' or 'pages' table found in the database.")
            conn.close()
            return None

        conn.close()
        st.success(f"Database inspection completed for {os.path.basename(db_path)}")
        return term_counts
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        return None

@st.cache_data(hash_funcs={str: lambda x: x})
def categorize_terms(db_file, similarity_threshold=0.7, min_freq=5):
    try:
        update_log(f"Starting term categorization from {os.path.basename(db_file)} with hybrid NLP and MLP classifier")
        conn = sqlite3.connect(db_file)
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            update_log(f"No valid content found in {os.path.basename(db_file)}")
            st.warning(f"No valid content found in {os.path.basename(db_file)}.")
            return {}, Counter()

        update_log(f"Loaded {len(df)} content entries")
        
        # Define categories with expanded seed terms including "damage"
        categories = {
            "Deformation": ["plastic deformation", "yield strength", "plastic strain", "ductility", "inelastic strain", 
                            "flow stress", "volume expansion", "swelling", "volumetric expansion", "volume change", 
                            "dimensional change", "volume deformation", "volume increase", "volumetric strain", 
                            "expansion strain", "dilatation", "volume distortion", "elastic deformation", "shear strain", 
                            "volume swelling", "deformation damage"],
            "Fatigue": ["fatigue life", "cyclic loading", "cycle life", "fatigue crack", "stress cycling", 
                        "mechanical fatigue", "low cycle fatigue", "high cycle fatigue", "fatigue damage", 
                        "fatigue failure", "cyclic deformation", "cyclic stress", "endurance limit", "S-N curve", 
                        "cyclic mechanical damage", "cycle damage"],
            "Crack and Fracture": ["electrode cracking", "crack propagation", "crack growth", "crack initiation", 
                                   "micro-cracking", "fracture", "fracture toughness", "brittle fracture", 
                                   "ductile fracture", "crack branching", "crack tip", "stress intensity factor", 
                                   "fracture mechanics", "J-integral", "cleavage fracture", "microfracture", 
                                   "crack extension", "stress corrosion cracking", "crack damage", "fracture damage"],
            "Degradation": ["SEI formation", "electrolyte degradation", "capacity fade", "cycle degradation", 
                            "electrode degradation", "electrolyte decomposition", "aging", "thermal degradation", 
                            "mechanical degradation", "capacity loss", "fading", "deterioration", "corrosion", 
                            "passivation", "side reaction", "chemical degradation", "electrolyte aging", 
                            "capacity deterioration", "degradation damage"]
        }
        exclude_words = ["et al", "electrochem soc", "phys", "model", "based", "high", "field", 
                         "electrode", "electrolyte", "cathode", "anode", "separator", "current collector", 
                         "phase", "battery"]
        
        # Define regex patterns for hybrid categorization
        category_patterns = {
            "Deformation": re.compile(r'\b(plastic|elastic|volumetric|expansion|strain|deformation|swelling|distortion|change|increase|damage)\b', re.IGNORECASE),
            "Fatigue": re.compile(r'\b(fatigue|cyclic|cycle|stress|loading|damage|failure|deformation)\b', re.IGNORECASE),
            "Crack and Fracture": re.compile(r'\b(crack|fracture|propagation|initiation|growth|branching|tip|mechanics|damage)\b', re.IGNORECASE),
            "Degradation": re.compile(r'\b(degradation|fade|loss|aging|deterioration|corrosion|passivation|reaction|decomposition|formation|damage)\b', re.IGNORECASE)
        }
        
        # Target features for validation
        target_features = categories.copy()
        target_embeddings = {cat: get_scibert_embedding(terms) for cat, terms in target_features.items()}
        target_embeddings = {cat: [emb for emb in embs if emb is not None] for cat, embs in target_embeddings.items()}
        
        # Generate embeddings for seed terms in batch
        all_seed_terms = []
        seed_labels = []
        seed_term_to_cat = {}
        cat_list = list(categories.keys())
        label_encoder = LabelEncoder()
        label_encoder.fit(cat_list)
        
        for cat, terms in categories.items():
            for term in terms:
                all_seed_terms.append(term)
                seed_labels.append(cat)
                seed_term_to_cat[term] = cat
        
        seed_embeddings = get_scibert_embedding(all_seed_terms)
        valid_seed_embeddings = []
        valid_seed_labels = []
        for emb, label in zip(seed_embeddings, seed_labels):
            if emb is not None:
                valid_seed_embeddings.append(emb)
                valid_seed_labels.append(label)
        
        if not valid_seed_embeddings:
            update_log("No valid seed embeddings for MLP training")
            raise ValueError("No valid seed embeddings for MLP training")
        
        X_train = np.array(valid_seed_embeddings)
        y_train = label_encoder.transform(valid_seed_labels)
        
        # Train MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        update_log(f"Trained MLPClassifier with {len(X_train)} seed examples")
        
        seed_embeddings_dict = dict(zip(all_seed_terms, seed_embeddings))
        seed_embeddings = {cat: [seed_embeddings_dict[term] for term in terms if seed_embeddings_dict[term] is not None] for cat, terms in categories.items()}
        
        categorized_terms = {cat: [] for cat in categories}
        uncategorized_terms = []
        term_freqs = Counter()
        
        # Extract all terms
        all_extracted_terms = []
        progress_bar = st.progress(0)
        for i, content in enumerate(df["content"].dropna()):
            if len(content) > nlp.max_length:
                content = content[:nlp.max_length]
                update_log(f"Truncated content for entry {i+1}")
            doc = nlp(content.lower())
            phrases = [span.text.strip() for span in doc.noun_chunks if 1 < len(span.text.split()) <= 3]
            single_words = [token.text for token in doc if token.text.isalpha() and not token.is_stop and len(token.text) > 3]
            words = [token.text for token in doc if token.text.isalpha() and not token.is_stop]
            n_grams = list(chain(ngrams(words, 2), ngrams(words, 3)))
            n_gram_phrases = [' '.join(gram) for gram in n_grams if 1 < len(gram) <= 3]
            all_terms = phrases + n_gram_phrases + single_words
            term_freqs.update(all_terms)
            all_extracted_terms.extend(all_terms)
            progress_bar.progress((i + 1) / len(df) / 2)
        
        # Categorize terms with hybrid approach
        unique_terms = [t for t, f in term_freqs.most_common() if f >= min_freq and not any(w in t.lower() for w in exclude_words)]
        unique_terms = unique_terms[:1000]  # Limit to top 1000 terms
        term_embeddings = get_scibert_embedding(unique_terms)
        term_embeddings_dict = dict(zip(unique_terms, term_embeddings))
        
        regex_bonus = 0.15  # Increased to prioritize regex matches
        mlp_weight = 0.4   # Slightly reduced to balance with regex and embeddings
        for j, term in enumerate(unique_terms):
            term_embedding = term_embeddings_dict.get(term)
            if term_embedding is None:
                update_log(f"Skipping term '{term}' due to missing embedding")
                continue
                
            matched_cats = [cat for cat, pattern in category_patterns.items() if pattern.search(term)]
            
            # MLP prediction
            if term_embedding is not None:
                term_emb_array = np.array([term_embedding])
                mlp_prob = mlp.predict_proba(term_emb_array)[0]
                mlp_pred_cat = label_encoder.inverse_transform([np.argmax(mlp_prob)])[0]
                mlp_score = np.max(mlp_prob)
            else:
                mlp_pred_cat = None
                mlp_score = 0
            
            best_cat = None
            best_score = 0
            for cat, embeddings in seed_embeddings.items():
                cat_bonus = regex_bonus if cat in matched_cats else 0
                emb_scores = []
                for seed_emb in embeddings:
                    if np.linalg.norm(term_embedding) == 0 or np.linalg.norm(seed_emb) == 0:
                        continue
                    score = np.dot(term_embedding, seed_emb) / (np.linalg.norm(term_embedding) * np.linalg.norm(seed_emb))
                    emb_scores.append(score)
                if emb_scores:
                    sim_score = max(emb_scores)
                else:
                    sim_score = 0
                
                mlp_bonus = mlp_weight * mlp_score if mlp_pred_cat == cat else 0
                total_score = sim_score + cat_bonus + mlp_bonus
                
                # Prioritize KEY_TERMS
                if term in KEY_TERMS:
                    total_score += 0.2  # Boost for key terms
                
                if total_score > similarity_threshold and total_score > best_score:
                    best_cat = cat
                    best_score = total_score
            
            if not best_cat:
                if matched_cats:
                    best_cat = matched_cats[0]
                    best_score = 0.8 + (mlp_weight * mlp_score if mlp_pred_cat == best_cat else 0)
                elif mlp_pred_cat and mlp_score > 0.7:
                    best_cat = mlp_pred_cat
                    best_score = mlp_score
            
            if best_cat:
                categorized_terms[best_cat].append((term, term_freqs[term], best_score))
                update_log(f"Assigned term '{term}' to category '{best_cat}' with score {best_score:.2f}")
            else:
                uncategorized_terms.append((term, term_freqs[term], 0))
                update_log(f"Term '{term}' not categorized (score={best_score:.2f})")
            
            progress_bar.progress(0.5 + (j + 1) / len(unique_terms) / 2)
        
        # Sort terms per category by frequency
        for cat in categorized_terms:
            categorized_terms[cat] = sorted(categorized_terms[cat], key=lambda x: x[1], reverse=True)
        
        # Validation and refinement step
        validation_threshold = 0.75  # Lowered from 0.85
        relevance_threshold = 0.5    # Lowered from 0.6
        refined_categorized_terms = {cat: [] for cat in categories}
        for cat, terms in categorized_terms.items():
            target_embs = target_embeddings.get(cat, [])
            if not target_embs:
                refined_categorized_terms[cat] = terms[:20]
                update_log(f"No target embeddings for category '{cat}', keeping top 20 terms")
                continue
            for term, freq, score in terms:
                term_emb = term_embeddings_dict.get(term)
                if term_emb is None:
                    if term in KEY_TERMS:
                        refined_categorized_terms[cat].append((term, freq, score))
                        update_log(f"Retained key term '{term}' in category '{cat}' despite missing embedding")
                    else:
                        update_log(f"Skipping term '{term}' in category '{cat}' due to missing embedding")
                    continue
                similarities = [np.dot(term_emb, t_emb) / (np.linalg.norm(term_emb) * np.linalg.norm(t_emb))
                                for t_emb in target_embs if np.linalg.norm(term_emb) != 0 and np.linalg.norm(t_emb) != 0]
                max_target_sim = max(similarities + [0])
                avg_target_sim = np.mean(similarities) if similarities else 0
                if (max_target_sim >= validation_threshold or score >= validation_threshold or term in KEY_TERMS) and avg_target_sim >= relevance_threshold:
                    refined_categorized_terms[cat].append((term, freq, score))
                else:
                    update_log(f"Term '{term}' in category '{cat}' did not meet validation thresholds (max_sim={max_target_sim:.2f}, avg_sim={avg_target_sim:.2f}, score={score:.2f})")
            refined_categorized_terms[cat] = refined_categorized_terms[cat][:20]
        
        # Log retention statistics
        for cat, terms in refined_categorized_terms.items():
            update_log(f"Category '{cat}' retained {len(terms)} terms after validation")
        
        update_log(f"Refined {sum(len(terms) for terms in refined_categorized_terms.values())} terms across {len(refined_categorized_terms)} categories after validation")
        return refined_categorized_terms, term_freqs
    except Exception as e:
        update_log(f"Error categorizing terms: {str(e)}")
        st.error(f"Error categorizing terms: {str(e)}")
        return {}, Counter()

def process_content_chunk(content, terms_pattern):
    content_lower = content.lower()
    found_terms = set(terms_pattern.findall(content_lower))
    return found_terms

@st.cache_data(hash_funcs={str: lambda x: x})
def build_knowledge_graph_data(categorized_terms, db_file, min_co_occurrence=5, top_n=10):
    try:
        update_log(f"Building knowledge graph data for top {top_n} terms per category")
        start_time = time.time()
        
        conn = sqlite3.connect(db_file)
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Aggressive sampling for large datasets
        if len(df) > 500:
            df = df.sample(frac=0.1, random_state=42)
            update_log(f"Sampled 10% of data for efficiency: {len(df)} rows")

        G = nx.Graph()
        
        # Add category nodes
        for cat in categorized_terms.keys():
            G.add_node(cat, type="category", freq=0, size=2000, color="skyblue")
        
        # Add terms as nodes (limited to top_n per category)
        term_to_category = {}
        all_terms = []
        term_freqs = {}
        term_embeddings = {}
        
        for cat, terms in categorized_terms.items():
            terms = terms[:top_n]  # Limit to top_n terms per category
            term_texts = [term for term, _, _ in terms]
            embeddings = get_scibert_embedding(term_texts)
            max_freq = max([f for _, f, _ in terms], default=1)
            
            for (term, freq, score), emb in zip(terms, embeddings):
                if emb is None:
                    continue
                term_embeddings[term] = emb
                size = 1000 + 3000 * (freq / max_freq) if term in KEY_TERMS else 500 + 2000 * (freq / max_freq)
                color = "gold" if term in KEY_TERMS else "salmon"
                G.add_node(term, type="term", freq=freq, category=cat, unit=term_units.get(cat, "None"),
                          size=size, color=color, score=score)
                G.add_edge(cat, term, weight=1.0, type="category-term", label="belongs_to")
                term_freqs[term] = freq
                term_to_category[term] = cat
                all_terms.append(term)

        # Timeout check
        if time.time() - start_time > 300:
            update_log("Knowledge graph construction timed out after 5 minutes")
            st.warning("Knowledge graph construction timed out. Try increasing min_co_occurrence or reducing top_n.")
            return None, None

        # Create regex pattern for terms
        terms_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, all_terms)) + r')\b', re.IGNORECASE)
        
        # Compute co-occurrences
        co_occurrence_counts = defaultdict(lambda: defaultdict(int))
        progress_bar = st.progress(0)
        with Pool(processes=8) as pool:
            results = pool.starmap(
                process_content_chunk,
                [(content, terms_pattern) for content in df["content"].values]
            )
            
            for i, found_terms in enumerate(results):
                if i % 10 == 0:
                    progress_bar.progress(min(1.0, i / len(df)))
                if found_terms:
                    terms_list = list(found_terms)
                    for term1, term2 in combinations(terms_list, 2):
                        co_occurrence_counts[term1][term2] += 1
                        co_occurrence_counts[term2][term1] += 1
        
        # Add co-occurrence edges
        for term1, related_terms in co_occurrence_counts.items():
            for term2, count in related_terms.items():
                if count >= min_co_occurrence and term1 in G.nodes and term2 in G.nodes:
                    cat1 = term_to_category.get(term1, "Unknown")
                    cat2 = term_to_category.get(term2, "Unknown")
                    rel_type = "intra_category" if cat1 == cat2 else "inter_category"
                    label = f"co-occurs_with ({count})" if cat1 == cat2 else f"related_to ({count})"
                    
                    term1_embedding = term_embeddings.get(term1)
                    term2_embedding = term_embeddings.get(term2)
                    
                    if term1_embedding is not None and term2_embedding is not None:
                        key_term_sim1 = max([cosine_similarity([term1_embedding], [kt_emb])[0][0] for kt_emb in KEY_TERMS_EMBEDDINGS] + [0])
                        key_term_sim2 = max([cosine_similarity([term2_embedding], [kt_emb])[0][0] for kt_emb in KEY_TERMS_EMBEDDINGS] + [0])
                        weight = count * (1 + 0.5 * (key_term_sim1 + key_term_sim2))
                    else:
                        weight = count
                    
                    G.add_edge(term1, term2, weight=weight, type="term-term", 
                              relationship=rel_type, label=label, strength=count)

        # Add hierarchical relationships only for high-frequency terms
        for term in list(G.nodes):
            if G.nodes[term].get("type") == "term" and term_freqs.get(term, 0) >= min_co_occurrence:
                for potential_parent in list(G.nodes):
                    if (G.nodes[potential_parent].get("type") == "term" and 
                        potential_parent != term and 
                        len(potential_parent) < len(term) and
                        potential_parent in term.lower() and
                        term_freqs.get(potential_parent, 0) >= min_co_occurrence):
                        G.add_edge(potential_parent, term, weight=2.0, 
                                  type="hierarchical", label="is_part_of", strength=2.0)
        
        # Generate DataFrames for nodes and edges
        nodes_df = pd.DataFrame([(n, d["type"], d.get("category", ""), d.get("freq", 0), 
                                d.get("unit", "None"), d.get("score", 0)) 
                               for n, d in G.nodes(data=True)], 
                              columns=["node", "type", "category", "frequency", "unit", "similarity_score"])
        
        edges_df = pd.DataFrame([(u, v, d["weight"], d["type"], d.get("label", ""), 
                                d.get("relationship", ""), d.get("strength", 0)) 
                               for u, v, d in G.edges(data=True)], 
                              columns=["source", "target", "weight", "type", "label", "relationship", "strength"])
        
        nodes_csv_filename = f"knowledge_graph_nodes_{uuid.uuid4().hex}.csv"
        edges_csv_filename = f"knowledge_graph_edges_{uuid.uuid4().hex}.csv"
        nodes_csv_path = os.path.join(DB_DIR, nodes_csv_filename)
        edges_csv_path = os.path.join(DB_DIR, edges_csv_filename)
        nodes_df.to_csv(nodes_csv_path, index=False)
        edges_df.to_csv(edges_csv_path, index=False)
        
        with open(nodes_csv_path, "rb") as f:
            nodes_csv_data = f.read()
        with open(edges_csv_path, "rb") as f:
            edges_csv_data = f.read()
        
        st.session_state.knowledge_graph = G
        
        update_log(f"Knowledge graph built in {time.time() - start_time:.2f} seconds with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G, (nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename)
    
    except Exception as e:
        update_log(f"Error building knowledge graph data: {str(e)}")
        st.error(f"Error building knowledge graph data: {str(e)}")
        return None, None

def enhance_ner_with_knowledge_graph(ner_df, knowledge_graph):
    if ner_df.empty or knowledge_graph is None:
        return ner_df
    
    enhanced_ner = []
    
    for _, row in ner_df.iterrows():
        entity_text = row["entity_text"]
        entity_label = row["entity_label"]
        
        if entity_text in knowledge_graph.nodes:
            neighbors = list(knowledge_graph.neighbors(entity_text))
            strong_connections = []
            for neighbor in neighbors:
                edge_data = knowledge_graph.get_edge_data(entity_text, neighbor)
                if edge_data and edge_data.get("strength", 0) > 3:
                    strong_connections.append((neighbor, edge_data.get("strength", 0)))
            
            strong_connections.sort(key=lambda x: x[1], reverse=True)
            context_enhanced = row["context"]
            if strong_connections:
                related_terms = ", ".join([f"{term}({strength})" for term, strength in strong_connections[:3]])
                context_enhanced += f" [KG: Related to {related_terms}]"
            
            enhanced_row = row.to_dict()
            enhanced_row["context"] = context_enhanced
            enhanced_row["kg_related_terms"] = ", ".join([term for term, _ in strong_connections])
            enhanced_row["kg_connection_strength"] = sum([strength for _, strength in strong_connections]) / max(1, len(strong_connections))
            
            enhanced_ner.append(enhanced_row)
        else:
            enhanced_ner.append(row.to_dict())
    
    return pd.DataFrame(enhanced_ner)

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
            "Deformation": ["plastic deformation", "yield strength", "plastic strain", "ductility", "inelastic strain", 
                            "flow stress", "volume expansion", "swelling", "volumetric expansion", "volume change", 
                            "dimensional change", "volume deformation", "volume increase", "volumetric strain", 
                            "expansion strain", "dilatation", "volume distortion", "elastic deformation", "shear strain", 
                            "volume swelling", "deformation damage"],
            "Fatigue": ["fatigue life", "cyclic loading", "cycle life", "fatigue crack", "stress cycling", 
                        "mechanical fatigue", "low cycle fatigue", "high cycle fatigue", "fatigue damage", 
                        "fatigue failure", "cyclic deformation", "cyclic stress", "endurance limit", "S-N curve", 
                        "cyclic mechanical damage", "cycle damage"],
            "Crack and Fracture": ["electrode cracking", "crack propagation", "crack growth", "crack initiation", 
                                   "micro-cracking", "fracture", "fracture toughness", "brittle fracture", 
                                   "ductile fracture", "crack branching", "crack tip", "stress intensity factor", 
                                   "fracture mechanics", "J-integral", "cleavage fracture", "microfracture", 
                                   "crack extension", "stress corrosion cracking", "crack damage", "fracture damage"],
            "Degradation": ["SEI formation", "electrolyte degradation", "capacity fade", "cycle degradation", 
                            "electrode degradation", "electrolyte decomposition", "aging", "thermal degradation", 
                            "mechanical degradation", "capacity loss", "fading", "deterioration", "corrosion", 
                            "passivation", "side reaction", "chemical degradation", "electrolyte aging", 
                            "capacity deterioration", "degradation damage"]
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
        ref_embeddings = {cat: get_scibert_embedding(terms) for cat, terms in categories.items()}
        ref_embeddings = {cat: [emb for embs in ref_embeddings.values() for emb in embs if emb is not None] for cat in categories}
        term_patterns = {term: re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE) for term in selected_terms}
        
        entities = []
        entity_set = set()
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            try:
                text = row["content"].lower()
                text = re.sub(r"young's modulus|youngs modulus", "young’s modulus", text)
                if len(text) > nlp.max_length:
                    text = text[:nlp.max_length]
                    update_log(f"Truncated content for entry {row['paper_id']}")
                if not text.strip() or len(text) < 10:
                    update_log(f"Skipping entry {row['paper_id']} due to empty/short content")
                    continue
                doc = nlp(text)
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
                    update_log(f"No valid spans in entry {row['paper_id']}")
                    continue
                for span, orig_sent, nearby_sent in spans:
                    span_text = span.text.lower().strip()
                    if not span_text:
                        update_log(f"Skipping empty span in entry {row['paper_id']}")
                        continue
                    term_matched = False
                    for term in selected_terms:
                        if term_patterns[term].search(span_text) or term_patterns[term].search(orig_sent) or term_patterns[term].search(nearby_sent):
                            term_matched = True
                            break
                    if not term_matched:
                        span_embedding = get_scibert_embedding(span_text)
                        if span_embedding is None:
                            continue
                        term_embeddings = get_scibert_embedding(selected_terms)
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
                                value /= 1_000_000
                            elif unit == "MPA·M^0.5":
                                unit = "MPa·m^0.5"
                            elif unit == "CYCLES":
                                unit = "cycles"
                        except ValueError:
                            continue
                    span_embedding = get_scibert_embedding(span_text)
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
                            context_embedding = get_scibert_embedding(context)
                            if context_embedding is None:
                                continue
                            unit_valid = False
                            for v_unit in valid_units.get(best_label, []):
                                unit_embedding = get_scibert_embedding(f"{span_text} {v_unit}")
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
            enhanced_entities = enhance_ner_with_knowledge_graph(entities_df, st.session_state.knowledge_graph)
            return enhanced_entities
        else:
            return pd.DataFrame(entities)
    except Exception as e:
        update_log(f"NER analysis failed: {str(e)}")
        st.error(f"NER analysis failed: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def plot_word_cloud(terms, top_n, font_size, font_type, colormap):
    term_dict = {term: freq for term, freq, _ in terms[:top_n]}
    font_path = None
    if font_type and font_type != "None":
        font_map = {'DejaVu Sans': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'}
        font_path = font_map.get(font_type, font_type)
        if not os.path.exists(font_path):
            update_log(f"Font path '{font_path}' not found")
            font_path = None
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", min_font_size=8, max_font_size=font_size,
        font_path=font_path, colormap=colormap, max_words=top_n, prefer_horizontal=0.9
    ).generate_from_frequencies(term_dict)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud of Top {top_n} Terms")
    plt.tight_layout()
    return fig

@st.cache_data
def plot_combined_word_cloud(categorized_terms, top_n, font_size, font_type):
    term_dict = {}
    category_colors = {
        "Deformation": "red",
        "Fatigue": "blue",
        "Crack and Fracture": "green",
        "Degradation": "purple"
    }
    term_to_color = {}
    for cat, terms in categorized_terms.items():
        for term, freq, _ in terms[:top_n]:
            if term not in term_dict:  # Avoid duplicates across categories
                term_dict[term] = freq
                term_to_color[term] = category_colors.get(cat, "black")
    
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return term_to_color.get(word, "black")
    
    font_path = None
    if font_type and font_type != "None":
        font_map = {'DejaVu Sans': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'}
        font_path = font_map.get(font_type, font_type)
        if not os.path.exists(font_path):
            update_log(f"Font path '{font_path}' not found")
            font_path = None
    
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", min_font_size=8, max_font_size=font_size,
        font_path=font_path, max_words=len(term_dict), prefer_horizontal=0.9,
        color_func=color_func
    ).generate_from_frequencies(term_dict)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Combined Word Cloud of Top Terms")
    
    # Add legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat) for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, title="Categories", loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    return fig

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

# Database selection
st.header("Select or Upload Database")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files if f in [RELIABILITY_DB_FILE, UNIVERSE_DB_FILE]] + ["Upload a new .db file"]
if os.path.basename(UNIVERSE_DB_FILE) not in db_options and os.path.exists(UNIVERSE_DB_FILE):
    db_options.insert(0, os.path.basename(UNIVERSE_DB_FILE))
default_index = db_options.index(os.path.basename(UNIVERSE_DB_FILE)) if os.path.basename(UNIVERSE_DB_FILE) in db_options else 0
db_selection = st.selectbox("Select Database", db_options, index=default_index, key="db_select")
uploaded_file = None
if db_selection == "Upload a new .db file":
    uploaded_file = st.file_uploader("Upload SQLite Database (.db)", type=["db"], key="db_upload")
    if uploaded_file:
        temp_db_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.db_file = temp_db_path
        update_log(f"Uploaded database saved as {temp_db_path}")
else:
    if db_selection == os.path.basename(RELIABILITY_DB_FILE):
        st.session_state.db_file = RELIABILITY_DB_FILE
    elif db_selection == os.path.basename(UNIVERSE_DB_FILE):
        st.session_state.db_file = UNIVERSE_DB_FILE
    else:
        st.session_state.db_file = os.path.join(DB_DIR, db_selection)
    update_log(f"Selected database: {db_selection}")

# Main app logic
if st.session_state.db_file and os.path.exists(st.session_state.db_file):
    tab1, tab2, tab3, tab4 = st.tabs(["Database Inspection", "Term Categorization", "Knowledge Graph", "NER Analysis"])
    with tab1:
        st.header("Database Inspection")
        if st.button("Inspect Database", key="inspect_button"):
            with st.spinner(f"Inspecting {os.path.basename(st.session_state.db_file)}..."):
                st.session_state.term_counts = inspect_database(st.session_state.db_file)
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="inspection_logs")
    
    with tab2:
        st.header("Term Categorization")
        analyze_terms_button = st.button("Categorize Terms", key="categorize_terms")
        with st.sidebar:
            st.subheader("Analysis Parameters")
            exclude_words = [w.strip().lower() for w in st.text_input("Exclude Words/Phrases (comma-separated)", 
                                                                      value="et al, electrochem soc, phys, electrode, electrolyte, cathode, anode, separator, current collector, phase, battery", 
                                                                      key="exclude_words").split(",") if w.strip()]
            similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="similarity_threshold")
            min_freq = st.slider("Minimum Frequency", min_value=1, max_value=20, value=5, key="min_freq")
            top_n = st.slider("Number of Top Terms", min_value=5, max_value=20, value=10, key="top_n")
            wordcloud_font_size = st.slider("Word Cloud Font Size", min_value=20, max_value=80, value=40, key="wordcloud_font_size")
            font_type = st.selectbox("Font Type", ["None", "DejaVu Sans"], key="font_type")
            colormap = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "hot", "cool", "rainbow"], key="colormap")
        
        if analyze_terms_button:
            with st.spinner(f"Categorizing terms from {os.path.basename(st.session_state.db_file)}..."):
                st.session_state.categorized_terms, st.session_state.term_counts = categorize_terms(st.session_state.db_file, similarity_threshold, min_freq)
        
        if st.session_state.categorized_terms:
            filtered_terms = {cat: [(t, f, s) for t, f, s in terms if not any(w in t.lower() for w in exclude_words)] for cat, terms in st.session_state.categorized_terms.items()}
            if not any(filtered_terms.values()):
                st.warning("No terms remain after applying exclude words.")
            else:
                st.success(f"Categorized terms into {len(filtered_terms)} categories!")
                for cat, terms in filtered_terms.items():
                    if terms:
                        st.subheader(f"{cat} Terms")
                        term_df = pd.DataFrame(terms, columns=["Term/Phrase", "Frequency", "Similarity Score"])
                        st.dataframe(term_df, use_container_width=True)
                        csv_data = term_df.to_csv(index=False)
                        st.download_button(f"Download {cat} Terms CSV", csv_data, f"{cat.lower()}_terms.csv", "text/csv", key=f"download_{cat.lower()}")
                        fig = plot_word_cloud(terms, top_n, wordcloud_font_size, font_type, colormap)
                        if fig:
                            st.pyplot(fig)
                
                # Combined word cloud
                st.subheader("Combined Word Cloud with Category-Based Colors")
                fig_combined = plot_combined_word_cloud(filtered_terms, top_n, wordcloud_font_size, font_type)
                if fig_combined:
                    st.pyplot(fig_combined)
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="categorize_logs")
    
    with tab3:
        st.header("Knowledge Graph")
        if st.button("Build Knowledge Graph", key="build_graph"):
            if st.session_state.categorized_terms is None:
                st.warning("No categorized terms found. Please run 'Categorize Terms' in the Term Categorization tab first.")
                update_log("Attempted to build knowledge graph without categorized terms. User must run Term Categorization first.")
            else:
                with st.spinner("Building knowledge graph..."):
                    G, csv_data = build_knowledge_graph_data(st.session_state.categorized_terms, st.session_state.db_file, min_co_occurrence=min_freq, top_n=top_n)
                    if G and G.edges():
                        fig, ax = plt.subplots(figsize=(12, 12))
                        pos = nx.kamada_kawai_layout(G, scale=2.0)
                        if st.session_state.categorized_terms:
                            for cat in st.session_state.categorized_terms.keys():
                                if G.has_node(cat):
                                    pos[cat][1] += 2
                        node_colors = []
                        node_sizes = []
                        for node in G.nodes:
                            if G.nodes[node]["type"] == "category":
                                node_colors.append("skyblue")
                                node_sizes.append(2000)
                            else:
                                node_colors.append(G.nodes[node].get("color", "salmon"))
                                node_sizes.append(G.nodes[node].get("size", 500))
                        
                        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
                        
                        category_term_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "category-term"]
                        term_term_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "term-term"]
                        hierarchical_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "hierarchical"]
                        
                        nx.draw_networkx_edges(G, pos, edgelist=category_term_edges, width=2.0, edge_color="blue", ax=ax)
                        nx.draw_networkx_edges(G, pos, edgelist=term_term_edges, width=1.0, edge_color="gray", ax=ax)
                        nx.draw_networkx_edges(G, pos, edgelist=hierarchical_edges, width=1.5, edge_color="green", style="dashed", ax=ax)
                        
                        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                        
                        edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
                        
                        ax.set_title("PREKNOWLEDGE: Knowledge Graph of Battery Reliability Phenomena")
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        st.subheader("Community Detection")
                        try:
                            communities = community_louvain.best_partition(G)
                            if not communities or max(communities.values(), default=0) == 0:
                                st.warning("Community detection failed to produce meaningful communities. Displaying graph with default coloring.")
                                update_log("Community detection produced no valid communities (empty or all zeros). Using default coloring.")
                                community_colors = ["skyblue" for _ in G.nodes()]  # Fallback to a single color
                            else:
                                max_community = max(communities.values())
                                community_colors = [cm.get_cmap("viridis")(communities[node] / max_community) for node in G.nodes()]
                            fig_comm, ax_comm = plt.subplots(figsize=(12, 12))
                            nx.draw_networkx_nodes(G, pos, node_color=community_colors, node_size=node_sizes, alpha=0.8, ax=ax_comm)
                            nx.draw_networkx_edges(G, pos, edgelist=category_term_edges, width=2.0, edge_color="blue", ax=ax_comm)
                            nx.draw_networkx_edges(G, pos, edgelist=term_term_edges, width=1.0, edge_color="gray", ax=ax_comm)
                            nx.draw_networkx_edges(G, pos, edgelist=hierarchical_edges, width=1.5, edge_color="green", style="dashed", ax=ax_comm)
                            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_comm)
                            ax_comm.set_title("Community Detection in Knowledge Graph")
                            plt.tight_layout()
                            st.pyplot(fig_comm)
                        except Exception as e:
                            st.error(f"Error in community detection: {str(e)}")
                            update_log(f"Community detection failed: {str(e)}")
                        
                        st.subheader("Focused Subgraph on Key Terms")
                        key_nodes = [node for node in G.nodes if node in KEY_TERMS]
                        neighbors = set()
                        for node in key_nodes:
                            neighbors.update(nx.neighbors(G, node))
                        focused_nodes = set(key_nodes) | neighbors
                        sub_G = G.subgraph(focused_nodes)
                        
                        if sub_G.edges():
                            fig_sub, ax_sub = plt.subplots(figsize=(10, 8))
                            pos_sub = nx.kamada_kawai_layout(sub_G, scale=2.0)
                            node_colors_sub = [G.nodes[node].get("color", "salmon") for node in sub_G.nodes()]
                            node_sizes_sub = [G.nodes[node].get("size", 500) for node in sub_G.nodes()]
                            nx.draw_networkx_nodes(sub_G, pos_sub, node_color=node_colors_sub, node_size=node_sizes_sub, alpha=0.8, ax=ax_sub)
                            edge_widths_sub = [1.5 * d.get("weight", 1) for _, _, d in sub_G.edges(data=True)]
                            nx.draw_networkx_edges(sub_G, pos_sub, width=edge_widths_sub, alpha=0.5, ax=ax_sub)
                            nx.draw_networkx_labels(sub_G, pos_sub, font_size=8, ax=ax_sub)
                            ax_sub.set_title("Focused Subgraph on Key Terms")
                            plt.tight_layout()
                            st.pyplot(fig_sub)
                        
                        nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename = csv_data
                        st.download_button(
                            label="Download Knowledge Graph Nodes",
                            data=nodes_csv_data,
                            file_name="knowledge_graph_nodes.csv",
                            mime="text/csv",
                            key="download_graph_nodes"
                        )
                        st.download_button(
                            label="Download Knowledge Graph Edges",
                            data=edges_csv_data,
                            file_name="knowledge_graph_edges.csv",
                            mime="text/csv",
                            key="download_graph_edges"
                        )
                    else:
                        st.warning("No knowledge graph generated. Check logs for details.")
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="graph_logs")
    
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
            default_terms = [term for term in ["electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress", "micro-cracking"] if term in available_terms]
            selected_terms = st.multiselect("Select Terms for NER", available_terms, default_terms, key="select_terms")
            
            use_kg_enhancement = st.checkbox("Use Knowledge Graph for NER Enhancement", value=True, 
                                           help="Use knowledge graph relationships to improve NER results")
            
            if st.button("Run NER Analysis", key="ner_analyze"):
                if not selected_terms:
                    st.warning("Select at least one term for NER analysis.")
                else:
                    with st.spinner(f"Processing NER analysis for {len(selected_terms)} terms..."):
                        ner_df = perform_ner_on_terms(st.session_state.db_file, selected_terms)
                        st.session_state.ner_results = ner_df
                    if ner_df.empty:
                        st.warning("No entities were found.")
                    else:
                        st.success(f"Extracted {len(ner_df)} entities!")
                        
                        if use_kg_enhancement and "kg_related_terms" in ner_df.columns:
                            st.subheader("NER Results Enhanced with Knowledge Graph")
                            st.dataframe(
                                ner_df[["paper_id", "title", "entity_text", "entity_label", "value", "unit", "kg_related_terms", "kg_connection_strength"]].head(100),
                                use_container_width=True
                            )
                        else:
                            st.dataframe(
                                ner_df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "context", "score"]].head(100),
                                use_container_width=True
                            )
                        
                        ner_csv = ner_df.to_csv(index=False)
                        st.download_button("Download NER Data CSV", ner_csv, "ner_data.csv", "text/csv", key="download_ner")
                        st.subheader("NER Visualizations")
                        fig_hist = plot_ner_histogram(ner_df, top_n, colormap)
                        if fig_hist:
                            st.pyplot(fig_hist)
                        fig_box = plot_ner_value_boxplot(ner_df, top_n, colormap)
                        if fig_box:
                            st.pyplot(fig_box)
                        categories_units = {
                            "Deformation": "%",
                            "Fatigue": "cycles",
                            "Crack and Fracture": "μm",
                            "Degradation": "%"
                        }
                        figs_hist_values = plot_ner_value_histograms(ner_df, categories_units, top_n, colormap)
                        if figs_hist_values:
                            st.subheader("Value Distribution Histograms")
                            for fig in figs_hist_values:
                                st.pyplot(fig)
                        else:
                            st.warning("No numerical values available for histogram plotting.")
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="ner_logs")
else:
    st.warning(f"Select or upload a database file. Ensure {os.path.basename(UNIVERSE_DB_FILE)} is available for full-text analysis.")

# Notes
st.markdown("""
---
**Notes**
- **PREKNOWLEDGE Overview**: Prioritizes key terms (e.g., electrode cracking, SEI formation, cyclic mechanical damage) in the knowledge graph, optimized for speed with limited terms and efficient co-occurrence detection.
- **Database Inspection**: View tables, schemas, and sample data from `battery_reliability.db` or `battery_reliability_universe.db`.
- **Term Categorization**: Groups terms into Deformation, Fatigue, Crack and Fracture, and Degradation using hybrid regex patterns, SciBERT embeddings, and MLPClassifier. Excludes component terms (e.g., electrode, electrolyte) and uncategorized terms.
- **Knowledge Graph**: Visualizes relationships with a focus on key terms, optimized with fewer nodes and faster layout (kamada_kawai).
- **NER Analysis**: Extracts entities with context-aware unit assignment, enhanced with knowledge graph relationships.
- **Word Cloud**: Combined word cloud uses distinct colors for each category (red: Deformation, blue: Fatigue, green: Crack and Fracture, purple: Degradation) with a legend.
- Place `battery_reliability_universe.db` in the script's directory or upload a custom .db file. Ensure the database has a `papers` table with a `content` column.
- Check `battery_reliability_analysis.log` for detailed logs, including term categorization decisions.
""")