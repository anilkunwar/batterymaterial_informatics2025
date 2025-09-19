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
from community import community_louvain  # For community detection

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
st.set_page_config(page_title="Battery Reliability Analysis Tool (SciBERT)", layout="wide")
st.title("Battery Reliability Analysis: Electrode Cracking, SEI Formation, Degradation")
st.markdown("""
This tool inspects SQLite databases (`battery_reliability.db` and `battery_reliability_universe.db`), categorizes terms related to battery reliability challenges (e.g., electrode cracking, SEI formation, cyclic mechanical damage, electrolyte degradation, capacity fade), builds a knowledge graph, and performs NER analysis using SciBERT. The default database is `battery_reliability_universe.db` for full-text analysis of arXiv papers.
Select a database, then then use the tabs to inspect the database, categorize terms, visualize relationships, or extract entities with numerical values.
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
    st.session_state.db_file = UNIVERSE_DB_FILE if os.path.exists(UNIVERSE_DB_FILE) else None  # Default to battery_reliability_universe.db if it exists
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
def get_scibert_embedding(text):
    try:
        if not text.strip():
            update_log(f"Skipping empty text for SciBERT embedding")
            return None
        inputs = scibert_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
        norm = np.linalg.norm(last_hidden_state)
        if norm == 0:
            update_log(f"Zero norm for embedding of '{text}'")
            return None
        return last_hidden_state / norm
    except Exception as e:
        update_log(f"SciBERT embedding failed for '{text}': {str(e)}")
        return None

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

# Precompute embeddings for key terms
KEY_TERMS_EMBEDDINGS = [get_scibert_embedding(term) for term in KEY_TERMS if get_scibert_embedding(term) is not None]

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

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pages';")
        if cursor.fetchone():
            st.subheader("Schema of 'pages' Table")
            cursor.execute("PRAGMA table_info(pages);")
            schema = cursor.fetchall()
            schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
            st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)

            query = "SELECT filename, page_num, substr(text, 1, 200) as sample_content FROM pages WHERE text IS NOT NULL LIMIT 5"
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

            query = "SELECT filename, page_num, substr(text, 1, 1000) as text FROM pages WHERE text IS NOT NULL LIMIT 10"
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
            conn.close()
            st.success(f"Database inspection completed for {os.path.basename(db_path)}")
            return term_counts
        else:
            st.warning("No 'pages' table found in the database.")
            conn.close()
            return None
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        return None

@st.cache_data(hash_funcs={str: lambda x: x})
def categorize_terms(db_file, similarity_threshold=0.7, min_freq=5):
    try:
        update_log(f"Starting term categorization from {os.path.basename(db_file)}")
        conn = sqlite3.connect(db_file)
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            update_log(f"No valid content found in {os.path.basename(db_file)}")
            st.warning(f"No valid content found in {os.path.basename(db_file)}.")
            return {}, Counter()

        update_log(f"Loaded {len(df)} content entries")
        categories = {
            "Deformation": ["plastic deformation", "yield strength", "plastic strain", "ductility", "inelastic strain", "flow stress", "volume expansion", "swelling", "volumetric expansion", "volume change", "dimensional change", "volume deformation", "volume increase", "volumetric strain", "expansion strain", "dilatation", "volume distortion"],
            "Fatigue": ["fatigue life", "cyclic loading", "cycle life", "fatigue crack", "stress cycling", "mechanical fatigue", "low cycle fatigue", "high cycle fatigue", "fatigue damage", "fatigue failure", "cyclic deformation", "cyclic stress", "endurance limit", "S-N curve"],
            "Crack and Fracture": ["electrode cracking", "crack propagation", "crack growth", "crack initiation", "micro-cracking", "fracture", "fracture toughness", "brittle fracture", "ductile fracture", "crack branching", "crack tip", "stress intensity factor", "fracture mechanics", "J-integral", "cleavage fracture"],
            "Degradation": ["SEI formation", "electrolyte degradation", "capacity fade", "cycle degradation", "electrode degradation", "electrolyte decomposition", "aging", "thermal degradation", "mechanical degradation", "capacity loss", "fading", "deterioration", "corrosion", "passivation", "side reaction"]
        }
        component_terms = ["electrode", "electrolyte", "phase", "cathode", "anode", "solid electrolyte", "separator", "current collector"]
        exclude_words = ["et al", "electrochem soc", "phys", "model", "based", "high", "field"]
        
        # Generate embeddings for seed terms
        seed_embeddings = {cat: [get_scibert_embedding(term) for term in terms if get_scibert_embedding(term) is not None] for cat, terms in categories.items()}
        component_embeddings = [get_scibert_embedding(term) for term in component_terms if get_scibert_embedding(term) is not None]
        
        categorized_terms = {cat: [] for cat in categories}
        categorized_terms["Component"] = []
        other_terms = []
        term_freqs = Counter()
        
        # Extract all terms first
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
        
        # Categorize terms
        for term in set(all_extracted_terms):
            if any(w in term.lower() for w in exclude_words):
                continue
            if term_freqs[term] < min_freq:
                continue
                
            term_embedding = get_scibert_embedding(term)
            if term_embedding is None:
                continue
                
            best_cat = None
            best_score = 0
            for cat, embeddings in seed_embeddings.items():
                for seed_emb in embeddings:
                    if np.linalg.norm(term_embedding) == 0 or np.linalg.norm(seed_emb) == 0:
                        continue
                    score = np.dot(term_embedding, seed_emb) / (np.linalg.norm(term_embedding) * np.linalg.norm(seed_emb))
                    if score > similarity_threshold and score > best_score:
                        best_cat = cat
                        best_score = score
            
            if best_cat:
                categorized_terms[best_cat].append((term, term_freqs[term], best_score))
            elif any(np.dot(term_embedding, comp_emb) / (np.linalg.norm(term_embedding) * np.linalg.norm(comp_emb)) > similarity_threshold for comp_emb in component_embeddings if np.linalg.norm(term_embedding) != 0 and np.linalg.norm(comp_emb) != 0):
                categorized_terms["Component"].append((term, term_freqs[term], best_score))
            else:
                other_terms.append((term, term_freqs[term], best_score))
            
            progress_bar.progress(0.5 + (i + 1) / len(df) / 2)
        
        # Sort terms by frequency within each category
        for cat in categorized_terms:
            categorized_terms[cat] = sorted(categorized_terms[cat], key=lambda x: x[1], reverse=True)
        categorized_terms["Other"] = sorted(other_terms, key=lambda x: x[1], reverse=True)[:50]  # Limit other terms to top 50
        
        update_log(f"Categorized {sum(len(terms) for terms in categorized_terms.values())} terms across {len(categorized_terms)} categories")
        return categorized_terms, term_freqs
    except Exception as e:
        update_log(f"Error categorizing terms: {str(e)}")
        st.error(f"Error categorizing terms: {str(e)}")
        return {}, Counter()

@st.cache_data(hash_funcs={str: lambda x: x})
def build_knowledge_graph_data(categorized_terms, db_file, min_co_occurrence=2, top_n=10):
    try:
        update_log(f"Building knowledge graph data for top {top_n} terms per category")
        conn = sqlite3.connect(db_file)
        query = "SELECT content FROM papers WHERE content IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()

        G = nx.Graph()
        
        # Add category nodes with enhanced attributes
        categories = list(categorized_terms.keys())
        for cat in categories:
            G.add_node(cat, type="category", freq=0, size=2000, color="skyblue")
        
        # Add all terms as nodes with proper categorization
        term_to_category = {}
        for cat, terms in categorized_terms.items():
            for term, freq, score in terms:
                term_to_category[term] = cat
        
        # Add all terms as nodes with enhanced attributes
        term_freqs = {}
        for cat, terms in categorized_terms.items():
            for term, freq, score in terms:  # Include all terms, not limited to top_n
                G.add_node(term, type="term" if cat != "Component" else "component", 
                          freq=freq, category=cat, unit=term_units.get(cat, "None"),
                          size=500 + 2000 * (freq / max([f for _, f, _ in terms], default=1)),
                          color="salmon" if cat != "Component" else "lightgreen",
                          score=score)
                G.add_edge(cat, term, weight=1.0, type="category-term", label="belongs_to")
                term_freqs[term] = freq

        # Compute co-occurrences with enhanced relationship detection
        co_occurrence_counts = defaultdict(lambda: defaultdict(int))
        
        for content in df["content"].values:
            content_lower = content.lower()
            doc = nlp(content_lower)
            
            # Extract sentences and find terms in each sentence
            for sent in doc.sents:
                sent_terms = []
                for term in term_freqs:
                    if re.search(rf'\b{re.escape(term)}\b', sent.text, re.IGNORECASE):
                        sent_terms.append(term)
                
                # Create combinations of all terms in the sentence
                for term1, term2 in combinations(sent_terms, 2):
                    if term1 != term2:
                        co_occurrence_counts[term1][term2] += 1
                        co_occurrence_counts[term2][term1] += 1
        
        # Add co-occurrence edges with enhanced attributes
        for term1, related_terms in co_occurrence_counts.items():
            for term2, count in related_terms.items():
                if count >= min_co_occurrence and term1 in G.nodes and term2 in G.nodes:
                    # Determine relationship type based on categories
                    cat1 = term_to_category.get(term1, "Other")
                    cat2 = term_to_category.get(term2, "Other")
                    
                    if cat1 == cat2:
                        rel_type = "intra_category"
                        label = f"co-occurs_with ({count})"
                    else:
                        rel_type = "inter_category"
                        label = f"related_to ({count})"
                    
                    # Add edge with enhanced attributes
                    G.add_edge(term1, term2, weight=count, type="term-term", 
                              relationship=rel_type, label=label, strength=count)
        
        # Add hierarchical relationships between terms
        for term in list(G.nodes):
            if G.nodes[term].get("type") == "term":
                # Find parent terms (shorter terms that are substrings)
                for potential_parent in list(G.nodes):
                    if (G.nodes[potential_parent].get("type") == "term" and 
                        potential_parent != term and 
                        len(potential_parent) < len(term) and
                        potential_parent in term.lower()):
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
        
        # Store the graph in session state for later use
        st.session_state.knowledge_graph = G
        
        return G, (nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename)
    
    except Exception as e:
        update_log(f"Error building knowledge graph data: {str(e)}")
        return None, None

def enhance_ner_with_knowledge_graph(ner_df, knowledge_graph):
    """
    Enhance NER results using knowledge graph insights
    """
    if ner_df.empty or knowledge_graph is None:
        return ner_df
    
    enhanced_ner = []
    
    for _, row in ner_df.iterrows():
        entity_text = row["entity_text"]
        entity_label = row["entity_label"]
        
        # Find related entities in the knowledge graph
        if entity_text in knowledge_graph.nodes:
            # Get neighbors in the knowledge graph
            neighbors = list(knowledge_graph.neighbors(entity_text))
            
            # Find related terms with strong connections
            strong_connections = []
            for neighbor in neighbors:
                edge_data = knowledge_graph.get_edge_data(entity_text, neighbor)
                if edge_data and edge_data.get("strength", 0) > 3:  # Threshold for strong connection
                    strong_connections.append((neighbor, edge_data.get("strength", 0)))
            
            # Sort by strength
            strong_connections.sort(key=lambda x: x[1], reverse=True)
            
            # Add context from knowledge graph
            context_enhanced = row["context"]
            if strong_connections:
                related_terms = ", ".join([f"{term}({strength})" for term, strength in strong_connections[:3]])
                context_enhanced += f" [KG: Related to {related_terms}]"
            
            # Create enhanced row
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
        ref_embeddings = {cat: [get_scibert_embedding(term) for term in terms if get_scibert_embedding(term) is not None] for cat, terms in categories.items()}
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
                        term_embeddings = [get_scibert_embedding(term) for term in selected_terms if get_scibert_embedding(term) is not None]
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
                                    unit = v_unit  # Correct to the valid unit if close
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
                        # Allow qualitative if None is in valid_units
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
        
        # Enhance NER results with knowledge graph if available
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
        if unit:  # If a specific unit is requested
            cat_df = cat_df[cat_df["unit"] == unit]
        if cat_df.empty:
            update_log(f"No data for {category} with unit {unit}")
            continue
        
        values = cat_df["value"].values
        if len(values) == 0:
            update_log(f"No numerical values for {category} with unit {unit}")
            continue
        
        fig, ax = plt.subplots(figsize=(8, 4))
        color = cm.get_cmap(colormap)(0.5)  # Use a consistent color from colormap
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
# Ensure battery_reliability_universe.db is included and set as default
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
            exclude_words = [w.strip().lower() for w in st.text_input("Exclude Words/Phrases (comma-separated)", value="et al, electrochem soc, phys", key="exclude_words").split(",") if w.strip()]
            similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="similarity_threshold")
            min_freq = st.slider("Minimum Frequency", min_value=1, max_value=20, value=5, key="min_freq")
            top_n = st.slider("Number of Top Terms", min_value=5, max_value=30, value=10, key="top_n")
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
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="categorize_logs")
    
    with tab3:
        st.header("Knowledge Graph")
        if st.button("Build Knowledge Graph", key="build_graph"):
            if st.session_state.categorized_terms:
                with st.spinner("Building knowledge graph..."):
                    G, csv_data = build_knowledge_graph_data(st.session_state.categorized_terms, st.session_state.db_file, min_co_occurrence=min_freq)
                    if G and G.edges():
                        # Create the figure with a hierarchical layout
                        fig, ax = plt.subplots(figsize=(12, 12))
                        # Use a hierarchical layout to emphasize category branches
                        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
                        # Adjust positions to group terms by category
                        for cat in categories:
                            if G.has_node(cat):
                                pos[cat][1] += 2  # Shift category nodes upward
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
                        
                        # Draw nodes
                        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
                        
                        # Draw edges with different styles based on type
                        category_term_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "category-term"]
                        term_term_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "term-term"]
                        hierarchical_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "hierarchical"]
                        
                        nx.draw_networkx_edges(G, pos, edgelist=category_term_edges, width=2.0, edge_color="blue", ax=ax)
                        nx.draw_networkx_edges(G, pos, edgelist=term_term_edges, width=1.0, edge_color="gray", ax=ax)
                        nx.draw_networkx_edges(G, pos, edgelist=hierarchical_edges, width=1.5, edge_color="green", style="dashed", ax=ax)
                        
                        # Draw labels
                        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                        
                        # Add edge labels
                        edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
                        
                        ax.set_title("Knowledge Graph of Battery Reliability Phenomena (All Terms, Branched by Category)")
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Show community detection visualization
                        st.subheader("Community Detection")
                        community_fig = visualize_knowledge_graph_communities(G)
                        if community_fig:
                            st.pyplot(community_fig)
                        
                        # Focused subgraph on key terms
                        st.subheader("Focused Subgraph on Key Terms")
                        # Create focused subgraph
                        key_nodes = [node for node in G.nodes if node in key_terms]
                        neighbors = set()
                        for node in key_nodes:
                            neighbors.update(nx.neighbors(G, node))
                        focused_nodes = set(key_nodes) | neighbors
                        sub_G = G.subgraph(focused_nodes)
                        
                        # Plot focused subgraph
                        if sub_G.edges():
                            fig_sub, ax_sub = plt.subplots(figsize=(10, 8))
                            pos_sub = nx.spring_layout(sub_G, k=0.6, iterations=50, seed=42)
                            node_colors_sub = [G.nodes[node].get("color", "salmon") for node in sub_G.nodes()]
                            node_sizes_sub = [G.nodes[node].get("size", 500) for node in sub_G.nodes()]
                            nx.draw_networkx_nodes(sub_G, pos_sub, node_color=node_colors_sub, node_size=node_sizes_sub, alpha=0.8, ax=ax_sub)
                            edge_widths_sub = [1.5 * d.get("weight", 1) for _, _, d in sub_G.edges(data=True)]
                            nx.draw_networkx_edges(sub_G, pos_sub, width=edge_widths_sub, alpha=0.5, ax=ax_sub)
                            nx.draw_networkx_labels(sub_G, pos_sub, font_size=8, ax=ax_sub)
                            ax_sub.set_title("Focused Subgraph on Key Terms")
                            plt.tight_layout()
                            st.pyplot(fig_sub)
                        
                        # Download CSVs
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
                        st.warning("No knowledge graph generated. Check logs.")
            else:
                st.warning("Run term categorization first.")
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
            
            # Add option to use knowledge graph for NER enhancement
            use_kg_enhancement = st.checkbox("Use Knowledge Graph for NER Enhancement", value=True, 
                                           help="Use knowledge graph relationships to improve NER results")
            
            if st.button("Run NER Analysis", key="ner_analyze"):
                if not selected_terms:
                    st.warning("Select at least one term for NER analysis.")
                else:
                    with st.spinner(f"Processing NER analysis for {len(selected_terms)} terms..."):
                        ner_df = perform_ner_on_terms(db_file, selected_terms)
                        st.session_state.ner_results = ner_df
                    if ner_df.empty:
                        st.warning("No entities were found.")
                    else:
                        st.success(f"Extracted {len(ner_df)} entities!")
                        
                        # Show enhanced NER results if knowledge graph was used
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
                        # Plot histograms for numerical values with specified units
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
- **Database Inspection**: View tables, schemas, and sample data from `battery_reliability.db` or `battery_reliability_universe.db`. Default database is `battery_reliability_universe.db` for full-text analysis of arXiv papers.
- **Term Categorization**: Groups terms into Deformation, Fatigue, Crack and Fracture, Degradation using SciBERT embeddings, based on full-text content from `battery_reliability_universe.db`.
- **Knowledge Graph**: Visualizes relationships between categories, terms, and components with enhanced branching and community detection, using `battery_reliability_universe.db`.
- **NER Analysis**: Extracts entities with context-aware unit assignment from `battery_reliability_universe.db`, enhanced with knowledge graph relationships.
- **Knowledge Graph Enhancement**: Uses community detection to identify related term clusters and hierarchical relationships.
- Place `battery_reliability.db` and `battery_reliability_universe.db` in the script's directory or upload them. `battery_reliability_universe.db` is required for full-text analysis and must have a `content` column.
- The script dynamically adjusts to missing columns (e.g., `categories`, `abstract`) in `battery_reliability_universe.db` by querying only available columns.
- Check `battery_reliability_analysis.log` for detailed logs, including schema details and query adjustments.
""")