import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from collections import Counter
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
import math
import glob
import uuid
import seaborn as sns

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

# Logging setup
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'common_term_ner_scibert.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit configuration
st.set_page_config(page_title="Battery Mechanics Analysis Tool (SciBERT)", layout="wide")
st.title("Battery Mechanics Analysis: Fatigue, Fracture, Crack, Plasticity, Degradation")
st.markdown("""
This tool inspects SQLite databases, categorizes terms related to battery mechanics phenomena, builds a knowledge graph, and performs NER analysis using SciBERT.
Select or upload a database, then use the tabs to inspect the database, categorize terms, visualize relationships, or extract entities with numerical values.
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
    hyphenated_phrases = ["lithium-ion", "Li-ion", "young’s modulus", "sei growth", "phase field", "active material", "crack propagation", "plastic deformation"]
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
    st.session_state.db_file = None
if "term_counts" not in st.session_state:
    st.session_state.term_counts = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None

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

def inspect_database(db_path):
    try:
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

            terms_to_search = ["crack", "fracture", "plastic deformation", "fatigue", "degradation"]
            st.subheader("Term Frequency in 'text' Column")
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
        query = "SELECT text AS content FROM pages WHERE text IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            update_log(f"No valid content found in {os.path.basename(db_file)}")
            st.warning(f"No valid content found in {os.path.basename(db_file)}.")
            return {}, Counter()

        update_log(f"Loaded {len(df)} content entries")
        total_words = 0
        term_counts = Counter()
        categories = {
            "Plasticity": ["plastic deformation", "yield strength", "plastic strain", "ductility", "inelastic strain", "flow stress"],
            "Fatigue": ["fatigue life", "cyclic loading", "cycle life", "fatigue crack", "stress cycling", "mechanical fatigue"],
            "Crack": ["crack", "cracking", "crack propagation", "crack growth", "crack initiation", "microcrack"],
            "Fracture": ["fracture", "fracture toughness", "brittle fracture", "ductile fracture", "crack branching"],
            "Degradation": ["degradation", "capacity fade", "cycle degradation", "sei growth", "electrode degradation", "electrolyte decomposition"]
        }
        component_terms = ["electrode", "electrolyte", "phase", "cathode", "anode"]
        exclude_words = ["et al", "electrochem soc", "phys", "model", "based", "high", "field"]
        
        # Generate embeddings for seed terms
        seed_embeddings = {cat: [get_scibert_embedding(term) for term in terms if get_scibert_embedding(term) is not None] for cat, terms in categories.items()}
        component_embeddings = [get_scibert_embedding(term) for term in component_terms if get_scibert_embedding(term) is not None]
        
        categorized_terms = {cat: [] for cat in categories}
        other_terms = []
        term_freqs = Counter()
        
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
            all_terms = phrases + single_words
            term_counts.update(all_terms)
            
            for term in all_terms:
                if any(w in term.lower() for w in exclude_words):
                    continue
                term_embedding = get_scibert_embedding(term)
                if term_embedding is None:
                    continue
                term_freqs[term] += 1
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
                if best_cat and term_freqs[term] >= min_freq:
                    categorized_terms[best_cat].append((term, term_freqs[term], best_score))
                elif term in component_terms or any(np.dot(term_embedding, comp_emb) / (np.linalg.norm(term_embedding) * np.linalg.norm(comp_emb)) > similarity_threshold for comp_emb in component_embeddings if np.linalg.norm(term_embedding) != 0 and np.linalg.norm(comp_emb) != 0):
                    categorized_terms.setdefault("Component", []).append((term, term_freqs[term], best_score))
                elif term_freqs[term] >= min_freq:
                    other_terms.append((term, term_freqs[term], best_score))
            
            progress_bar.progress((i + 1) / len(df))
        
        # Sort terms by frequency within each category
        for cat in categorized_terms:
            categorized_terms[cat] = sorted(categorized_terms[cat], key=lambda x: x[1], reverse=True)
        other_terms = sorted(other_terms, key=lambda x: x[1], reverse=True)
        categorized_terms["Other"] = other_terms[:50]  # Limit other terms to top 50
        
        update_log(f"Categorized {sum(len(terms) for terms in categorized_terms.values())} terms across {len(categorized_terms)} categories")
        return categorized_terms, term_freqs
    except Exception as e:
        update_log(f"Error categorizing terms: {str(e)}")
        st.error(f"Error categorizing terms: {str(e)}")
        return {}, Counter()

@st.cache_data(hash_funcs={str: lambda x: x})
def build_knowledge_graph(categorized_terms, db_file, min_co_occurrence=2, top_n=10):
    try:
        update_log(f"Building knowledge graph for top {top_n} terms per category")
        conn = sqlite3.connect(db_file)
        query = "SELECT text AS content FROM pages WHERE text IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()

        G = nx.Graph()
        # Add category nodes
        categories = list(categorized_terms.keys())
        for cat in categories:
            G.add_node(cat, type="category", freq=0)
        
        # Add term and component nodes
        term_freqs = {}
        for cat, terms in categorized_terms.items():
            for term, freq, _ in terms[:top_n]:
                G.add_node(term, type="term" if cat != "Component" else "component", freq=freq, category=cat)
                G.add_edge(cat, term, weight=1.0, type="category-term")
                term_freqs[term] = freq

        # Compute co-occurrences
        for content in df["content"].values:
            content_lower = content.lower()
            doc = nlp(content_lower)
            terms_present = []
            for sent in doc.sents:
                sent_terms = []
                for term in term_freqs:
                    if re.search(rf'\b{re.escape(term)}\b', sent.text):
                        sent_terms.append(term)
                terms_present.extend(combinations(sent_terms, 2))
            for term1, term2 in terms_present:
                if term1 != term2:
                    if G.has_edge(term1, term2):
                        G[term1][term2]["weight"] += 1
                    else:
                        G.add_edge(term1, term2, weight=1, type="term-term")

        # Filter edges by min_co_occurrence
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "term-term" and d["weight"] < min_co_occurrence]
        G.remove_edges_from(edges_to_remove)

        if G.edges():
            fig, ax = plt.subplots(figsize=(10, 10))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            node_colors = []
            node_sizes = []
            for node in G.nodes:
                if G.nodes[node]["type"] == "category":
                    node_colors.append("skyblue")
                    node_sizes.append(1000)
                elif G.nodes[node]["type"] == "component":
                    node_colors.append("lightgreen")
                    node_sizes.append(800)
                else:
                    node_colors.append("salmon")
                    node_sizes.append(500 + 2000 * (G.nodes[node]["freq"] / max(term_freqs.values(), default=1)))
            edge_colors = ["gray" if G[u][v]["type"] == "term-term" else "blue" for u, v in G.edges()]
            edge_widths = [2 * G[u][v]["weight"] / max([d["weight"] for _, _, d in G.edges(data=True)], default=1) for u, v in G.edges()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, edge_color=edge_colors, width=edge_widths, font_size=8, font_weight="bold", ax=ax)
            ax.set_title(f"Knowledge Graph of Battery Mechanics Phenomena (Top {top_n} Terms per Category)")
            plt.tight_layout()
            
            nodes_df = pd.DataFrame([(n, d["type"], d.get("category", ""), d.get("freq", 0)) for n, d in G.nodes(data=True)], columns=["node", "type", "category", "frequency"])
            edges_df = pd.DataFrame([(u, v, d["weight"], d["type"]) for u, v, d in G.edges(data=True)], columns=["source", "target", "weight", "type"])
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
            return fig, (nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename)
        update_log("No edges in knowledge graph after filtering")
        return None, None
    except Exception as e:
        update_log(f"Error building knowledge graph: {str(e)}")
        return None, None

def perform_ner_on_terms(db_file, selected_terms):
    try:
        update_log(f"Starting NER analysis for terms: {', '.join(selected_terms)}")
        conn = sqlite3.connect(db_file)
        query = "SELECT filename as title, page_num as id, text as content, NULL as year FROM pages WHERE text IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            update_log(f"No valid content found in {os.path.basename(db_file)}")
            st.error("No valid content found.")
            return pd.DataFrame()

        categories = {
            "Plasticity": ["plastic deformation", "yield strength", "plastic strain", "ductility", "inelastic strain", "flow stress"],
            "Fatigue": ["fatigue life", "cyclic loading", "cycle life", "fatigue crack", "stress cycling", "mechanical fatigue"],
            "Crack": ["crack", "cracking", "crack propagation", "crack growth", "crack initiation", "microcrack"],
            "Fracture": ["fracture", "fracture toughness", "brittle fracture", "ductile fracture", "crack branching"],
            "Degradation": ["degradation", "capacity fade", "cycle degradation", "sei growth", "electrode degradation", "electrolyte decomposition"]
        }
        valid_units = {
            "Plasticity": ["MPa", "%"],
            "Fatigue": ["cycles", "MPa"],
            "Crack": ["μm", None],
            "Fracture": ["MPa·m^0.5", None],
            "Degradation": ["%", "nm"]
        }
        valid_ranges = {
            "Plasticity": [(0, 1000, "MPa"), (0, 100, "%")],
            "Fatigue": [(1, 10000, "cycles"), (0, 1000, "MPa")],
            "Crack": [(0, 1000, "μm"), (None, None, None)],
            "Fracture": [(0, 10, "MPa·m^0.5"), (None, None, None)],
            "Degradation": [(0, 100, "%"), (0, 1000, "nm")]
        }
        numerical_pattern = r"(\d+\.?\d*[eE]?-?\d*|\d+)\s*(MPa|GPa|kPa|Pa|%|μm|nm|cm²/s|mAh/g|V|S/cm|MPa·m\^0\.5|cycles)"
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
                    update_log(f"Truncated content for entry {row['id']}")
                if not text.strip() or len(text) < 10:
                    update_log(f"Skipping entry {row['id']} due to empty/short content")
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
                    update_log(f"No valid spans in entry {row['id']}")
                    continue
                for span, orig_sent, nearby_sent in spans:
                    span_text = span.text.lower().strip()
                    if not span_text:
                        update_log(f"Skipping empty span in entry {row['id']}")
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
                            elif unit == "MAH/G":
                                unit = "mAh/g"
                            elif unit == "CM²/S":
                                unit = "cm²/s"
                            elif unit == "MPA·M^0.5":
                                unit = "MPa·m^0.5"
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
                            for valid_unit in valid_units.get(best_label, []):
                                if valid_unit is None:
                                    continue
                                unit_embedding = get_scibert_embedding(f"{span_text} {valid_unit}")
                                if unit_embedding is None:
                                    continue
                                unit_score = np.dot(context_embedding, unit_embedding) / (np.linalg.norm(context_embedding) * np.linalg.norm(unit_embedding))
                                if unit_score > 0.6:
                                    unit_valid = True
                                    break
                            if not unit_valid:
                                continue
                        for min_val, max_val, expected_unit in valid_ranges.get(best_label, [(None, None, None)]):
                            if expected_unit is None and value is None:
                                break
                            if expected_unit == unit and min_val is not None and max_val is not None:
                                if not (min_val <= value <= max_val):
                                    continue
                    entity_key = (row["id"], span_text, best_label, value if value is not None else "", unit if unit is not None else "")
                    if entity_key in entity_set:
                        continue
                    entity_set.add(entity_key)
                    context_start = max(0, span.start_char - 100)
                    context_end = min(len(text), span.end_char + 100)
                    context_text = text[context_start:context_end].replace("\n", " ")
                    entities.append({
                        "paper_id": row["id"],
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
                update_log(f"Error processing entry {row['id']}: {str(e)}")
        update_log(f"Extracted {len(entities)} entities")
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

# Database selection
st.header("Select or Upload Database")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
db_selection = st.selectbox("Select Database", db_options, key="db_select")
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
    if db_selection:
        st.session_state.db_file = os.path.join(DB_DIR, db_selection)
        update_log(f"Selected database: {db_selection}")

# Main app logic
if st.session_state.db_file:
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
                    fig, csv_data = build_knowledge_graph(st.session_state.categorized_terms, st.session_state.db_file, min_co_occurrence=min_freq, top_n=top_n)
                    if fig:
                        st.pyplot(fig)
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
            default_terms = [term for term in ["crack", "fracture", "plastic deformation", "fatigue life", "degradation"] if term in available_terms]
            selected_terms = st.multiselect("Select Terms for NER", available_terms, default_terms, key="select_terms")
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
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="ner_logs")
else:
    st.warning("Select or upload a database file.")

# Notes
st.markdown("""
---
**Notes**
- **Database Inspection**: View tables, schemas, and sample data. Full-text search requires FTS5 in `page_fts` table.
- **Term Categorization**: Groups terms into Plasticity, Fatigue, Crack, Fracture, Degradation using SciBERT embeddings.
- **Knowledge Graph**: Visualizes relationships between categories, terms, and components (Electrode, Electrolyte, Phase).
- **NER Analysis**: Extracts entities with context-aware unit assignment (e.g., μm for Crack, MPa·m^0.5 for Fracture).
- Place `lithiumbattery_miniuniverse.db` in the script's directory or upload it.
- Check `common_term_ner_scibert.log` for detailed logs.
""")
