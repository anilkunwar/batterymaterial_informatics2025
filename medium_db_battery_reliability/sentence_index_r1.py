# ==================== IMPORTS ====================
import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
import re
import numpy as np
import torch
import logging
import glob
import uuid
import pickle
import faiss
from rank_bm25 import BM25Okapi
import tempfile
import zipfile
import io

# Try to import transformers for SciBERT
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("Transformers not installed. Please install: `pip install transformers`")

# ==================== CONFIGURATION ====================
DB_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    filename=os.path.join(DB_DIR, 'index_builder.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="Battery Index Builder", layout="wide")
st.title("Battery Database Index Builder")
st.markdown("""
This tool builds a sentence index (FAISS + BM25) from a SQLite database of battery papers.
You can then download the index files for later use in the main analysis app.
""")

# ==================== MEMORY & CACHE ====================
@st.cache_resource
def load_spacy_model():
    try:
        nlp_model = spacy.load("en_core_web_sm")
        nlp_model.max_length = 100000
        
        @Language.component("custom_tokenizer")
        def custom_tokenizer(doc):
            hyphenated_phrases = ["electrode-cracking", "SEI-formation", "cyclic-mechanical-damage",
                                  "diffusion-induced-stress", "electrolyte-degradation", "capacity-fade",
                                  "lithium-ion", "Li-ion", "crack-propagation"]
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

@st.cache_resource
def load_scibert():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        model = model.to("cpu")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load SciBERT: {e}")
        return None, None

def get_scibert_embedding_batch(texts):
    tokenizer, model = load_scibert()
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
        return all_embeddings
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return [None] * len(texts)

def get_embedding_for_text(text):
    emb = get_scibert_embedding_batch([text])
    return emb[0] if emb else None

# ==================== INDEX BUILDING ====================
def build_index(db_path, progress_callback=None):
    """
    Build FAISS index, BM25 index, and metadata from the database.
    Returns a dictionary of file paths for the generated index files.
    """
    # Create a temporary directory to store the files
    temp_dir = tempfile.mkdtemp()
    faiss_path = os.path.join(temp_dir, "faiss.index")
    metadata_path = os.path.join(temp_dir, "metadata.pkl")
    bm25_path = os.path.join(temp_dir, "bm25.pkl")
    tokenized_path = os.path.join(temp_dir, "tokenized_sentences.pkl")

    # Load papers
    conn = sqlite3.connect(db_path)
    query = "SELECT id, title, year, content FROM papers WHERE content IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        st.error("No papers with content found.")
        return None

    nlp = load_spacy_model()
    all_sentences = []
    all_sentence_texts = []
    tokenized_sentences = []

    total_papers = len(df)
    for idx, row in df.iterrows():
        doc = nlp(row["content"][:nlp.max_length])
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
        if progress_callback:
            progress_callback((idx+1)/total_papers, f"Processing paper {idx+1}/{total_papers}")

    if not all_sentences:
        st.error("No sentences extracted.")
        return None

    # Compute embeddings
    st.info("Computing sentence embeddings...")
    embeddings = get_scibert_embedding_batch(all_sentence_texts)
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    valid_embeddings = np.array([embeddings[i] for i in valid_indices], dtype=np.float32)
    valid_sentences = [all_sentences[i] for i in valid_indices]
    valid_tokenized = [tokenized_sentences[i] for i in valid_indices]

    # Build FAISS index
    dim = valid_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product (cosine)
    index.add(valid_embeddings)
    faiss.write_index(index, faiss_path)

    # Save metadata
    with open(metadata_path, "wb") as f:
        pickle.dump(valid_sentences, f)

    # Build BM25 index
    bm25 = BM25Okapi(valid_tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save tokenized sentences
    with open(tokenized_path, "wb") as f:
        pickle.dump(valid_tokenized, f)

    return {
        "faiss": faiss_path,
        "metadata": metadata_path,
        "bm25": bm25_path,
        "tokenized": tokenized_path,
        "temp_dir": temp_dir
    }

# ==================== UI ====================
# Find existing .db files in the current directory
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = ["Upload a new .db file"] + [os.path.basename(f) for f in db_files]
db_selection = st.selectbox("Select or upload a database", db_options)

uploaded_file = None
if db_selection == "Upload a new .db file":
    uploaded_file = st.file_uploader("Choose a .db file", type=["db"])
    if uploaded_file:
        # Save temporarily to a unique name
        temp_db_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
        db_path = temp_db_path
    else:
        db_path = None
else:
    db_path = os.path.join(DB_DIR, db_selection)

if db_path and os.path.exists(db_path):
    st.info(f"Selected database: {os.path.basename(db_path)}")

    if st.button("Build Index", type="primary"):
        if db_path:
            progress_bar = st.progress(0, text="Starting...")
            status_text = st.empty()

            def update_progress(progress, message):
                progress_bar.progress(progress, text=message)
                status_text.text(message)

            with st.spinner("Building index, this may take a while..."):
                result = build_index(db_path, progress_callback=update_progress)

            if result:
                st.success("Index built successfully!")
                # Display download buttons
                st.subheader("Download Index Files")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    with open(result["faiss"], "rb") as f:
                        st.download_button("faiss.index", f, file_name="faiss.index", mime="application/octet-stream")
                with col2:
                    with open(result["metadata"], "rb") as f:
                        st.download_button("metadata.pkl", f, file_name="metadata.pkl", mime="application/octet-stream")
                with col3:
                    with open(result["bm25"], "rb") as f:
                        st.download_button("bm25.pkl", f, file_name="bm25.pkl", mime="application/octet-stream")
                with col4:
                    with open(result["tokenized"], "rb") as f:
                        st.download_button("tokenized_sentences.pkl", f, file_name="tokenized_sentences.pkl", mime="application/octet-stream")

                # Option to download all as zip
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(result["faiss"], arcname="faiss.index")
                    zipf.write(result["metadata"], arcname="metadata.pkl")
                    zipf.write(result["bm25"], arcname="bm25.pkl")
                    zipf.write(result["tokenized"], arcname="tokenized_sentences.pkl")
                zip_buffer.seek(0)
                st.download_button(
                    label="Download all as ZIP",
                    data=zip_buffer,
                    file_name="sentence_index.zip",
                    mime="application/zip"
                )
            else:
                st.error("Index building failed. Check logs.")
        else:
            st.error("No database selected.")
else:
    if db_selection != "Upload a new .db file":
        st.warning("Selected database file not found.")

st.markdown("""
---
**Note:** The index files are generated in a temporary directory and will be deleted when the app stops. Use the download buttons to save them permanently.
""")
