import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import sqlite3
from datetime import datetime
import logging
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from collections import Counter
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

# Define database directory and files
DB_DIR = os.path.dirname(os.path.abspath(__file__))
RELIABILITY_DB_FILE = os.path.join(DB_DIR, "battery_reliability.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "battery_reliability_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'battery_reliability_query.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Battery Reliability Challenges Query Tool", layout="wide")
st.title("Battery Reliability Challenges Query Tool with SciBERT")
st.markdown("""
This tool queries arXiv for papers on **battery reliability challenges**, focusing on factors that degrade performance, such as **electrode cracking**, **SEI formation**, **cyclic mechanical damage**, **diffusion-induced stress**, **electrolyte degradation**, and **capacity fade**. It uses SciBERT to prioritize relevant abstracts (>30% relevance) and stores metadata in `battery_reliability.db` and full PDF text in `battery_reliability_universe.db` for fallback searches.
""")

# Dependency check
st.sidebar.header("Setup")
st.sidebar.markdown("""
**Dependencies**:
- `arxiv`, `pymupdf`, `pandas`, `streamlit`, `transformers`, `torch`, `scipy`, `numpy`, `tenacity`
- Install: `pip install arxiv pymupdf pandas streamlit transformers torch scipy numpy tenacity`
""")

# Load SciBERT model and tokenizer
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# Create PDFs directory
pdf_dir = os.path.join(DB_DIR, "pdfs")
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir}")

# Initialize session state for logs
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# Define key terms related to battery reliability challenges
KEY_TERMS = [
    "electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced stress",
    "electrical response", "micro-cracking", "electrolyte degradation", "capacity fade",
    "lithium plating", "thermal runaway", "mechanical degradation", "cycle life",
    "lithium", "electrode", "crack", "fracture", "battery", "particles", "cathode",
    "mechanical", "cycles", "electrolyte", "degradation", "surface", "capacity", "cycling", "stress", "diffusion"
]

# SciBERT scoring
@st.cache_data
def score_abstract_with_scibert(abstract):
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True, return_attention_mask=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        logits = outputs.logits.numpy()
        probs = softmax(logits, axis=1)
        relevance_prob = probs[0][1]
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [i for i, token in enumerate(tokens) if any(kw.lower() in token.lower() for kw in KEY_TERMS)]
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1 and relevance_prob < 0.5:
                relevance_prob = min(relevance_prob + 0.2 * len(keyword_indices), 1.0)
        update_log(f"SciBERT scored abstract: {relevance_prob:.3f} (keywords matched: {len(keyword_indices)})")
        return relevance_prob
    except Exception as e:
        update_log(f"SciBERT scoring failed: {str(e)}")
        # Fallback scoring
        abstract_lower = abstract.lower()
        word_counts = Counter(re.findall(r'\b\w+\b', abstract_lower))
        total_words = sum(word_counts.values())
        score = sum(word_counts.get(kw.lower(), 0) for kw in KEY_TERMS) / (total_words + 1e-6)
        max_possible_score = len(KEY_TERMS) / 10
        relevance_prob = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        update_log(f"Fallback scoring: {relevance_prob:.3f}")
        return relevance_prob

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        update_log(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# Initialize database
def initialize_db(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                download_status TEXT,
                matched_terms TEXT,
                relevance_prob REAL,
                pdf_path TEXT,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                paper_id TEXT,
                entity_text TEXT,
                entity_label TEXT,
                value REAL,
                unit TEXT,
                context TEXT,
                phase TEXT,
                score REAL,
                co_occurrence BOOLEAN,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON parameters(paper_id)")
        conn.commit()
        conn.close()
        update_log(f"Initialized database schema for {db_file}")
    except Exception as e:
        update_log(f"Failed to initialize {db_file}: {str(e)}")
        st.error(f"Failed to initialize {db_file}: {str(e)}")

# Create battery_reliability_universe.db incrementally
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_universe_db(paper, db_file=UNIVERSE_DB_FILE):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper["id"],
            paper.get("title", ""),
            paper.get("authors", "Unknown"),
            paper.get("year", 0),
            paper.get("content", "No text extracted")
        ))
        conn.commit()
        conn.close()
        update_log(f"Updated {db_file} with paper {paper['id']}")
        return db_file
    except Exception as e:
        update_log(f"Error updating {db_file}: {str(e)}")
        raise

# Save to SQLite
def save_to_sqlite(papers_df, params_list, reliability_db_file=RELIABILITY_DB_FILE):
    try:
        initialize_db(reliability_db_file)
        conn = sqlite3.connect(reliability_db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        params_df = pd.DataFrame(params_list)
        if not params_df.empty:
            params_df.to_sql("parameters", conn, if_exists="append", index=False)
        conn.close()
        update_log(f"Saved {len(papers_df)} papers and {len(params_list)} parameters to {reliability_db_file}")
        return f"Saved to {reliability_db_file}"
    except Exception as e:
        update_log(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Query arXiv
@st.cache_data
def query_arxiv(query, categories, max_results, start_year, end_year):
    try:
        query_terms = query.strip().split()
        formatted_terms = [term.strip('"').replace(" ", "+") for term in query_terms]
        api_query = " ".join(formatted_terms)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=api_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in client.results(search):
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                abstract = result.summary.lower()
                title = result.title.lower()
                query_words = set(word.lower().strip('"') for word in query_terms)
                matched_terms = [word for word in query_words if word in abstract or word in title]
                if not matched_terms:
                    continue
                relevance_prob = score_abstract_with_scibert(result.summary)
                abstract_highlighted = abstract
                for term in matched_terms:
                    abstract_highlighted = re.sub(r'\b{}\b'.format(re.escape(term)), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary,
                    "abstract_highlighted": abstract_highlighted,
                    "pdf_url": result.pdf_url,
                    "download_status": "Not downloaded",
                    "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                    "relevance_prob": round(relevance_prob * 100, 2),
                    "pdf_path": None,
                    "content": None
                })
            if len(papers) >= max_results:
                break
        papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
        update_log(f"Found {len(papers)} papers")
        return papers
    except Exception as e:
        update_log(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}. Try simplifying the query.")
        return []

# Download PDF and extract text
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_pdf_and_extract(pdf_url, paper_id, paper_metadata):
    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    try:
        urllib.request.urlretrieve(pdf_url, pdf_path)
        file_size = os.path.getsize(pdf_path) / 1024
        text = extract_text_from_pdf(pdf_path)
        if not text.startswith("Error"):
            paper_data = {
                "id": paper_id,
                "title": paper_metadata.get("title", ""),
                "authors": paper_metadata.get("authors", "Unknown"),
                "year": paper_metadata.get("year", 0),
                "content": text
            }
            create_universe_db(paper_data)
            update_log(f"Downloaded and extracted text for paper {paper_id} ({file_size:.2f} KB)")
            return f"Downloaded ({file_size:.2f} KB)", pdf_path, text
        else:
            update_log(f"Text extraction failed for paper {paper_id}: {text}")
            return f"Failed: {text}", None, text
    except Exception as e:
        update_log(f"PDF download failed for {paper_id}: {str(e)}")
        return f"Failed: {str(e)}", None, f"Error: {str(e)}"

# Main Streamlit app
st.header("arXiv Query for Battery Reliability Challenges")
st.markdown("Search for abstracts on factors that challenge battery reliability, such as **electrode cracking**, **SEI formation**, **cyclic mechanical damage**, **diffusion-induced stress**, **electrolyte degradation**, and **capacity fade** using SciBERT.")

log_container = st.empty()
def display_logs():
    log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

with st.sidebar:
    st.subheader("Search Parameters")
    query = st.text_input("Query", value=' OR '.join([f'"{term}"' for term in KEY_TERMS]))
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"]
    categories = st.multiselect("Categories", default_categories, default=default_categories)
    max_results = st.slider("Max Papers", min_value=1, max_value=500, value=10)
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=2010)
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
    output_formats = st.multiselect("Output Formats", ["SQLite (.db)", "CSV", "JSON"], default=["SQLite (.db)"])
    search_button = st.button("Search arXiv")

if search_button:
    if not query.strip():
        st.error("Enter a valid query.")
    elif not categories:
        st.error("Select at least one category.")
    elif start_year > end_year:
        st.error("Start year must be ≤ end year.")
    else:
        with st.spinner("Querying arXiv..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("No papers found. Broaden query or categories.")
        else:
            st.success(f"Found **{len(papers)}** papers. Filtering for relevance > 30%...")
            relevant_papers = [p for p in papers if p["relevance_prob"] > 30.0]
            if not relevant_papers:
                st.warning("No papers with relevance > 30%. Broaden query or check 'battery_reliability_query.log'.")
            else:
                st.success(f"**{len(relevant_papers)}** papers with relevance > 30%. Downloading PDFs...")
                progress_bar = st.progress(0)
                for i, paper in enumerate(relevant_papers):
                    if paper["pdf_url"]:
                        status, pdf_path, content = download_pdf_and_extract(paper["pdf_url"], paper["id"], paper)
                        paper["download_status"] = status
                        paper["pdf_path"] = pdf_path
                        paper["content"] = content
                    progress_bar.progress((i + 1) / len(relevant_papers))
                    time.sleep(1)  # Avoid rate-limiting
                    update_log(f"Processed paper {i+1}/{len(relevant_papers)}: {paper['title']}")
                
                df = pd.DataFrame(relevant_papers)
                st.subheader("Papers (Relevance > 30%)")
                st.dataframe(
                    df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "relevance_prob", "download_status"]],
                    use_container_width=True
                )
                
                if "SQLite (.db)" in output_formats:
                    sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), [])
                    st.info(sqlite_status)
                
                if "CSV" in output_formats:
                    csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                    st.download_button(
                        label="Download Paper Metadata CSV",
                        data=csv,
                        file_name="battery_reliability_papers.csv",
                        mime="text/csv"
                    )
                
                if "JSON" in output_formats:
                    json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
                    st.download_button(
                        label="Download Paper Metadata JSON",
                        data=json_data,
                        file_name="battery_reliability_papers.json",
                        mime="application/json"
                    )
                
                display_logs()
