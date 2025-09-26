import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
import re
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import uuid
import logging
import glob

# Matplotlib and Seaborn configuration for publication-quality plots
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Publication-friendly font
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,  # High resolution for publication
    'savefig.transparent': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})
sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.3})

# Directory setup
DB_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "battery_reliability_universe.db")

# Logging setup
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'battery_reliability_ner.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit configuration
st.set_page_config(page_title="Robust NER for Battery Reliability Analysis", layout="wide")
st.title("Robust NER for Battery Reliability Analysis")
st.markdown("""
This application performs Named Entity Recognition (NER) on battery reliability data, categorizing quantities with units (e.g., '100 MPa', '28 cycles') into Deformation, Fatigue, Crack and Fracture, or Degradation using a statistical classifier. Results are visualized in publication-quality histograms with statistical summaries.
""")

# Load spaCy model
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
                          "electrolyte-degradation", "capacity-fade", "crack-propagation", "crack-damage", "fracture-damage"]
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
if "db_file" not in st.session_state:
    st.session_state.db_file = UNIVERSE_DB_FILE if os.path.exists(UNIVERSE_DB_FILE) else None
if "unit_classifier" not in st.session_state:
    st.session_state.unit_classifier = None

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

# Define valid units and ranges
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

def train_unit_classifier():
    """Train an MLPClassifier for quantity-unit categorization."""
    training_data = [
        ("100 MPa", "yield strength in electrode", "Deformation"),
        ("50 %", "volume expansion observed", "Deformation"),
        ("28 cycles", "fatigue life under cyclic loading", "Fatigue"),
        ("500 cycles", "cyclic mechanical damage", "Fatigue"),
        ("10 μm", "crack growth measured", "Crack and Fracture"),
        ("2 MPa·m^0.5", "fracture toughness", "Crack and Fracture"),
        ("30 %", "capacity fade after cycling", "Degradation"),
        ("50 nm", "SEI formation thickness", "Degradation"),
        ("200 MPa", "stress cycling in fatigue test", "Fatigue"),
        ("5 %", "mechanical degradation rate", "Degradation"),
        ("100 μm", "micro-cracking observed", "Crack and Fracture"),
        ("1000 cycles", "endurance limit reached", "Fatigue"),
        ("0.5 MPa·m^0.5", "stress intensity factor", "Crack and Fracture"),
        ("20 %", "electrolyte degradation", "Degradation")
    ]
    
    features = []
    labels = []
    context_terms = {
        "Deformation": ["yield strength", "plastic strain", "volume expansion", "swelling", "elastic deformation"],
        "Fatigue": ["cyclic loading", "fatigue life", "stress cycling", "cyclic mechanical damage", "endurance limit"],
        "Crack and Fracture": ["crack growth", "fracture toughness", "micro-cracking", "stress intensity factor", "crack propagation"],
        "Degradation": ["capacity fade", "SEI formation", "electrolyte degradation", "mechanical degradation", "capacity loss"]
    }
    
    for qty_unit, context, category in training_data:
        qty_embedding = get_scibert_embedding(qty_unit)
        context_embedding = get_scibert_embedding(context)
        if qty_embedding is None or context_embedding is None:
            continue
        
        context_features = []
        for cat, terms in context_terms.items():
            term_embs = get_scibert_embedding(terms)
            term_embs = [emb for emb in term_embs if emb is not None]
            if term_embs:
                sims = [cosine_similarity([context_embedding], [t_emb])[0][0] for t_emb in term_embs]
                context_features.append(max(sims) if sims else 0)
            else:
                context_features.append(0)
        
        feature_vector = np.concatenate([qty_embedding, context_embedding, context_features])
        features.append(feature_vector)
        labels.append(category)
    
    if not features:
        update_log("No valid features for unit classifier training")
        return None
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels)
    classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    classifier.fit(features, y_train)
    update_log(f"Trained unit classifier with {len(features)} examples")
    return classifier, label_encoder

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
        numerical_pattern = r"(\d+\.?\d*[eE]?-?\d*|\d+)\s*(mpa|gpa|kpa|pa|%|μm|nm|MPa·m\^0\.5|cycles|MPa|GPa)"
        term_patterns = {term: re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE) for term in selected_terms}
        
        if st.session_state.unit_classifier is None:
            classifier, label_encoder = train_unit_classifier()
            if classifier is None:
                update_log("Unit classifier training failed")
                st.error("Unit classifier training failed")
                return pd.DataFrame()
            st.session_state.unit_classifier = (classifier, label_encoder)
        else:
            classifier, label_encoder = st.session_state.unit_classifier
        
        context_terms = {
            "Deformation": ["yield strength", "plastic strain", "volume expansion", "swelling", "elastic deformation"],
            "Fatigue": ["cyclic loading", "fatigue life", "stress cycling", "cyclic mechanical damage", "endurance limit"],
            "Crack and Fracture": ["crack growth", "fracture toughness", "micro-cracking", "stress intensity factor", "crack propagation"],
            "Degradation": ["capacity fade", "SEI formation", "electrolyte degradation", "mechanical degradation", "capacity loss"]
        }
        
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
                        start_sent_idx = max(0, sent_idx - 3)
                        end_sent_idx = min(len(list(doc.sents)), sent_idx + 4)
                        context_text = " ".join([s.text for s in list(doc.sents)[start_sent_idx:end_sent_idx]])
                        for nearby_sent in list(doc.sents)[start_sent_idx:end_sent_idx]:
                            matches = re.finditer(numerical_pattern, nearby_sent.text, re.IGNORECASE)
                            for match in matches:
                                start_char = nearby_sent.start_char + match.start()
                                end_char = nearby_sent.start_char + match.end()
                                span = doc.char_span(start_char, end_char, alignment_mode="expand")
                                if span:
                                    spans.append((span, sent.text, context_text))
                if not spans:
                    update_log(f"No valid spans in entry {row['paper_id']}")
                    continue
                for span, orig_sent, context_text in spans:
                    span_text = span.text.lower().strip()
                    if not span_text:
                        update_log(f"Skipping empty span in entry {row['paper_id']}")
                        continue
                    term_matched = False
                    for term in selected_terms:
                        if term_patterns[term].search(span_text) or term_patterns[term].search(orig_sent) or term_patterns[term].search(context_text):
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
                    if value is None or unit is None:
                        continue
                    
                    qty_embedding = get_scibert_embedding(span_text)
                    context_embedding = get_scibert_embedding(context_text)
                    if qty_embedding is None or context_embedding is None:
                        continue
                    
                    context_features = []
                    for cat, terms in context_terms.items():
                        term_embs = get_scibert_embedding(terms)
                        term_embs = [emb for emb in term_embs if emb is not None]
                        if term_embs:
                            sims = [cosine_similarity([context_embedding], [t_emb])[0][0] for t_emb in term_embs]
                            context_features.append(max(sims) if sims else 0)
                        else:
                            context_features.append(0)
                    
                    feature_vector = np.concatenate([qty_embedding, context_embedding, context_features])
                    feature_vector = feature_vector.reshape(1, -1)
                    pred_probs = classifier.predict_proba(feature_vector)[0]
                    pred_label_idx = np.argmax(pred_probs)
                    best_label = label_encoder.inverse_transform([pred_label_idx])[0]
                    best_score = pred_probs[pred_label_idx]
                    
                    if unit not in valid_units.get(best_label, []):
                        update_log(f"Unit '{unit}' not valid for category '{best_label}' in span '{span_text}'")
                        continue
                    
                    range_valid = False
                    for min_val, max_val, expected_unit in valid_ranges.get(best_label, [(None, None, None)]):
                        if expected_unit == unit and min_val is not None and max_val is not None:
                            if min_val <= value <= max_val:
                                range_valid = True
                                break
                    if not range_valid:
                        update_log(f"Value {value} {unit} out of range for category '{best_label}' in span '{span_text}'")
                        continue
                    
                    entity_key = (row["paper_id"], span_text, best_label, value, unit)
                    if entity_key in entity_set:
                        continue
                    entity_set.add(entity_key)
                    context_start = max(0, span.start_char - 100)
                    context_end = min(len(text), span.end_char + 100)
                    context_display = text[context_start:context_end].replace("\n", " ")
                    entities.append({
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "year": row["year"],
                        "entity_text": span.text,
                        "entity_label": best_label,
                        "value": value,
                        "unit": unit,
                        "context": context_display,
                        "score": best_score
                    })
                progress_bar.progress((i + 1) / len(df))
            except Exception as e:
                update_log(f"Error processing entry {row['paper_id']}: {str(e)}")
        update_log(f"Extracted {len(entities)} entities")
        
        return pd.DataFrame(entities)
    except Exception as e:
        update_log(f"NER analysis failed: {str(e)}")
        st.error(f"NER analysis failed: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def plot_ner_value_histograms(df, categories_units, top_n, colormap):
    if df.empty or df["value"].isna().all():
        update_log("Empty or invalid DataFrame for histogram plotting")
        return [], pd.DataFrame()
    value_df = df[df["value"].notna() & df["unit"].notna()]
    if value_df.empty:
        update_log("No valid numerical data for histogram plotting")
        return [], pd.DataFrame()
    
    figs = []
    stats_data = []
    palette = sns.color_palette("muted", n_colors=1)[0]  # Use a single muted color for consistency
    
    for category, unit in categories_units.items():
        cat_df = value_df[value_df["entity_label"] == category]
        if unit:
            cat_df = cat_df[cat_df["unit"] == unit]
        if cat_df.empty:
            update_log(f"No data for {category} with unit {unit}")
            continue
        
        values = cat_df["value"].values
        if len(values) < 2:  # Need at least 2 points for a histogram
            update_log(f"Insufficient data points ({len(values)}) for {category} with unit {unit}")
            continue
        
        # Outlier detection using IQR
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
        outliers_removed = len(values) - len(filtered_values)
        
        if len(filtered_values) < 2:
            update_log(f"Insufficient data points after outlier filtering ({len(filtered_values)}) for {category} with unit {unit}")
            continue
        
        # Calculate statistics
        mean_val = np.mean(filtered_values)
        median_val = np.median(filtered_values)
        std_val = np.std(filtered_values)
        count_val = len(filtered_values)
        
        stats_data.append({
            "Category": category,
            "Unit": unit,
            "Count": count_val,
            "Mean": mean_val,
            "Median": median_val,
            "Std": std_val
        })
        
        # Calculate bin width using Freedman-Diaconis rule with a minimum bin width
        iqr = IQR
        bin_width = max(2 * iqr * len(filtered_values) ** (-1/3), 0.1)  # Ensure minimum bin width
        if bin_width == 0:
            bin_width = 1
        num_bins = max(5, min(50, int((max(filtered_values) - min(filtered_values)) / bin_width)))  # Limit bins to 5-50
        
        # Round bin edges to meaningful intervals
        min_val = min(filtered_values)
        max_val = max(filtered_values)
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_edges = np.round(bin_edges, 1 if max_val - min_val > 10 else 2)  # Adjust precision based on range
        
        # Create publication-quality histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_values, bins=bin_edges, color=palette, edgecolor="black", linewidth=1.2, ax=ax)
        
        # Customize axes
        ax.set_xlabel(f"{unit}", fontsize=12, weight='bold')
        ax.set_ylabel("Frequency", fontsize=12, weight='bold')
        ax.set_title(f"{category} Value Distribution\n({unit})", fontsize=14, weight='bold', pad=15)
        
        # Add statistical annotations
        stats_text = (
            f"Mean: {mean_val:.2f}\n"
            f"Median: {median_val:.2f}\n"
            f"Std Dev: {std_val:.2f}\n"
            f"Count: {count_val}"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1),
                fontsize=10)
        
        # Add caption for outliers
        if outliers_removed > 0:
            ax.text(0.02, -0.15, f"Note: {outliers_removed} outliers removed (IQR method)", transform=ax.transAxes,
                    fontsize=8, color='gray')
        
        # Adjust layout for publication
        plt.tight_layout()
        
        # Save figure to a buffer for download
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        figs.append((fig, buf))
    
    stats_df = pd.DataFrame(stats_data)
    update_log(f"Generated {len(figs)} histograms for categories: {list(categories_units.keys())}")
    return figs, stats_df

# Database selection
st.header("Select or Upload Database")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
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
    st.session_state.db_file = os.path.join(DB_DIR, db_selection)
    update_log(f"Selected database: {db_selection}")

# Main app logic
if st.session_state.db_file and os.path.exists(st.session_state.db_file):
    st.header("NER Analysis")
    selected_terms = st.multiselect(
        "Select Terms for NER",
        ["electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced-stress", 
         "micro-cracking", "electrolyte degradation", "capacity fade", "fatigue life", "crack propagation"],
        default=["electrode cracking", "SEI formation", "cyclic mechanical damage", "diffusion-induced-stress", "micro-cracking"],
        key="select_terms"
    )
    
    with st.sidebar:
        st.subheader("Analysis Parameters")
        top_n = st.slider("Number of Top Entities", min_value=5, max_value=20, value=10, key="top_n")
        colormap = st.selectbox("Color Map", ["muted"], key="colormap")  # Limited to muted for publication quality
    
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
                    ner_df[["paper_id", "title", "entity_text", "entity_label", "value", "unit", "context", "score"]].head(100),
                    use_container_width=True
                )
                
                ner_csv = ner_df.to_csv(index=False)
                st.download_button("Download NER Data CSV", ner_csv, "ner_data.csv", "text/csv", key="download_ner")
                
                st.subheader("Value Distribution Histograms")
                categories_units = {
                    "Deformation": "%",
                    "Fatigue": "cycles",
                    "Crack and Fracture": "μm",
                    "Degradation": "%"
                }
                figs, stats_df = plot_ner_value_histograms(ner_df, categories_units, top_n, colormap)
                if figs:
                    for fig, buf in figs:
                        st.pyplot(fig)
                        st.download_button(
                            label=f"Download {fig.axes[0].get_title().split('\n')[0]} PNG",
                            data=buf,
                            file_name=f"{fig.axes[0].get_title().split('\n')[0].replace(' ', '_').lower()}.png",
                            mime="image/png"
                        )
                    st.subheader("Statistical Summary")
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.warning("No numerical values available for histogram plotting.")
    
    st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="ner_logs")

# Notes
st.markdown("""
---
**Notes**
- **Purpose**: Performs robust NER to categorize quantities with units (e.g., '100 MPa' → Deformation, '28 cycles' → Fatigue) using a statistical MLPClassifier trained on SciBERT embeddings and context features.
- **Database**: Requires a SQLite database with a `papers` table containing `id`, `title`, `year`, and `content` columns.
- **NER Improvements**: Uses a trained classifier to avoid mislabeling (e.g., '100 MPa' as '%', '28 cycles' as '%'), with strict unit and range validation.
- **Visualizations**: Publication-quality histograms show value distributions per category-unit pair, with statistical summaries (mean, median, std, count), outlier filtering, and high-resolution output (300 DPI).
- **Dependencies**: Install `spacy`, `transformers`, `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `streamlit`.
- Ensure `battery_reliability_universe.db` is in the script's directory or upload a custom .db file.
- Check `battery_reliability_ner.log` for detailed logs.
""")