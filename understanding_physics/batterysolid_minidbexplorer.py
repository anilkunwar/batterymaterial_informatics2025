import streamlit as st
import sqlite3
import pandas as pd
import requests
import os
from io import BytesIO
import re
from collections import Counter
from itertools import combinations
import spacy
from spacy.language import Language
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Lithium Battery DB Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Lithium Battery Research DB Analysis")
st.write(
    "Analyze SQLite databases (`lithiumbattery_minimetadata.db` or `lithiumbattery_miniuniverse.db`) "
    "for lithium-ion battery research papers on plasticity, fracture, fatigue, and damage."
)

# -----------------------------
# Load spaCy model
# -----------------------------
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    st.error("Failed to load spaCy model. Please add 'en_core_web_md' (pip install spacy; python -m spacy download en_core_web_md).")
    raise

# -----------------------------
# GitHub repository DB files
# -----------------------------
DB_URLS = {
    "lithiumbattery_minimetadata.db": "https://raw.githubusercontent.com/<your-repo>/<branch>/lithiumbattery_minimetadata.db",
    "lithiumbattery_miniuniverse.db": "https://raw.githubusercontent.com/<your-repo>/<branch>/lithiumbattery_miniuniverse.db",
}

# -----------------------------
# Sidebar: DB selection
# -----------------------------
st.sidebar.header("Database Selection")
db_option = st.sidebar.selectbox(
    "Select or upload a database",
    options=["Select from GitHub"] + list(DB_URLS.keys()) + ["Upload custom .db file"],
    index=0,
)

db_path = None
if db_option == "Upload custom .db file":
    uploaded_db = st.sidebar.file_uploader("Upload SQLite DB file", type=["db", "sqlite", "sqlite3"])
    if uploaded_db:
        with open("temp.db", "wb") as f:
            f.write(uploaded_db.read())
        db_path = "temp.db"
elif db_option in DB_URLS:
    try:
        response = requests.get(DB_URLS[db_option], stream=True)
        if response.status_code == 200:
            db_path = db_option
            with open(db_path, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Failed to fetch {db_option} from GitHub.")
    except Exception as e:
        st.error(f"Error downloading {db_option}: {e}")

# -----------------------------
# Database connection and schema check
# -----------------------------
def get_db_connection(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    return con

def check_schema(con: sqlite3.Connection, expected_tables: List[str]) -> bool:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    return all(t in tables for t in expected_tables)

# -----------------------------
# Tab setup
# -----------------------------
if db_path:
    con = get_db_connection(db_path)
    is_metadata_db = check_schema(con, ["papers", "topic_counts"])
    is_universe_db = check_schema(con, ["pages"])
    if not (is_metadata_db or is_universe_db):
        st.error("Selected DB does not match expected schema (needs 'papers'/'topic_counts' or 'pages').")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Data Inspection", "Term Analysis", "NER Visualizations"])

    # -----------------------------
    # Tab 1: Data Inspection
    # -----------------------------
    with tab1:
        st.header("Data Inspection")
        if is_metadata_db:
            st.subheader("Papers Table")
            df_papers = pd.read_sql_query("SELECT * FROM papers", con)
            st.dataframe(df_papers, use_container_width=True)

            st.subheader("Topic Counts")
            df_topics = pd.read_sql_query("SELECT * FROM topic_counts", con)
            st.dataframe(df_topics, use_container_width=True)

            # Simple stats
            st.write(f"Total papers: {len(df_papers)}")
            st.write(f"Total topic counts: {len(df_topics)}")
            topic_pivot = df_topics.pivot(index="paper_id", columns="topic", values="count").fillna(0)
            st.write("Topic counts per paper:")
            st.dataframe(topic_pivot, use_container_width=True)

        if is_universe_db:
            st.subheader("Pages Table")
            df_pages = pd.read_sql_query("SELECT * FROM pages LIMIT 100", con)
            st.dataframe(df_pages, use_container_width=True)
            st.write(f"Total pages: {pd.read_sql_query('SELECT COUNT(*) FROM pages', con).iloc[0][0]}")

            # Full-text search if available
            try:
                search_query = st.text_input("Search pages (if FTS5 enabled):", "")
                if search_query:
                    df_search = pd.read_sql_query(
                        "SELECT filename, page_num, text, rank FROM page_fts WHERE page_fts MATCH ? ORDER BY rank LIMIT 50",
                        con, params=(search_query,)
                    )
                    st.write("Search results:")
                    st.dataframe(df_search, use_container_width=True)
            except sqlite3.OperationalError:
                st.info("Full-text search not available (FTS5 not enabled in this DB).")

    # -----------------------------
    # Tab 2: Common Term Analysis (Semantic Similarity)
    # -----------------------------
    with tab2:
        st.header("Common Term Analysis")
        if not is_universe_db:
            st.warning("This tab requires 'lithiumbattery_miniuniverse.db' with a 'pages' table.")
        else:
            st.write("Extracting common terms across papers and computing semantic similarity using spaCy.")

            # Load all text
            df_pages = pd.read_sql_query("SELECT filename, text FROM pages", con)
            docs_by_file = df_pages.groupby("filename")["text"].apply(lambda x: " ".join(x)).to_dict()

            # User options
            top_n_terms = st.slider("Number of top terms to analyze", 10, 100, 50)
            similarity_threshold = st.slider("Semantic similarity threshold", 0.0, 1.0, 0.7)

            # Extract terms
            all_terms: Counter = Counter()
            for text in docs_by_file.values():
                doc = nlp(text[:1000000])  # Limit to avoid memory issues
                terms = [
                    token.lemma_.lower() for token in doc
                    if token.is_alpha and not token.is_stop and len(token.text) > 2
                ]
                all_terms.update(terms)
            top_terms = [term for term, _ in all_terms.most_common(top_n_terms)]

            # Compute similarity matrix
            term_docs = [nlp(" ".join([term] * 3)) for term in top_terms]  # Boost term context
            sim_matrix = np.zeros((len(top_terms), len(top_terms)))
            for i, doc_i in enumerate(term_docs):
                for j, doc_j in enumerate(term_docs[i+1:], i+1):
                    sim = doc_i.similarity(doc_j)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

            # Display similarity heatmap
            fig = px.imshow(
                sim_matrix,
                x=top_terms,
                y=top_terms,
                color_continuous_scale="Viridis",
                title="Term Similarity Heatmap",
                range_color=[0, 1],
            )
            fig.update_layout(width=800, height=800)
            st.plotly_chart(fig, use_container_width=True)

            # Terms co-occurring in papers
            st.subheader("Term Co-occurrence Across Papers")
            term_presence: Dict[str, Set[str]] = {term: set() for term in top_terms}
            for filename, text in docs_by_file.items():
                doc = nlp(text[:1000000])
                doc_terms = set(token.lemma_.lower() for token in doc if token.lemma_.lower() in top_terms)
                for term in doc_terms:
                    term_presence[term].add(filename)

            # Build co-occurrence matrix
            co_matrix = np.zeros((len(top_terms), len(top_terms)))
            for i, term_i in enumerate(top_terms):
                for j, term_j in enumerate(top_terms[i+1:], i+1):
                    common_papers = len(term_presence[term_i] & term_presence[term_j])
                    co_matrix[i, j] = common_papers
                    co_matrix[j, i] = common_papers

            fig_co = px.imshow(
                co_matrix,
                x=top_terms,
                y=top_terms,
                color_continuous_scale="Blues",
                title="Term Co-occurrence Across Papers",
                text_auto=".0f",
            )
            fig_co.update_layout(width=800, height=800)
            st.plotly_chart(fig_co, use_container_width=True)

    # -----------------------------
    # Tab 3: NER Visualizations
    # -----------------------------
    with tab3:
        st.header("NER Analysis and Visualizations")
        if not is_universe_db:
            st.warning("This tab requires 'lithiumbattery_miniuniverse.db' with a 'pages' table.")
        else:
            st.write("Performing Named Entity Recognition (NER) and generating visualizations.")

            # NER extraction
            entity_counts: Counter = Counter()
            entity_types: Dict[str, List[str]] = {}
            df_pages = pd.read_sql_query("SELECT filename, text FROM pages", con)
            for _, row in df_pages.iterrows():
                doc = nlp(row["text"][:1000000])
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "DATE"]:
                        entity_counts[ent.text] += 1
                        if ent.label_ not in entity_types:
                            entity_types[ent.label_] = []
                        entity_types[ent.label_].append(ent.text)

            # Word Cloud
            st.subheader("Word Cloud of Entities")
            if entity_counts:
                wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(
                    dict(entity_counts)
                )
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No entities found for word cloud.")

            # Network of Entities
            st.subheader("Entity Co-occurrence Network")
            G = nx.Graph()
            entity_pairs: Counter = Counter()
            for _, row in df_pages.iterrows():
                doc = nlp(row["text"][:1000000])
                entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]]
                for pair in combinations(entities, 2):
                    entity_pairs[tuple(sorted(pair))] += 1

            for (e1, e2), count in entity_pairs.most_common(50):
                G.add_edge(e1, e2, weight=count)

            pos = nx.spring_layout(G)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_sizes = [G.degree(node) * 10 for node in G.nodes()]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.5, color="#888"),
                hoverinfo="none"
            ))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                text=list(G.nodes()), textposition="top center",
                marker=dict(size=node_sizes, color="skyblue"),
                hoverinfo="text"
            ))
            fig.update_layout(showlegend=False, width=800, height=600, title="Entity Co-occurrence Network")
            st.plotly_chart(fig, use_container_width=True)

            # Histogram of Entity Types
            st.subheader("Histogram of Entity Types")
            ent_type_counts = Counter([ent.label_ for ents in entity_types.values() for ent in ents])
            fig_hist = px.bar(
                x=list(ent_type_counts.keys()),
                y=list(ent_type_counts.values()),
                labels={"x": "Entity Type", "y": "Count"},
                title="Entity Type Distribution",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Radar Chart of Entity Types per Paper
            st.subheader("Radar Chart of Entity Types per Paper")
            if is_metadata_db:
                df_papers = pd.read_sql_query("SELECT filename FROM papers", con)
                radar_data = []
                for filename in df_papers["filename"][:5]:  # Limit for performance
                    df_file = df_pages[df_pages["filename"] == filename]
                    text = " ".join(df_file["text"])
                    doc = nlp(text[:1000000])
                    counts = Counter(ent.label_ for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "DATE"])
                    radar_data.append({"filename": filename, **counts})

                if radar_data:
                    fig_radar = go.Figure()
                    for data in radar_data:
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[data.get(cat, 0) for cat in ["PERSON", "ORG", "GPE", "PRODUCT", "DATE"]],
                            theta=["PERSON", "ORG", "GPE", "PRODUCT", "DATE"],
                            name=data["filename"],
                            fill="toself"
                        ))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("No entity data for radar chart.")

            # Hierarchical Sunburst Chart
            st.subheader("Hierarchical Sunburst Chart of Entities")
            sunburst_data = []
            for ent_type, ents in entity_types.items():
                for ent in set(ents):
                    sunburst_data.append({"type": ent_type, "entity": ent, "count": entity_counts[ent]})

            if sunburst_data:
                fig_sunburst = px.sunburst(
                    sunburst_data,
                    path=["type", "entity"],
                    values="count",
                    title="Entity Hierarchy",
                    maxdepth=2,
                )
                st.plotly_chart(fig_sunburst, use_container_width=True)
            else:
                st.info("No entities for sunburst chart.")

    con.close()
    if db_path == "temp.db" and os.path.exists("temp.db"):
        os.remove("temp.db")

else:
    st.warning("Please select or upload a valid SQLite database to proceed.")

st.markdown(
    """
---
**Notes**
- **Data Inspection**: View raw tables and basic stats. Full-text search requires FTS5 in `lithiumbattery_miniuniverse.db`.
- **Term Analysis**: Uses spaCy's `en_core_web_md` for semantic similarity and term co-occurrence across papers.
- **NER Visualizations**: Extracts entities (PERSON, ORG, GPE, PRODUCT, DATE) and visualizes them via word cloud, network, histogram, radar chart, and sunburst chart.
- Replace `<your-repo>/<branch>` in DB URLs with the actual GitHub repository path.
- Visualizations are limited to manageable data sizes for performance (e.g., top 50 terms, top 50 entity pairs).
    """
)
