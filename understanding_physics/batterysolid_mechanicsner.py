import streamlit as st
import sqlite3
import os
import pandas as pd
import spacy
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Lithium Battery DB Explorer", layout="wide")

st.title("Lithium Battery Papers – DB Explorer")

# ----------------------------
# Helper: find .db files
# ----------------------------
def list_db_files():
    files = [f for f in os.listdir('.') if f.endswith('.db')]
    return files

db_files = list_db_files()
selected_db = st.selectbox("Select a database file", db_files)

if not selected_db:
    st.warning("No .db files found in this directory. Please upload or place them here.")
    st.stop()

# ----------------------------
# Connect to DB
# ----------------------------
conn = sqlite3.connect(selected_db)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🔎 Inspection", "📚 Common Term Analysis", "🧠 NER Analysis"])

# ----------------------------
# Tab 1: Inspection
# ----------------------------
with tab1:
    st.header("Database Inspection")

    # Papers metadata
    try:
        df_meta = pd.read_sql("SELECT * FROM papers", conn)
        st.subheader("Papers Metadata")
        st.dataframe(df_meta, use_container_width=True)
    except Exception:
        st.info("No 'papers' table found.")

    # Pages sample
    try:
        df_pages = pd.read_sql("SELECT * FROM pages LIMIT 200", conn)
        st.subheader("Pages (sample)")
        st.dataframe(df_pages, use_container_width=True)
    except Exception:
        st.info("No 'pages' table found.")

# ----------------------------
# Tab 2: Common Term Analysis
# ----------------------------
with tab2:
    st.header("Common Term Semantic Similarity")

    nlp = spacy.load("en_core_web_sm")

    # Aggregate text per paper
    try:
        df_pages = pd.read_sql("SELECT paper_checksum, text FROM pages", conn)
    except Exception:
        st.warning("No 'pages' table for term analysis.")
        st.stop()

    grouped = df_pages.groupby("paper_checksum")["text"].apply(lambda x: " ".join(x)).reset_index()
    grouped["doc"] = grouped["text"].apply(nlp)

    # Compute similarities
    sims = []
    for i in range(len(grouped)):
        for j in range(i+1, len(grouped)):
            s = grouped["doc"].iloc[i].similarity(grouped["doc"].iloc[j])
            sims.append({
                "paper_a": grouped["paper_checksum"].iloc[i],
                "paper_b": grouped["paper_checksum"].iloc[j],
                "similarity": s
            })
    df_sims = pd.DataFrame(sims)

    st.subheader("Pairwise Semantic Similarity")
    st.dataframe(df_sims, use_container_width=True)

    fig = px.scatter(df_sims, x="paper_a", y="paper_b", size="similarity",
                     color="similarity", title="Semantic Similarity Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Tab 3: NER Analysis
# ----------------------------
with tab3:
    st.header("Named Entity Recognition (NER) Analysis")

    try:
        df_pages = pd.read_sql("SELECT text FROM pages", conn)
    except Exception:
        st.warning("No 'pages' table for NER.")
        st.stop()

    texts = " ".join(df_pages["text"].dropna().tolist())
    doc = nlp(texts)

    ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "MATERIAL", "PRODUCT", "FAC", "NORP"]]
    counts = Counter(ents).most_common(50)

    # Wordcloud
    st.subheader("Wordcloud of Entities")
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(counts))
    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

    # Histogram
    st.subheader("Entity Frequency Histogram")
    df_counts = pd.DataFrame(counts, columns=["Entity", "Count"])
    fig_hist = px.bar(df_counts, x="Entity", y="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Radar chart
    st.subheader("Radar Chart of Top Entities")
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=df_counts["Count"],
        theta=df_counts["Entity"],
        fill='toself'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig_radar, use_container_width=True)

    # Sunburst
    st.subheader("Hierarchical Sunburst Chart")
    df_counts["Parent"] = "Entities"
    fig_sun = px.sunburst(df_counts, path=["Parent", "Entity"], values="Count")
    st.plotly_chart(fig_sun, use_container_width=True)

    # Network
    st.subheader("Entity Co-occurrence Network")
    G = nx.Graph()
    window_size = 5
    tokens = [t.text for t in doc if not t.is_space]
    for i in range(len(tokens)-window_size):
        window = tokens[i:i+window_size]
        ents_in_window = [t for t in window if t in dict(counts)]
        for a in ents_in_window:
            for b in ents_in_window:
                if a != b:
                    G.add_edge(a, b, weight=G[a][b]['weight']+1 if G.has_edge(a, b) else 1)
    pos = nx.spring_layout(G, k=0.3)
    edge_x, edge_y, node_x, node_y, node_text = [], [], [], [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    fig_net = go.Figure()\n    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')))\n    fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition=\"top center\"))\n    st.plotly_chart(fig_net, use_container_width=True)\n\nst.markdown(\"---\")\nst.info(\"This app explores lithium battery DB files: inspection, semantic similarity (spaCy), and NER visualizations.\")\n```  \n\n⚙️ **Dependencies**: \n```bash\npip install streamlit pandas spacy wordcloud matplotlib plotly networkx\npython -m spacy download en_core_web_sm\n```  \n\nWould you like me to make the **semantic similarity analysis** more focused on lithium-ion battery terms (custom dictionary), instead of generic spaCy embeddings?
