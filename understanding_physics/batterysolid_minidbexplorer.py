import streamlit as st
import sqlite3
import os
import pandas as pd
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Utility to read DB
def load_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn

# Find all .db files in working directory
def get_db_files():
    return [f for f in os.listdir(".") if f.endswith(".db")]

st.title("🔋 Lithium Battery DB Explorer")

# --- Database selection ---
st.sidebar.subheader("📂 Database Selection")

# Explicit DB files we care about
default_db_files = [
    "lithiumbattery_minimetadata.db",
    "lithiumbattery_miniuniverse.db"
]

# Check which ones actually exist in current folder
available_dbs = [f for f in default_db_files if os.path.exists(f)]

# Add uploader option
uploaded_db = st.sidebar.file_uploader("Or upload a .db file", type=["db"])

if uploaded_db:
    # Save uploaded file temporarily
    temp_db_path = os.path.join("temp_uploaded.db")
    with open(temp_db_path, "wb") as f:
        f.write(uploaded_db.getbuffer())
    selected_db = temp_db_path
    st.sidebar.success(f"✅ Loaded uploaded DB: {uploaded_db.name}")
elif available_dbs:
    selected_db = st.sidebar.selectbox("Select Database File", available_dbs)
else:
    st.error("❌ No lithium battery .db files found. Please upload one or place it in the script folder.")
    st.stop()

# Open connection
conn = load_db(selected_db)


# Tabs
tab1, tab2, tab3 = st.tabs(["📂 Inspect Data", "🧩 Common Term Analysis", "🔎 NER & Visualizations"])

# -------------------- TAB 1: INSPECTION --------------------
with tab1:
    st.subheader("Inspect Papers Table")
    try:
        papers_df = pd.read_sql_query("SELECT * FROM papers", conn)
        st.dataframe(papers_df)
    except Exception as e:
        st.warning(f"Could not load 'papers' table: {e}")

    st.subheader("Inspect Pages Table")
    try:
        pages_df = pd.read_sql_query("SELECT * FROM pages", conn)
        st.dataframe(pages_df)
    except Exception as e:
        st.warning(f"Could not load 'pages' table: {e}")

# -------------------- TAB 2: SEMANTIC SIMILARITY --------------------
with tab2:
    st.subheader("Semantic Similarity of Papers")
    try:
        papers_df = pd.read_sql_query("SELECT id, fulltext FROM papers", conn)
        if len(papers_df) > 1:
            docs = [nlp(str(text)) for text in papers_df["fulltext"]]
            similarities = []
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    sim = docs[i].similarity(docs[j])
                    similarities.append({
                        "Paper 1": papers_df["id"][i],
                        "Paper 2": papers_df["id"][j],
                        "Similarity": sim
                    })
            sim_df = pd.DataFrame(similarities)
            st.dataframe(sim_df)

            fig = px.imshow(
                pd.pivot_table(sim_df, values="Similarity", index="Paper 1", columns="Paper 2"),
                color_continuous_scale="Blues",
                title="Semantic Similarity Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 papers for similarity analysis.")
    except Exception as e:
        st.warning(f"Could not perform similarity analysis: {e}")

# -------------------- TAB 3: NER + VISUALIZATIONS --------------------
with tab3:
    st.subheader("Named Entity Recognition (NER)")

    try:
        papers_df = pd.read_sql_query("SELECT id, fulltext FROM papers", conn)
        all_entities = []
        entities_per_paper = {}

        for _, row in papers_df.iterrows():
            doc = nlp(str(row["fulltext"]))
            entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "MATERIAL", "PRODUCT", "FAC", "EVENT", "WORK_OF_ART"]]
            all_entities.extend(entities)
            entities_per_paper[row["id"]] = entities

        if not all_entities:
            st.info("No entities found in database text.")
        else:
            # WordCloud
            st.subheader("WordCloud of Entities")
            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_entities))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Histogram
            st.subheader("Entity Frequency Histogram")
            ent_series = pd.Series(all_entities).value_counts().head(20)
            fig = px.bar(ent_series, x=ent_series.index, y=ent_series.values, labels={"x": "Entity", "y": "Count"})
            st.plotly_chart(fig, use_container_width=True)

            # Radar Chart
            st.subheader("Radar Chart of Top Entities")
            top_entities = ent_series.head(8)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=top_entities.values, theta=top_entities.index, fill="toself", name="Entities"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Sunburst
            st.subheader("Hierarchical Sunburst Chart")
            ent_df = pd.DataFrame(ent_series.reset_index())
            ent_df.columns = ["Entity", "Count"]
            ent_df["Group"] = "Entities"
            fig = px.sunburst(ent_df, path=["Group", "Entity"], values="Count")
            st.plotly_chart(fig, use_container_width=True)

            # Network chart
            st.subheader("Entity Co-occurrence Network")
            G = nx.Graph()
            for paper_id, ents in entities_per_paper.items():
                for i, e1 in enumerate(ents):
                    for e2 in ents[i + 1:]:
                        if e1 != e2:
                            G.add_edge(e1, e2)

            pos = nx.spring_layout(G, k=0.3)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

            fig_net = go.Figure()
            fig_net.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none'
                )
            )
            fig_net.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(size=10, color='blue'),
                    hoverinfo='text'
                )
            )
            st.plotly_chart(fig_net, use_container_width=True)

    except Exception as e:
        st.warning(f"NER or visualization failed: {e}")

st.markdown("---")
st.info("This app explores lithium battery DB files: inspection, semantic similarity (spaCy), and NER visualizations.")
