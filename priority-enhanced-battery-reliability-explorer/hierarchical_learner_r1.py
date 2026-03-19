import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from networkx.algorithms import community
import community as community_louvain
from collections import Counter
import numpy as np
import traceback
import re

# -----------------------
# 1. Data Loading with Caching
# -----------------------
DB_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

@st.cache_data
def load_data():
    edges_path = os.path.join(DB_DIR, 'knowledge_graph_edges.csv')
    nodes_path = os.path.join(DB_DIR, 'knowledge_graph_nodes.csv')
    
    if not os.path.exists(edges_path) or not os.path.exists(nodes_path):
        st.error("❌ One or both CSV files are missing. Please upload 'knowledge_graph_edges.csv' and 'knowledge_graph_nodes.csv'.")
        st.stop()
    
    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(nodes_path)
    return edges_df, nodes_df

try:
    # Load data
    edges_df, nodes_df = load_data()

    # -----------------------
    # 2. Normalize Terms
    # -----------------------
    def normalize_term(term: str) -> str:
        if not isinstance(term, str):
            return ""
        term = term.lower().strip()
        replacements = {
            "batteries": "battery",
            "materials": "material",
            "mater": "material",
            "lithium ions": "lithium ion",
            "lithium ion(s)": "lithium ion",
            "fatigue": "fatigue",
        }
        return replacements.get(term, term)

    nodes_df["node"] = nodes_df["node"].apply(normalize_term)
    edges_df["source"] = edges_df["source"].apply(normalize_term)
    edges_df["target"] = edges_df["target"].apply(normalize_term)

    # -----------------------
    # 3. Build Graph
    # -----------------------
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["node"],
                   type=row["type"],
                   category=row["category"],
                   frequency=row["frequency"] if "frequency" in row else 0,
                   unit=row.get("unit", "None"),
                   similarity_score=row.get("similarity_score", 0))
    
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"],
                   weight=row.get("weight", 1),
                   type=row.get("type", ""),
                   label=row.get("label", ""),
                   relationship=row.get("relationship", ""),
                   strength=row.get("strength", 0))

    # -----------------------
    # 4. Sidebar Controls
    # -----------------------
    st.title("🔋 Battery Research Knowledge Graph Explorer")
    st.markdown("""
    Explore key concepts in **battery mechanical degradation research**.
    - **Nodes** = Terms (colored by cluster).
    - **Size** = Term frequency.
    - **Edges** = Relationships (thicker = stronger).
    Click a node in the sidebar to explore its details.
    """)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        edges_min_value = int(edges_df["weight"].min()) if "weight" in edges_df.columns else 1
        edges_max_value = int(edges_df["weight"].max()) if "weight" in edges_df.columns else 10
        min_weight = st.slider(
            "Min edge weight",
            min_value=edges_min_value,
            max_value=edges_max_value,
            value=max(edges_min_value, min(10, edges_max_value)), step=1
        )
        
    with col2:
        node_min_value = int(nodes_df["frequency"].min()) if "frequency" in nodes_df.columns else 0
        node_max_value = int(nodes_df["frequency"].max()) if "frequency" in nodes_df.columns else 100
        min_node_freq = st.slider(
            "Min node frequency",
            min_value=node_min_value,
            max_value=node_max_value,
            value=max(node_min_value, min(5, node_max_value)), step=1
        )

    categories = sorted(nodes_df["category"].dropna().unique()) if "category" in nodes_df.columns else []
    if not categories:
        categories = ["N/A"]
    selected_categories = st.sidebar.multiselect(
        "Filter by category",
        categories,
        default=categories
    )

    node_types = sorted(nodes_df["type"].dropna().unique()) if "type" in nodes_df.columns else []
    if not node_types:
        node_types = [""]
    selected_types = st.sidebar.multiselect(
        "Filter by node type",
        node_types,
        default=node_types
    )

    st.sidebar.markdown("**Visualization options**")
    show_node_labels = st.sidebar.checkbox("Show node labels", value=True)
    label_size_scale = st.sidebar.slider("Label size (global)", min_value=8, max_value=28, value=12)
    show_edge_labels = st.sidebar.checkbox("Show edge labels (may clutter)", value=False)

    layout_choice = st.sidebar.selectbox("Layout", ["Spring (weighted)", "Kamada-Kawai", "Spectral", "Circular"])

    # -----------------------
    # 5. Filter Graph
    # -----------------------
    def filter_graph(G, min_weight, min_freq, selected_categories, selected_types):
        G_filtered = nx.Graph()
        for n, d in G.nodes(data=True):
            if (d.get("frequency", 0) >= min_freq and
                d.get("category", "") in selected_categories and
                d.get("type", "") in selected_types):
                G_filtered.add_node(n, **d)
        for u, v, d in G.edges(data=True):
            if (u in G_filtered.nodes and v in G_filtered.nodes and
                d.get("weight", 0) >= min_weight):
                G_filtered.add_edge(u, v, **d)
        return G_filtered

    G_filtered = filter_graph(G, min_weight, min_node_freq, selected_categories, selected_types)

    # Immediately create filtered DataFrames for later use
    filtered_nodes_df = pd.DataFrame([
        {"node": n, "category": d.get("category", "N/A"), "type": d.get("type", "N/A"), "frequency": d.get("frequency", 0)}
        for n, d in G_filtered.nodes(data=True)
    ])
    filtered_edges_df = pd.DataFrame([
        {"source": u, "target": v, "weight": d.get("weight", 0), "type": d.get("type", ""), "label": d.get("label", "")}
        for u, v, d in G_filtered.edges(data=True)
    ])

    # Centrality measures (for radar and node details)
    if G_filtered.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G_filtered)
        betweenness_centrality = nx.betweenness_centrality(G_filtered)
        closeness_centrality = nx.closeness_centrality(G_filtered)
        centrality_df = pd.DataFrame({
            'node': list(degree_centrality.keys()),
            'degree': list(degree_centrality.values()),
            'betweenness': [betweenness_centrality.get(n, 0) for n in degree_centrality.keys()],
            'closeness': [closeness_centrality.get(n, 0) for n in degree_centrality.keys()]
        })
        centrality_df['category'] = centrality_df['node'].apply(lambda x: G_filtered.nodes[x].get('category', 'N/A'))
        centrality_df['type'] = centrality_df['node'].apply(lambda x: G_filtered.nodes[x].get('type', 'N/A'))
        centrality_df['frequency'] = centrality_df['node'].apply(lambda x: G_filtered.nodes[x].get('frequency', 0))
    else:
        centrality_df = pd.DataFrame()

    st.sidebar.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # -----------------------
    # 6. Community Detection
    # -----------------------
    comm_map = {}
    node_colors = []
    if G_filtered.number_of_nodes() > 0 and G_filtered.number_of_edges() > 0:
        G_weighted = nx.Graph()
        for n, d in G_filtered.nodes(data=True):
            G_weighted.add_node(n, **d)
        for u, v, d in G_filtered.edges(data=True):
            G_weighted.add_edge(u, v, weight=d.get("weight", 1))
        
        try:
            partition = community_louvain.best_partition(G_weighted, weight='weight')
            comm_map = partition
        except Exception:
            communities = community.greedy_modularity_communities(G_filtered)
            comm_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_map[node] = i

        unique_comms = sorted(set(comm_map.values()))
        base_colors = px.colors.qualitative.Plotly
        node_colors = []
        for node in G_filtered.nodes():
            cid = comm_map.get(node, 0)
            color = base_colors[cid % len(base_colors)]
            node_colors.append(color)
    else:
        node_colors = ["lightblue"] * G_filtered.number_of_nodes()

    # -----------------------
    # 7. Node Positions & Graph Visualization
    # -----------------------
    if G_filtered.number_of_nodes() > 0:
        if layout_choice == "Kamada-Kawai":
            pos = nx.kamada_kawai_layout(G_filtered)
        elif layout_choice == "Spectral":
            pos = nx.spectral_layout(G_filtered)
        elif layout_choice == "Circular":
            pos = nx.circular_layout(G_filtered)
        else:
            pos = nx.spring_layout(G_filtered, k=1, iterations=200, seed=42, weight='weight')

        frequencies = [G_filtered.nodes[node].get('frequency', 1) for node in G_filtered.nodes()]
        min_size, max_size = 12, 48
        if max(frequencies) > min(frequencies):
            node_sizes = [int(min_size + (max_size - min_size) *
                         (freq - min(frequencies)) / (max(frequencies) - min(frequencies)))
                         for freq in frequencies]
        else:
            node_sizes = [24] * len(frequencies)

        # Edge traces (one per edge for variable width)
        edge_traces = []
        edge_weight_list = [d.get("weight", 1) for u, v, d in G_filtered.edges(data=True)]
        ew_min, ew_max = (min(edge_weight_list), max(edge_weight_list)) if edge_weight_list else (1, 1)

        for u, v, d in G_filtered.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d.get("weight", 1)
            if ew_max > ew_min:
                width = 0.8 + 4.2 * (w - ew_min) / (ew_max - ew_min)
            else:
                width = 2.0
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=width, color="#888"),
                    hoverinfo='text',
                    text=f"{u} — {v}<br>weight: {w}<br>type: {d.get('type','')}"
                )
            )

        # Node trace
        node_x, node_y, node_text = [], [], []
        for node in G_filtered.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            nd = G_filtered.nodes[node]
            node_text.append(
                f"{node}<br>Category: {nd.get('category', 'N/A')}<br>Type: {nd.get('type', 'N/A')}<br>"
                f"Frequency: {nd.get('frequency', 'N/A')}<br>Unit: {nd.get('unit', 'N/A')}"
            )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_node_labels else 'markers',
            hoverinfo='text',
            text=[n for n in G_filtered.nodes()] if show_node_labels else None,
            textposition="top center",
            textfont=dict(size=label_size_scale),
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line_width=1.5
            )
        )

        # Edge label traces (optional)
        edge_label_traces = []
        if show_edge_labels:
            for u, v, d in G_filtered.edges(data=True):
                x0, y0 = pos[u]; x1, y1 = pos[v]
                xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                edge_label_traces.append(
                    go.Scatter(
                        x=[xm], y=[ym],
                        mode='text',
                        text=[d.get("label", "") or d.get("relationship","")],
                        textfont=dict(size=10),
                        hoverinfo='none',
                        showlegend=False
                    )
                )

        fig = go.Figure(data=edge_traces + [node_trace] + edge_label_traces)
        fig.update_layout(
            title='Battery Research Knowledge Graph',
            title_font_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Interactive graph. Hover for details, click on nodes to explore connections.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No nodes match the current filter criteria. Try adjusting the filters.")

    # ===================== NEW TABS =====================
    st.header("🔬 Advanced Damage Categorization & Hierarchy Visuals")
    tab_radar, tab_sunburst, tab_sankey = st.tabs([
        "📡 Radar Charts (Comparison)",
        "🌞 Sunburst (Hierarchy)",
        "🔀 Sankey (Flows + Quantities)"
    ])

    # --------------------- TAB 1: RADAR ---------------------
    with tab_radar:
        st.subheader("Radar: Compare Damage Types Across Dimensions")
        if not filtered_nodes_df.empty:
            # Group by category (adjust if your "damage types" are in another column)
            categories = filtered_nodes_df['category'].value_counts().nlargest(6).index.tolist()
            
            radar_traces = []
            for cat in categories:
                subset = filtered_nodes_df[filtered_nodes_df['category'] == cat]
                deg_avg = centrality_df[centrality_df['category'] == cat]['degree'].mean() if not centrality_df[centrality_df['category'] == cat].empty else 0
                
                radar_traces.append(go.Scatterpolar(
                    r=[
                        subset['frequency'].sum(),      # Total mentions
                        len(subset),                    # Number of variants
                        deg_avg * 10,                   # Centrality (scaled)
                        subset['frequency'].mean() * 5  # Average frequency (scaled)
                    ],
                    theta=["Total Frequency", "Num Variants", "Centrality", "Avg Frequency"],
                    name=cat,
                    fill='toself',
                    line=dict(width=2)
                ))
            
            fig_radar = go.Figure(data=radar_traces)
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, filtered_nodes_df['frequency'].max()*1.2])),
                title="Damage Category Profiles (Radar)",
                height=700
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Add columns like 'severity_score' or 'remedy_cost' to nodes.csv for even richer radars!")
        else:
            st.info("No nodes available for radar chart.")

    # --------------------- TAB 2: SUNBURST ---------------------
    with tab_sunburst:
        st.subheader("Sunburst: Damage → Cause → Remedy Hierarchy")
        if not filtered_nodes_df.empty:
            # Simple hierarchy using category as parent (Category → Node)
            sunburst_df = filtered_nodes_df.copy()
            sunburst_df['parent'] = sunburst_df['category']   # Level 1
            
            fig_sun = px.sunburst(
                sunburst_df,
                names='node',
                parents='parent',
                values='frequency',
                title="Hierarchical Breakdown (Category → Damage Type)",
                color='frequency',
                color_continuous_scale='Viridis'
            )
            fig_sun.update_layout(height=800)
            st.plotly_chart(fig_sun, use_container_width=True)
            
            st.info("✅ To get full hierarchy (Damage → Cause → Remedy): add a 'parent_node' column to nodes.csv or filter edges by relationship type='caused_by'/'remedied_by'. I can give you the exact code if you want.")
        else:
            st.info("No nodes available for sunburst chart.")

    # --------------------- TAB 3: SANKEY ---------------------
    with tab_sankey:
        st.subheader("Sankey: Damage → Causes → Remedies (with quantities)")
        if G_filtered.number_of_nodes() > 0 and G_filtered.number_of_edges() > 0:
            node_list = list(G_filtered.nodes())
            node_idx = {node: i for i, node in enumerate(node_list)}
            
            sources, targets, values = [], [], []
            for u, v, d in G_filtered.edges(data=True):
                sources.append(node_idx[u])
                targets.append(node_idx[v])
                values.append(d.get('weight', 1))  # or use frequency if you prefer
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_list,
                    color=node_colors  # re-uses community colors!
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="#888"
                )
            )])
            
            fig_sankey.update_layout(
                title_text="Battery Degradation Flow (width = quantity)",
                font_size=10,
                height=800
            )
            st.plotly_chart(fig_sankey, use_container_width=True)
            st.caption("Tip: Add a sidebar filter for edge type='caused_by' or 'remedied_by' to focus only on cause/remedy flows.")
        else:
            st.info("No edges available for Sankey diagram.")

    # ----------------------- 8. Hub Nodes and Node Details -----------------------
    if G_filtered.number_of_nodes() > 0 and not centrality_df.empty:
        st.sidebar.subheader("🔑 Top Hub Nodes")
        top_hubs = centrality_df.nlargest(10, 'degree')[['node', 'degree']]
        for _, row in top_hubs.iterrows():
            st.sidebar.write(f"**{row['node']}** ({row['degree']:.3f})")

        st.sidebar.subheader("🔍 Explore Node Details")
        selected_node = st.sidebar.selectbox("Choose a node", sorted(G_filtered.nodes()))
        if selected_node:
            node_data = G_filtered.nodes[selected_node]
            st.sidebar.markdown(f"### {selected_node.title()}")
            st.sidebar.write(f"**Category:** {node_data.get('category','N/A')}")
            st.sidebar.write(f"**Type:** {node_data.get('type','N/A')}")
            st.sidebar.write(f"**Frequency:** {node_data.get('frequency', 'N/A')}")
            st.sidebar.write(f"**Unit:** {node_data.get('unit', 'N/A')}")
            st.sidebar.write(f"**Similarity Score:** {node_data.get('similarity_score', 'N/A')}")
            
            node_cent = centrality_df[centrality_df['node'] == selected_node]
            if not node_cent.empty:
                st.sidebar.write(f"**Degree Centrality:** {node_cent['degree'].values[0]:.3f}")
                st.sidebar.write(f"**Betweenness Centrality:** {node_cent['betweenness'].values[0]:.3f}")
                st.sidebar.write(f"**Closeness Centrality:** {node_cent['closeness'].values[0]:.3f}")
            
            neighbors = list(G_filtered.neighbors(selected_node))
            if neighbors:
                st.sidebar.write("**Connected Terms:**")
                for n in neighbors:
                    w = G_filtered.edges[selected_node, n].get("weight", 1)
                    rel_type = G_filtered.edges[selected_node, n].get("type", "")
                    st.sidebar.write(f"- {n} ({rel_type}, weight: {w})")
            else:
                st.sidebar.write("No connected terms above current filter threshold.")

    # ----------------------- 9. Additional Analytics -----------------------
    st.subheader("📊 Graph Analytics")

    col1, col2 = st.columns(2)
    with col1:
        if not filtered_nodes_df.empty:
            category_counts = filtered_nodes_df['category'].value_counts()
            fig_cat = px.pie(values=category_counts.values, names=category_counts.index,
                            title="Node Distribution by Category (filtered)")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No nodes to show in category chart.")
    
    with col2:
        if not filtered_nodes_df.empty:
            top_nodes = filtered_nodes_df.nlargest(10, 'frequency')[['node', 'frequency']]
            fig_nodes = px.bar(top_nodes, x='frequency', y='node', orientation='h',
                              title="Top Nodes by Frequency (filtered)")
            st.plotly_chart(fig_nodes, use_container_width=True)
        else:
            st.info("No nodes to show in top-nodes chart.")

    if not filtered_edges_df.empty:
        edge_type_counts = filtered_edges_df['type'].value_counts()
        fig_edge = px.bar(x=edge_type_counts.index, y=edge_type_counts.values,
                         title="Edge Type Distribution (filtered)",
                         labels={'x': 'Edge Type', 'y': 'Count'})
        st.plotly_chart(fig_edge, use_container_width=True)
    else:
        st.info("No edges to show in edge-type chart.")

    if comm_map:
        st.subheader("👥 Community Analysis")
        community_summary = {}
        for node, comm_id in comm_map.items():
            if node not in G_filtered.nodes():
                continue
            community_summary.setdefault(comm_id, []).append(node)
        
        for comm_id, nodes in sorted(community_summary.items(), key=lambda x: (len(x[1]), x[0]), reverse=True):
            with st.expander(f"Community {comm_id} ({len(nodes)} nodes)"):
                st.write("**Nodes:** " + ", ".join(nodes[:20]))
                if len(nodes) > 20:
                    st.write(f"*... and {len(nodes) - 20} more*")
                
                categories_in_comm = [G_filtered.nodes[node].get('category', '') for node in nodes]
                category_counter = Counter(categories_in_comm)
                st.write("**Top categories:**")
                for cat, count in category_counter.most_common(5):
                    st.write(f"- {cat}: {count} nodes")
    else:
        st.info("Community detection did not produce communities for the current filter.")

    # ----------------------- 10. Data Export -----------------------
    st.sidebar.subheader("💾 Export Data")
    if st.sidebar.button("Export Filtered Graph as CSV"):
        filtered_nodes = []
        for node, data in G_filtered.nodes(data=True):
            filtered_nodes.append({
                'node': node,
                'type': data.get('type', ''),
                'category': data.get('category', ''),
                'frequency': data.get('frequency', 0),
                'unit': data.get('unit', ''),
                'similarity_score': data.get('similarity_score', 0)
            })
        
        filtered_edges = []
        for u, v, data in G_filtered.edges(data=True):
            filtered_edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0),
                'type': data.get('type', ''),
                'label': data.get('label', ''),
                'relationship': data.get('relationship', ''),
                'strength': data.get('strength', 0)
            })
        
        nodes_export = pd.DataFrame(filtered_nodes)
        edges_export = pd.DataFrame(filtered_edges)
        
        st.sidebar.download_button(
            label="Download Nodes CSV",
            data=nodes_export.to_csv(index=False),
            file_name="filtered_nodes.csv",
            mime="text/csv"
        )
        st.sidebar.download_button(
            label="Download Edges CSV",
            data=edges_export.to_csv(index=False),
            file_name="filtered_edges.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.text("Detailed error information:")
    st.text(traceback.format_exc())
