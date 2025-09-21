import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from networkx.algorithms import community
import community as community_louvain  # Louvain method for weighted graphs
from collections import Counter
import numpy as np
import traceback

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
    import re
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
        # Guard missing columns gracefully with .get when possible
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

    # Create two columns for filters
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

    # Category filter
    categories = sorted(nodes_df["category"].dropna().unique()) if "category" in nodes_df.columns else []
    if not categories:
        categories = ["N/A"]
    selected_categories = st.sidebar.multiselect(
        "Filter by category", 
        categories, 
        default=categories
    )

    # Node type filter
    node_types = sorted(nodes_df["type"].dropna().unique()) if "type" in nodes_df.columns else []
    if not node_types:
        node_types = [""]
    selected_types = st.sidebar.multiselect(
        "Filter by node type", 
        node_types, 
        default=node_types
    )

    # Visualization toggles
    st.sidebar.markdown("**Visualization options**")
    show_node_labels = st.sidebar.checkbox("Show node labels", value=True)
    label_size_scale = st.sidebar.slider("Label size (global)", min_value=8, max_value=28, value=12)
    show_edge_labels = st.sidebar.checkbox("Show edge labels (may clutter)", value=False)

    # Layout selection
    layout_choice = st.sidebar.selectbox("Layout", ["Spring (weighted)", "Kamada-Kawai", "Spectral", "Circular"])

    # Filter the graph
    def filter_graph(G, min_weight, min_freq, selected_categories, selected_types):
        G_filtered = nx.Graph()
        
        # Add nodes that meet criteria
        for n, d in G.nodes(data=True):
            if (d.get("frequency", 0) >= min_freq and 
                (d.get("category", "") in selected_categories) and
                (d.get("type", "") in selected_types)):
                G_filtered.add_node(n, **d)
        
        # Add edges that meet criteria
        for u, v, d in G.edges(data=True):
            if (u in G_filtered.nodes and v in G_filtered.nodes and 
                d.get("weight", 0) >= min_weight):
                G_filtered.add_edge(u, v, **d)
        
        return G_filtered

    G_filtered = filter_graph(G, min_weight, min_node_freq, selected_categories, selected_types)

    # Show graph stats
    st.sidebar.markdown(f"**Graph Stats:** {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    # -----------------------
    # 5. Community Detection (Louvain for weighted graphs)
    # -----------------------
    comm_map = {}
    node_colors = []
    if G_filtered.number_of_nodes() > 0 and G_filtered.number_of_edges() > 0:
        # Create a weighted graph for community detection
        G_weighted = nx.Graph()
        for n, d in G_filtered.nodes(data=True):
            G_weighted.add_node(n, **d)
        for u, v, d in G_filtered.edges(data=True):
            G_weighted.add_edge(u, v, weight=d.get("weight", 1))
        
        try:
            partition = community_louvain.best_partition(G_weighted, weight='weight')
            comm_map = partition
        except Exception:
            # fallback to greedy modularity if Louvain fails
            communities = community.greedy_modularity_communities(G_filtered)
            comm_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_map[node] = i

        # Generate distinct colors for communities (cycle if too many)
        unique_comms = sorted(set(comm_map.values()))
        base_colors = px.colors.qualitative.Plotly  # 10 colors
        node_colors = []
        for node in G_filtered.nodes():
            cid = comm_map.get(node, 0)
            color = base_colors[cid % len(base_colors)]
            node_colors.append(color)
    else:
        node_colors = ["lightblue"] * G_filtered.number_of_nodes()

    # -----------------------
    # 6. Node Positions & Visualization
    # -----------------------
    if G_filtered.number_of_nodes() > 0:
        # Layout selection
        if layout_choice == "Kamada-Kawai":
            pos = nx.kamada_kawai_layout(G_filtered)
        elif layout_choice == "Spectral":
            pos = nx.spectral_layout(G_filtered)
        elif layout_choice == "Circular":
            pos = nx.circular_layout(G_filtered)
        else:
            # Spring layout with weight
            pos = nx.spring_layout(G_filtered, k=1, iterations=200, seed=42, weight='weight')
        
        # Prepare node sizes based on frequency
        frequencies = [G_filtered.nodes[node].get('frequency', 1) for node in G_filtered.nodes()]
        min_size, max_size = 12, 48
        if max(frequencies) > min(frequencies):
            node_sizes = [int(min_size + (max_size - min_size) * 
                         (freq - min(frequencies)) / (max(frequencies) - min(frequencies)) )
                         for freq in frequencies]
        else:
            node_sizes = [24] * len(frequencies)
        
        # Create per-edge traces so widths and hover are accurate
        edge_traces = []
        edge_widths = []
        edge_weight_list = []
        for u, v, d in G_filtered.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d.get("weight", 1)
            edge_weight_list.append(w)
        
        # normalize edge widths
        if edge_weight_list:
            ew_min = min(edge_weight_list)
            ew_max = max(edge_weight_list)
        else:
            ew_min = ew_max = 1
        # Build edge traces now (one trace per edge)
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
            edge_widths.append(width)
        
        # Node trace (single trace for markers)
        node_x = []
        node_y = []
        node_text = []
        node_list = list(G_filtered.nodes())
        for node in node_list:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = G_filtered.nodes[node]
            node_text.append(
                f"{node}<br>"
                f"Category: {node_data.get('category', 'N/A')}<br>"
                f"Type: {node_data.get('type', 'N/A')}<br>"
                f"Frequency: {node_data.get('frequency', 'N/A')}<br>"
                f"Unit: {node_data.get('unit', 'N/A')}"
            )

        # Optionally prepare edge label traces (as mid-point text)
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

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_node_labels else 'markers',
            hoverinfo='text',
            text=[n for n in node_list] if show_node_labels else None,
            textposition="top center",
            textfont=dict(size=label_size_scale),
            texttemplate=None,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line_width=1.5
            )
        )
        
        # Build figure
        traces = []
        traces.extend(edge_traces)
        traces.append(node_trace)
        if show_edge_labels:
            traces.extend(edge_label_traces)
        
        fig = go.Figure(data=traces)

        # Update layout
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

    # -----------------------
    # 7. Hub Nodes and Analysis
    # -----------------------
    if G_filtered.number_of_nodes() > 0:
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G_filtered)
        betweenness_centrality = nx.betweenness_centrality(G_filtered)
        closeness_centrality = nx.closeness_centrality(G_filtered)
        
        # Create a DataFrame with centrality measures
        centrality_df = pd.DataFrame({
            'node': list(degree_centrality.keys()),
            'degree': list(degree_centrality.values()),
            'betweenness': [betweenness_centrality.get(n, 0) for n in degree_centrality.keys()],
            'closeness': [closeness_centrality.get(n, 0) for n in degree_centrality.keys()]
        })
        
        # Add node attributes
        centrality_df['category'] = centrality_df['node'].apply(
            lambda x: G_filtered.nodes[x].get('category', 'N/A'))
        centrality_df['type'] = centrality_df['node'].apply(
            lambda x: G_filtered.nodes[x].get('type', 'N/A'))
        centrality_df['frequency'] = centrality_df['node'].apply(
            lambda x: G_filtered.nodes[x].get('frequency', 0))
        
        # Display top hub nodes
        st.sidebar.subheader("🔑 Top Hub Nodes")
        top_hubs = centrality_df.nlargest(10, 'degree')[['node', 'degree']]
        for _, row in top_hubs.iterrows():
            st.sidebar.write(f"**{row['node']}** ({row['degree']:.3f})")
        
        # -----------------------
        # 8. Node Details Panel
        # -----------------------
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
            
            # Centrality measures
            node_centrality = centrality_df[centrality_df['node'] == selected_node]
            if not node_centrality.empty:
                st.sidebar.write(f"**Degree Centrality:** {node_centrality['degree'].values[0]:.3f}")
                st.sidebar.write(f"**Betweenness Centrality:** {node_centrality['betweenness'].values[0]:.3f}")
                st.sidebar.write(f"**Closeness Centrality:** {node_centrality['closeness'].values[0]:.3f}")
            
            # Neighbors
            neighbors = list(G_filtered.neighbors(selected_node))
            if neighbors:
                st.sidebar.write("**Connected Terms:**")
                for n in neighbors:
                    w = G_filtered.edges[selected_node, n].get("weight", 1)
                    rel_type = G_filtered.edges[selected_node, n].get("type", "")
                    st.sidebar.write(f"- {n} ({rel_type}, weight: {w})")
            else:
                st.sidebar.write("No connected terms above current filter threshold.")
        
        # -----------------------
        # 9. Additional Analytics
        # -----------------------
        st.subheader("📊 Graph Analytics")
        
        # Use filtered nodes/edges for analytics so charts reflect current view
        filtered_nodes_df = pd.DataFrame([
            {
                "node": n,
                "category": d.get("category", "N/A"),
                "type": d.get("type", "N/A"),
                "frequency": d.get("frequency", 0)
            }
            for n, d in G_filtered.nodes(data=True)
        ])
        
        filtered_edges_df = pd.DataFrame([
            {
                "source": u, "target": v,
                "weight": d.get("weight", 0),
                "type": d.get("type", ""),
                "label": d.get("label","")
            }
            for u, v, d in G_filtered.edges(data=True)
        ])
        
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
        
        # Edge type distribution (filtered)
        if not filtered_edges_df.empty:
            edge_type_counts = filtered_edges_df['type'].value_counts()
            fig_edge = px.bar(x=edge_type_counts.index, y=edge_type_counts.values,
                             title="Edge Type Distribution (filtered)",
                             labels={'x': 'Edge Type', 'y': 'Count'})
            st.plotly_chart(fig_edge, use_container_width=True)
        else:
            st.info("No edges to show in edge-type chart.")
        
        # Community analysis
        if comm_map:
            st.subheader("👥 Community Analysis")
            # Build community summary only using nodes present in comm_map & filtered graph
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
                    
                    # Show most common categories in this community
                    categories_in_comm = [G_filtered.nodes[node].get('category', '') for node in nodes]
                    category_counter = Counter(categories_in_comm)
                    st.write("**Top categories:**")
                    for cat, count in category_counter.most_common(5):
                        st.write(f"- {cat}: {count} nodes")
        else:
            st.info("Community detection did not produce communities for the current filter.")

    # -----------------------
    # 10. Data Export
    # -----------------------
    st.sidebar.subheader("💾 Export Data")
    if st.sidebar.button("Export Filtered Graph as CSV"):
        # Create filtered nodes and edges DataFrames
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
        
        # Convert to DataFrames
        nodes_export = pd.DataFrame(filtered_nodes)
        edges_export = pd.DataFrame(filtered_edges)
        
        # Provide download buttons
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
