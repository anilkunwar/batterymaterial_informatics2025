import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="SciBERT Relevance Scorer", layout="wide")

@st.cache_resource
def load_scibert():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_scibert()

# Updated target tokens for battery degradation research context
#target_tokens = {"fracture", "fatigue", "degradation", "battery"}
target_tokens = {"fracture", "fatigue", "degradation","crack","failure", "damage", "thermal runaway", "reliability", "battery", "batteries"}

# Available colormaps for Matplotlib
available_cmaps = sorted([m for m in plt.colormaps() if not m.endswith("_r")])

def get_attention_scores(text, max_tokens=None):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions  # tuple of 12 layers

    tokens = inputs["input_ids"][0]
    token_strs = tokenizer.convert_ids_to_tokens(tokens)

    # Use last layer, average over heads
    attn_last_layer = attentions[-1][0].mean(dim=0)  # shape: [seq_len, seq_len]

    # Clean padding tokens
    pad_token_id = tokenizer.pad_token_id
    seq_len = (tokens != pad_token_id).sum().item()
    if max_tokens is not None:
        seq_len = min(seq_len, max_tokens)
    token_strs = token_strs[:seq_len]
    attn_last_layer = attn_last_layer[:seq_len, :seq_len]

    token_scores = {}
    for i, tok in enumerate(token_strs):
        if tok in target_tokens:
            attn_score = attn_last_layer[:, i].sum().item()
            token_scores[tok] = attn_score

    return token_scores, attn_last_layer.numpy(), token_strs, seq_len

def compute_relevance(token_scores):
    if not token_scores:
        return 0.20
    avg_score = np.mean(list(token_scores.values()))
    num_tokens = len(token_scores)
    relevance = min(0.5 + 0.4 * num_tokens * avg_score, 1.0)
    return relevance

def plot_matplotlib_heatmap(attn_matrix, token_strs, cmap, fontsize, show_target_annotations):
    # Set publication-quality styling
    plt.style.use('ggplot')  # Professional style
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # High DPI for publication
    sns.heatmap(
        attn_matrix,
        xticklabels=token_strs,
        yticklabels=token_strs,
        cmap=cmap,
        cbar_kws={"label": "Attention Weight", "pad": 0.05},
        square=True,
        annot=False,  # Annotations handled separately
        vmin=0,
        vmax=1,
        ax=ax
    )

    # Add annotations for target tokens
    if show_target_annotations:
        for i, tok in enumerate(token_strs):
            if tok in target_tokens:
                ax.text(
                    i + 0.5, i + 0.5, f"{tok}\n({attn_matrix[:, i].sum():.2f})",
                    ha="center", va="center", color="red", fontsize=fontsize-2,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
                )

    ax.set_title("Token-to-Token Attention", fontsize=fontsize + 4, pad=20)
    ax.set_xlabel("Key Tokens", fontsize=fontsize)
    ax.set_ylabel("Query Tokens", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2, rotation=45)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Attention Weight", fontsize=fontsize, labelpad=10)
    cbar.ax.tick_params(labelsize=fontsize-2)

    fig.tight_layout()
    return fig

# Streamlit UI
st.title("🔬 SciBERT Relevance Scorer")

# Input section with updated scientific sentence
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Enter Abstract, Sentence, or Phrase",
                        "Mechanical degradation in lithium-ion batteries arises from fracture and fatigue damage during repeated charge-discharge cycling.")
with col2:
    max_tokens = st.slider("Max Tokens to Display", 5, 50, 20, help="Limit the number of tokens in the heatmap for clarity.")
    show_target_annotations = st.checkbox("Highlight Target Tokens in Matplotlib", value=True)

if st.button("Compute Relevance Score"):
    try:
        token_scores, attn_matrix, token_strs, seq_len = get_attention_scores(text, max_tokens=max_tokens)
        relevance = compute_relevance(token_scores)

        # Cache results
        st.session_state["token_scores"] = token_scores
        st.session_state["attn_matrix"] = attn_matrix
        st.session_state["token_strs"] = token_strs
        st.session_state["relevance"] = relevance
        st.session_state["seq_len"] = seq_len

    except Exception as e:
        st.error(f"Error during SciBERT inference: {e}")

# Render results if available
if "attn_matrix" in st.session_state:
    token_scores = st.session_state["token_scores"]
    attn_matrix = st.session_state["attn_matrix"]
    token_strs = st.session_state["token_strs"]
    relevance = st.session_state["relevance"]
    seq_len = st.session_state["seq_len"]

    st.subheader("📌 Token Attention Scores")
    st.json(token_scores)

    st.subheader("📈 Relevance Score")
    st.metric("Relevance Probability", f"{relevance:.3f}")

    # Plotly Heatmap
    st.subheader("🧠 Attention Map (Plotly, Last Layer, Averaged Heads)")
    fig_plotly = go.Figure(data=go.Heatmap(
        z=attn_matrix,
        x=[f"<b>{tok}</b>" if tok in target_tokens else tok for tok in token_strs],
        y=[f"<b>{tok}</b>" if tok in target_tokens else tok for tok in token_strs],
        text=[[f"Query: {token_strs[i]}<br>Key: {token_strs[j]}<br>Attention: {attn_matrix[i, j]:.3f}"
               for j in range(len(token_strs))] for i in range(len(token_strs))],
        hoverinfo="text",
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="Attention Weight", side="right", font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        zmin=0,
        zmax=1
    ))
    fig_plotly.update_layout(
        xaxis=dict(title="Key Tokens", tickangle=-45, tickfont=dict(size=12)),
        yaxis=dict(title="Query Tokens", tickfont=dict(size=12)),
        title="Token-to-Token Attention (Plotly)",
        height=600,
        width=600,
        margin=dict(l=40, r=40, t=80, b=80),
        font=dict(size=14)
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

    # Download Plotly heatmap
    try:
        import plotly.io as pio
        buf_plotly = BytesIO()
        fig_plotly.write_image(buf_plotly, format="png", scale=2)
        st.download_button(
            label="📥 Download Plotly Heatmap (PNG)",
            data=buf_plotly.getvalue(),
            file_name="plotly_attention_heatmap.png",
            mime="image/png"
        )
    except Exception as e:
        st.warning("Unable to generate Plotly heatmap download. Ensure kaleido is installed.")

    # Matplotlib Heatmap Customization
    st.subheader("🎨 Customize Matplotlib Heatmap")
    fontsize = st.slider("Font Size for Labels", 10, 24, 14, help="Adjust font size for Matplotlib heatmap labels.")
    cmap = st.selectbox("Matplotlib Colormap", available_cmaps, index=available_cmaps.index("viridis"))

    # Matplotlib Heatmap
    st.subheader("🖼️ Attention Map (Matplotlib, Last Layer, Averaged Heads)")
    fig_matplotlib = plot_matplotlib_heatmap(attn_matrix, token_strs, cmap, fontsize, show_target_annotations)
    st.pyplot(fig_matplotlib)

    # Download Matplotlib heatmap
    buf_matplotlib = BytesIO()
    fig_matplotlib.savefig(buf_matplotlib, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        label="📥 Download Matplotlib Heatmap (PNG)",
        data=buf_matplotlib.getvalue(),
        file_name="matplotlib_attention_heatmap.png",
        mime="image/png"
    )
