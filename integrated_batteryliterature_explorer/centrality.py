import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# Sidebar controls
# ==========================
st.sidebar.header("Radar Chart Settings")

font_size = st.sidebar.slider("Font Size", 8, 24, 12)  # int
line_width = st.sidebar.slider("Line Thickness", 1, 6, 2)  # int
grid_width = st.sidebar.slider("Grid Line Thickness", 0.5, 3.0, 1.0, step=0.1)  # float
label_padding = st.sidebar.slider("Label Padding (fraction of radius)", 0.05, 1.0, 0.5, step=0.01)  # float
num_grid_levels = st.sidebar.slider("Number of Grid Levels", 2, 6, 5)  # int

# ==========================
# Data
# ==========================
parameters = ["Degree", "Betweenness", "Closeness", "Eigenvector"]
failure_types = ["mechanical", "fracture", "crack", "cycling"]

values = {
    "mechanical": [1.0, 0.0, 1.0, 0.178],
    "fracture": [1.0, 0.0, 1.0, 0.178],
    "crack": [1.0, 0.0, 1.0, 0.178],
    "cycling": [1.0, 0.0, 1.0, 0.178],
}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ==========================
# Function to plot radar chart
# ==========================
def plot_radar(values, title, color):
    N = len(parameters)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]  # Close the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color=color, linewidth=line_width, label=title)
    ax.fill(angles, values, color=color, alpha=0.25)

    # Draw inner grids (excluding max) as dotted lines
    grid_levels = np.linspace(0, 1, num_grid_levels+1)[1:-1]  # skip 0 and 1
    for g in grid_levels:
        ax.plot(np.linspace(0, 2*np.pi, 100), [g]*100, color='grey', linestyle='dotted', linewidth=grid_width)
    
    # Draw outer grid (value=1) as solid
    ax.plot(np.linspace(0, 2*np.pi, 100), [1]*100, color='grey', linestyle='solid', linewidth=grid_width)

    # Set labels with padding
    for label, angle in zip(parameters, angles[:-1]):
        ax.text(
            angle, 
            1.0 + label_padding,  # padding from circumference
            label,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=font_size
        )

    # Radial grid numbers
    for g in np.linspace(0, 1, num_grid_levels+1)[1:]:  # include 1
        ax.text(np.pi/2, g, f"{g:.2f}", horizontalalignment='center',
                verticalalignment='bottom', fontsize=font_size*0.8)

    ax.set_ylim(0, 1)
    ax.set_yticklabels([])  # Hide default radial labels
    ax.set_xticks([])       # Hide default angular labels
    ax.set_title(title, fontsize=font_size + 2, pad=20)
    return fig

# ==========================
# Display four radar charts
# ==========================
st.header("Failure Mechanisms Radar Charts with Grid Labels")
for i, ft in enumerate(failure_types):
    st.subheader(ft.capitalize())
    fig = plot_radar(values[ft][:], ft, colors[i])
    st.pyplot(fig)

