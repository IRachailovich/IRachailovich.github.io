import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import functools
import pandas as pd
import plotly.express as px  
import matplotlib.colors as mcolors
from gaussians import GaussianProcess
from tueplots.constants.color import rgb

# --- 1. Setup & Styling ---
st.set_page_config(page_title="Gaussian Process Visualizer", layout="wide")

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.beamer_moml())
    plt.rcParams.update({'figure.dpi': 200})
except ImportError:
    plt.style.use('ggplot')

white_reds = mcolors.LinearSegmentedColormap.from_list("WhiteReds", ["white", "red"])

# --- 2. Helper Functions ---
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """Robust RBF kernel using broadcasting."""
    sqdist = np.sum((X1 - X2) ** 2, axis=-1)
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)

def zero_mean(x):
    return np.zeros_like(x).flatten() if x.ndim > 1 else np.zeros_like(x)

def get_dynamic_limits(x_data, y_data, pad=2.0):
    """Calculate flexible plot limits based on data."""
    xmin, xmax = -8.0, 8.0
    ymin, ymax = -10.0, 10.0
    
    if len(x_data) > 0:
        xmin = min(xmin, np.min(x_data) - pad)
        xmax = max(xmax, np.max(x_data) + pad)
        ymin = min(ymin, np.min(y_data) - pad)
        ymax = max(ymax, np.max(y_data) + pad)
        
    return (xmin, xmax), (ymin, ymax)

# --- 3. Session State ---
if 'user_data_x' not in st.session_state:
    st.session_state.user_data_x = []
if 'user_data_y' not in st.session_state:
    st.session_state.user_data_y = []
if 'loaded_pool_x' not in st.session_state:
    st.session_state.loaded_pool_x = np.array([])
if 'loaded_pool_y' not in st.session_state:
    st.session_state.loaded_pool_y = np.array([])

# --- 4. Sidebar ---
st.sidebar.header("Kernel Parameters")
length_scale = st.sidebar.slider("Length Scale", 0.1, 5.0, 1.0, 0.1)
sigma_f = st.sidebar.slider("Signal Variance", 0.1, 5.0, 1.0, 0.1)
sigma_noise = st.sidebar.slider("Noise Level", 0.01, 2.0, 0.1, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("2. Visual Style")

# --- NEW CONTROLS FOR SAMPLES ---
show_samples = st.sidebar.checkbox("Show Sampled Functions", value=True)

if show_samples:
    sample_alpha = st.sidebar.slider("Sample Opacity", 0.0, 1.0, 0.6, 0.05)
    sample_lw = st.sidebar.slider("Sample Line Width", 0.1, 3.0, 0.5, 0.1)
    num_samples_slider = st.sidebar.slider("Number of Samples", 1, 50, 25, 1)
else:
    # Defaults if hidden (won't be used, but good for safety)
    sample_alpha = 0.0
    sample_lw = 0.0
    num_samples_slider = 0

st.sidebar.markdown("---")
st.sidebar.header("3. Upload Data Pool")
uploaded_file = st.sidebar.file_uploader("Upload .mat file", type=['mat'])
if uploaded_file is not None:
    try:
        mat_data = scipy.io.loadmat(uploaded_file)
        st.session_state.loaded_pool_x = mat_data["X"].flatten()
        st.session_state.loaded_pool_y = mat_data["Y"][:, 0].flatten()
        st.sidebar.success(f"Loaded {len(st.session_state.loaded_pool_x)} points.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# --- 5. Data Prep ---
X_train = np.array(st.session_state.user_data_x)
Y_train = np.array(st.session_state.user_data_y)
(xlim_min, xlim_max), (ylim_min, ylim_max) = get_dynamic_limits(X_train, Y_train)

# --- 6. Main Layout ---
st.title("Interactive GP: Click & Select")
col_left, col_right = st.columns([1, 2])

# === LEFT COLUMN ===
with col_left:
    st.subheader("Manage Training Data")
    tab_select, tab_click, tab_manual = st.tabs(["Select from File", "Click to Add", "Manual Entry"])
    
    with tab_select:
        if len(st.session_state.loaded_pool_x) > 0:
            df_pool = pd.DataFrame({"X": st.session_state.loaded_pool_x, "Y": st.session_state.loaded_pool_y})
            selection = st.dataframe(df_pool, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row")
            if st.button("Add Selected"):
                indices = selection.selection.rows
                if indices:
                    st.session_state.user_data_x.extend(df_pool.iloc[indices]["X"].tolist())
                    st.session_state.user_data_y.extend(df_pool.iloc[indices]["Y"].tolist())
                    st.rerun()
        else:
            st.info("Upload .mat file first.")

    with tab_click:
        st.write("Click anywhere to add point:")
        x_grid = np.linspace(xlim_min, xlim_max, 80)   
        y_grid = np.linspace(ylim_min, ylim_max, 80) 
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        df_mesh = pd.DataFrame({"x": X_mesh.ravel(), "y": Y_mesh.ravel()})

        fig_click = px.scatter(df_mesh, x="x", y="y", height=400)
        fig_click.update_traces(
            marker=dict(opacity=0.0, size=50, color="red"),
            hovertemplate="Add Point<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>"
        )
        if len(st.session_state.user_data_x) > 0:
            fig_click.add_scatter(
                x=st.session_state.user_data_x, y=st.session_state.user_data_y, 
                mode='markers', marker=dict(color='black', size=10, symbol='x'),
                name="Existing", hoverinfo='skip'
            )
        fig_click.update_layout(
            xaxis=dict(range=[xlim_min, xlim_max], fixedrange=True, title="X"),
            yaxis=dict(range=[ylim_min, ylim_max], fixedrange=True, title="Y"),
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            clickmode='event+select',
            dragmode=False 
        )
        selection_event = st.plotly_chart(
            fig_click, on_select="rerun", selection_mode="points", 
            key="click_mesh_chart", use_container_width=True
        )
        if selection_event and "points" in selection_event.selection:
            points = selection_event.selection["points"]
            if len(points) > 0:
                pt = points[0]
                if pt.get("curve_number") == 0:
                    idx = pt["point_index"]
                    new_x = df_mesh.iloc[idx]["x"]
                    new_y = df_mesh.iloc[idx]["y"]
                    if not st.session_state.user_data_x or (st.session_state.user_data_x[-1] != new_x):
                        st.session_state.user_data_x.append(new_x)
                        st.session_state.user_data_y.append(new_y)
                        st.rerun()

    with tab_manual:
        with st.form("manual"):
            mx = st.number_input("X", 0.0)
            my = st.number_input("Y", 0.0)
            if st.form_submit_button("Add"):
                st.session_state.user_data_x.append(mx)
                st.session_state.user_data_y.append(my)
                st.rerun()

    if st.button("Clear All Points", type="primary"):
        st.session_state.user_data_x = []
        st.session_state.user_data_y = []
        st.rerun()

# === RIGHT COLUMN: GP PLOT ===
with col_right:
    
    k_fn = functools.partial(rbf_kernel, length_scale=length_scale, sigma_f=sigma_f)
    prior = GaussianProcess(m=zero_mean, k=k_fn)
    
    x_pad = (xlim_max - xlim_min) * 0.1 
    x_calc_min = xlim_min - x_pad
    x_calc_max = xlim_max + x_pad
    x_plot = np.linspace(x_calc_min, x_calc_max, 300)[:, None]
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    
    y_span = ylim_max - ylim_min
    calc_y_min = ylim_min - (y_span * 1.0) 
    calc_y_max = ylim_max + (y_span * 1.0) 

    # --- STYLE DICTIONARIES ---
    # We dynamically update these based on the sidebar sliders
    
    # 1. Std Dev (Heatmap)
    std_style = {"label": "Posterior Std Dev", "cmap": "afmhot", "alpha": 1}
    
    # 2. Sampled Functions (The thin grey lines)
    if show_samples:
        sample_style = {"color": rgb.tue_gray, "alpha": sample_alpha, "lw": sample_lw}
        n_samples = num_samples_slider
    else:
        sample_style = None # Passing None disables plotting of samples
        n_samples = 0

    # 3. Dashed Lines (Std Dev bounds)
    std_line_style = {"color": rgb.tue_ai_brightyellow, "linestyle": "--", "lw": 0.5, "alpha": 0.5}

    if len(X_train) == 0:
        st.subheader("GP Prior")
        prior.plot(ax, x_plot, num_samples=n_samples, rng=np.random.default_rng(42), 
                   color="white", f_range=(calc_y_min, calc_y_max), f_resolution=1000,
                   mean_kwargs=({"label": "Posterior Mean"}), 
                   std_kwargs=std_style, 
                   sampled_fun_kwargs=sample_style, # <--- UPDATED
                   std_lines_kwargs=std_line_style)
    else:
        st.subheader(f"GP Posterior ({len(X_train)} points)")
        posterior = prior.condition(X_train, Y_train, Lambda=sigma_noise**2)
        
        posterior.plot(ax, x_plot, num_samples=n_samples, rng=np.random.default_rng(42), 
                       color="white", f_range=(calc_y_min, calc_y_max), f_resolution=1000,
                       mean_kwargs=({"label": "Posterior Mean"}), 
                       std_kwargs=std_style, 
                       sampled_fun_kwargs=sample_style, # <--- UPDATED
                       std_lines_kwargs=std_line_style)
        
        ax.errorbar(X_train, Y_train, yerr=sigma_noise, fmt='o', color=rgb.tue_ai_gray, markersize=2.0)

    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    
    st.pyplot(fig)


