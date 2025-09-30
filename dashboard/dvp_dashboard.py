import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import xml.etree.ElementTree as ET  # kept for other parts if needed
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🧬 DVP Data Dashboard",
    page_icon="🧬",
    layout="wide"
)

# UI helpers
def set_sidebar_width(percent_wider: int = 20):
    """Widen Streamlit sidebar by approximately `percent_wider` percent.
    Default sidebar is roughly ~300px in many themes; we scale that to keep it simple.
    This affects display only and is safe across pages.
    """
    base_px = 300
    target_px = int(base_px * (1 + percent_wider / 100))
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebar"] {{
                min-width: {target_px}px;
                width: {target_px}px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Resolve project root (parent of the 'dashboard' folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(*parts: str) -> str:
    """Build an absolute path from project root using relative parts."""
    return str(PROJECT_ROOT.joinpath(*parts))

# Define explicit individual ordering and groups
ORDERED_INDIVIDUAL_IDS = [
    "NML4",  # Individual 1
    "NML5",  # Individual 2
    "NML6",  # Individual 3
    "NML7",  # Individual 4
    "NML9",  # Individual 5
    "NML10", # Individual 6
    "NML11", # Individual 7
    "NML15", # Individual 8
    "NML16", # Individual 9
    "NML17", # Individual 10
    "NML20", # Individual 11
    "NML31", # Individual 12
    "CHTL3", # Individual 13
    "CHTL34",# Individual 14 (Disrupted)
    "CHTL36",# Individual 15 (Disrupted)
    "CHTL38",# Individual 16 (Disrupted)
    "CHTL59",# Individual 17 (Disrupted)
    "CHTL73",# Individual 18
]

# Groups based on the above ordering:
# Individuals 14–17 are Disrupted; 1–13 and 18 are Healthy
GROUP_DICT = {
    1: ORDERED_INDIVIDUAL_IDS[:13] + [ORDERED_INDIVIDUAL_IDS[17]],  # Healthy: 1..13, 18
    2: ORDERED_INDIVIDUAL_IDS[13:17],                               # Disrupted: 14..17
}

def build_individual_label_mapping():
    """Return a dict mapping internal individual IDs to display labels
    using the explicit numbering provided by the user.
    """
    id_to_label: dict[str, str] = {}
    for idx, ind in enumerate(ORDERED_INDIVIDUAL_IDS, start=1):
        group = "Disrupted" if (14 <= idx <= 17) else "Healthy"
        id_to_label[ind] = f"Individual {idx} ({group})"
    return id_to_label

@st.cache_data
def load_data():
    """Load and preprocess the DVP data"""
    try:
        # Load intensity data
        intensity_path = project_path("data", "human", "scDVP_filtered.tsv")
        intensity = pd.read_csv(intensity_path, sep="\t")
            
        # Load scores data
        scores_path = project_path("data", "human", "all_scores.tsv")
        scores = pd.read_csv(scores_path, sep="\t")
            
        # Load volcano plot results for controls and disease
        controls_results_path = project_path("results", "human_controls", "results_1_cutoff=0.7.tsv")
        disease_results_path = project_path("results", "human_disrupted", "results_2_cutoff=0.7.tsv")
        
        controls_results = pd.read_csv(controls_results_path, sep="\t")
        disease_results = pd.read_csv(disease_results_path, sep="\t")
        
        return intensity, scores, controls_results, disease_results
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def filter_data_by_group(intensity, scores, group_id):
    """Filter data by individual group"""
    individuals = GROUP_DICT[group_id]
    
    # Filter intensity columns for selected group
    # Each sample column has format: PATIENT_shape_XX
    protein_col = ["protein"]
    sample_cols = [col for col in intensity.columns 
                   if any(col.startswith(individual + "_") for individual in individuals)]
    filtered_intensity = intensity[protein_col + sample_cols]
    
    # Filter scores for selected group  
    # Only keep columns that exist in filtered_intensity
    score_cols = [col for col in sample_cols if col in scores.columns]
    filtered_scores = scores[score_cols]
    
    return filtered_intensity, filtered_scores

@st.cache_data
def load_pickled_shapes(individual_id: str):
    """Load crop-local shapes for an individual from a pickle file.

    Expected format: dict {cap_id: np.ndarray (N_points, 2)} where coordinates are crop-local pixels.
    We convert to an ordered dict keyed by 1..N to align with sample numbering (suffix in column names).
    """
    pkl_path = project_path("dashboard", "statics", "Shapes", f"{individual_id}.pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # Preserve insertion order, map to numeric cell IDs starting at 1
        shapes_idx = {}
        for idx, (cap_id, pts) in enumerate(data.items(), start=1):
            pts = np.asarray(pts)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            shapes_idx[idx] = pts
        return shapes_idx
    except Exception as e:
        st.error(f"Error loading shapes pickle for {individual_id}: {e}")
        return None

def create_volcano_plot(results_df, title, selected_protein=None):
    """Create a volcano plot from results dataframe"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate -log10(p_value) for volcano plot
    neg_log_p = -np.log10(results_df['p_value'].replace(0, 1e-300))
    
    # Colors palette
    color_default = '#2E4053'
    color_central = '#481D6F'  # significant, Central vein (negative effect)
    color_portal = '#F97B19'   # significant, Portal vein (positive effect)

    # Base layer: all points in default color
    ax.scatter(results_df['coefficient'], neg_log_p,
               alpha=0.5, s=18, c=color_default, edgecolors='none')
    
    # Highlight significant points split by direction using p-value and effect size thresholds
    significant_mask = (results_df['p_value'] < 0.05) & ((results_df['coefficient'] < -1) | (results_df['coefficient'] > 1))
    if significant_mask.any():
        sig_df = results_df.loc[significant_mask]
        neg_mask = sig_df['coefficient'] < -1
        pos_mask = sig_df['coefficient'] > 1

        if neg_mask.any():
            ax.scatter(sig_df.loc[neg_mask, 'coefficient'],
                       -np.log10(sig_df.loc[neg_mask, 'p_value'].replace(0, 1e-300)),
                       alpha=0.9, s=26, c=color_central, label='Central vein (q<0.05, |coef|>1)', edgecolors='black', linewidth=0.3)
        if pos_mask.any():
            ax.scatter(sig_df.loc[pos_mask, 'coefficient'],
                       -np.log10(sig_df.loc[pos_mask, 'p_value'].replace(0, 1e-300)),
                       alpha=0.9, s=26, c=color_portal, label='Portal vein (q<0.05, |coef|>1)', edgecolors='black', linewidth=0.3)
    
    # Highlight selected protein if it exists in the results
    if selected_protein and selected_protein in results_df['index'].values:
        protein_idx = results_df['index'] == selected_protein
        protein_coeff = results_df.loc[protein_idx, 'coefficient'].iloc[0]
        protein_neg_log_p = neg_log_p[protein_idx].iloc[0]

        # Highlight the selected protein with a larger, bright marker
        ax.scatter(protein_coeff, protein_neg_log_p,
                   alpha=1.0, s=110, c='gold', edgecolors='black', linewidth=1.2,
                   marker='*', zorder=6, label='Selected protein')

        # Add protein name label next to the marker
        ax.annotate(selected_protein,
                    (protein_coeff, protein_neg_log_p),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'),
                    zorder=6)
    
    # Add significance threshold lines
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.6, label='q = 0.05')
    # Add vertical effect-size cutoffs
    ax.axvline(-1, color='lightgray', linestyle=':', alpha=0.7, label='zonation thresholds')
    ax.axvline(1, color='lightgray', linestyle=':', alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Zonation coefficient (Effect Size | Central → Portal)', fontsize=12)
    ax.set_ylabel('-log10(q-value)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def create_protein_expression_plot(intensity_df, scores_df, protein_name, group_name):
    """Create expression vs spatial score plot for a specific protein"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        # Get protein data
        protein_data = intensity_df[intensity_df['protein'] == protein_name]
        
        if protein_data.empty:
            ax.text(0.5, 0.5, f'Protein {protein_name} not found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'Protein Expression vs Spatial Score - {group_name}', fontweight='bold')
            return fig
        
        # Extract expression values (excluding protein column)
        expression_values = protein_data.iloc[0, 1:].values
        
        # Convert to float, handling empty strings and NaN
        expression_clean_list = []
        for val in expression_values:
            if pd.isna(val) or val == '' or val == 'NaN':
                expression_clean_list.append(np.nan)
            else:
                try:
                    expression_clean_list.append(float(val))
                except (ValueError, TypeError):
                    expression_clean_list.append(np.nan)
        
        expression_values = np.array(expression_clean_list)
        
        # Get sample names (excluding 'protein' column)
        sample_names = intensity_df.columns[1:].tolist()
        
        # Get spatial scores for the same samples
        # Check that we have the score row (row index 1 contains the actual scores)
        if len(scores_df) < 2:
            ax.text(0.5, 0.5, f'Insufficient score data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'Protein Expression vs Spatial Score - {group_name}', fontweight='bold')
            return fig
            
        # Get scores from the second row (index 1)
        score_values = []
        for sample in sample_names:
            if sample in scores_df.columns:
                score_val = scores_df.loc[1, sample]  # Row 1 contains the scores
                try:
                    score_values.append(float(score_val))
                except (ValueError, TypeError):
                    score_values.append(np.nan)
            else:
                score_values.append(np.nan)
        
        spatial_scores = np.array(score_values)
        
        # Remove NaN values from both arrays
        mask = ~(np.isnan(expression_values) | np.isnan(spatial_scores))
        expression_clean = expression_values[mask]
        spatial_clean = spatial_scores[mask]
        
        if len(expression_clean) == 0:
            ax.text(0.5, 0.5, f'No valid data pairs for protein {protein_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'Protein Expression vs Spatial Score - {group_name}', fontweight='bold')
            return fig
        
        # Z-score normalization of expression values
        if len(expression_clean) > 1 and np.std(expression_clean) > 0:
            expression_zscore = (expression_clean - np.mean(expression_clean)) / np.std(expression_clean)
        else:
            expression_zscore = expression_clean - np.mean(expression_clean) if len(expression_clean) > 0 else expression_clean
        
        # Create scatter plot
        sns.scatterplot(x=spatial_clean, y=expression_zscore, ax=ax, color='steelblue', edgecolor='black', s=50, alpha=0.5)

        # Add a LOWESS smoothing curve to show the trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(expression_zscore, spatial_clean, frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color='firebrick', linewidth=2)
        except Exception:
            # Fallback: simple rolling mean after sorting by x
            try:
                order = np.argsort(spatial_clean)
                x_sorted = spatial_clean[order]
                y_sorted = expression_zscore[order]
                # Rolling window size ~ 10% of points, min 5
                window = max(5, int(0.1 * len(x_sorted)))
                y_roll = pd.Series(y_sorted).rolling(window=window, min_periods=max(3, window//2), center=True).mean().to_numpy()
                ax.plot(x_sorted, y_roll, color='firebrick', linewidth=2)
            except Exception:
                pass
        
        # Customize plot
        ax.set_xlabel('Spatial Score (Central → Portal)', fontsize=12)
        ax.set_ylabel('Z-scored Protein Intensity', fontsize=12)
        ax.set_title(
            f'{protein_name} Expression vs Spatial Score - {group_name}',
            fontsize=14,
            fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        
    # Removed correlation/sample count textbox for a cleaner look
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting {protein_name}: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(f'Protein Expression vs Spatial Score - {group_name}', fontweight='bold')
    
    return fig

def create_individual_trajectory_plot(intensity_df, scores_df, protein_name, individual_id, group_name):
    """Create a trajectory scatter plot for a specific protein in a specific individual"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        # Get protein data
        protein_data = intensity_df[intensity_df['protein'] == protein_name]
        
        if protein_data.empty:
            ax.text(0.5, 0.5, f'Protein {protein_name} not found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{protein_name} Trajectory - {individual_id} ({group_name})', fontweight='bold')
            return fig
        
        # Find all sample columns for this individual
        individual_samples = [col for col in intensity_df.columns if col.startswith(individual_id + "_")]
        
        if not individual_samples:
            ax.text(0.5, 0.5, f'No samples found for individual {individual_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{protein_name} Trajectory - {individual_id} ({group_name})', fontweight='bold')
            return fig
        
        # Extract expression values for this individual's samples
        expression_values = []
        spatial_scores = []
        sample_names = []
        
        for sample in individual_samples:
            # Get expression value
            if sample in protein_data.columns:
                expr_val = protein_data[sample].iloc[0]
                if pd.isna(expr_val) or expr_val == '' or expr_val == 'NaN':
                    continue
                try:
                    expr_val = float(expr_val)
                except (ValueError, TypeError):
                    continue
                
                # Get spatial score for this sample
                if sample in scores_df.columns and len(scores_df) >= 2:
                    score_val = scores_df.loc[1, sample]  # Row 1 contains the scores
                    try:
                        score_val = float(score_val)
                    except (ValueError, TypeError):
                        continue
                    
                    expression_values.append(expr_val)
                    spatial_scores.append(score_val)
                    sample_names.append(sample)
        
        if len(expression_values) == 0:
            ax.text(0.5, 0.5, f'No valid data for protein {protein_name} in individual {individual_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{protein_name} Trajectory - {individual_id} ({group_name})', fontweight='bold')
            return fig
        
        # Convert to arrays
        expression_values = np.array(expression_values)
        spatial_scores = np.array(spatial_scores)
        
        # Z-score normalization of expression values
        if len(expression_values) > 1 and np.std(expression_values) > 0:
            expression_zscore = (expression_values - np.mean(expression_values)) / np.std(expression_values)
        else:
            expression_zscore = expression_values - np.mean(expression_values) if len(expression_values) > 0 else expression_values

        # Create scatter plot (no regression line, no legend)
        scatter = sns.scatterplot(
            x=spatial_scores,
            y=expression_zscore,
            ax=ax,
            s=80,
            color='steelblue',
            edgecolor='black',
            linewidth=1,
            alpha=0.85,
            legend=False,
        )
        
        # Customize plot
        ax.set_xlabel('Spatial Score (Central → Portal)', fontsize=12)
        ax.set_ylabel('Z-scored Protein Intensity', fontsize=12)
        ax.set_title(f'{protein_name} Trajectory - {individual_id} ({group_name})', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    # Removed sample count textbox for a cleaner look
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting {protein_name} for {individual_id}: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{protein_name} Trajectory - {individual_id} ({group_name})', fontweight='bold')
    
    return fig

def create_cell_shapes_plot(results_df, individual_id, protein_name, individual_group_name):
    """Create a plot showing all cell shapes for a specific individual"""
    fig, ax = plt.subplots(figsize=(12, 8.6))  # Increased width to accommodate colorbar

    results_df.index = results_df.protein
    valid_cell_ids = results_df.loc[protein_name, [col for col in results_df.columns if individual_id in col]]
    intensity = valid_cell_ids.values
    valid_cell_ids = valid_cell_ids.index.str.extract(r'_(\d+)$')[0]
    valid_cell_ids = valid_cell_ids.astype(int).values

    # Normalize intensity between 0 and 1
    intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity)) if np.max(intensity) > np.min(intensity) else intensity

    try:
        shapes_idx = load_pickled_shapes(individual_id)

        if shapes_idx is None:
            ax.text(0.5, 0.5, f'No shapes pickle found for {individual_id}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'Cell Shapes - {individual_id} ({individual_group_name})', fontweight='bold')
            return fig

        if not shapes_idx:
            ax.text(0.5, 0.5, f'No shapes in pickle for {individual_id}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'Cell Shapes - {individual_id} ({individual_group_name})', fontweight='bold')
            return fig

        # Create a colormap
        cmap = plt.cm.inferno

        # Optional background image: single expected filename next to the shapes
        try:
            bg_dir = project_path("dashboard", "statics", "Images")
            p = os.path.join(bg_dir, f"{individual_id}_crop.png")
            if os.path.exists(p):
                bg = plt.imread(p)
                h_img, w_img = bg.shape[0], bg.shape[1]
                ax.imshow(bg, extent=(0, w_img, 0, h_img), origin='lower', zorder=0)
        except FilenotFoundError:
            pass

        # Map numeric cell_id (1..N) to intensity for quick lookup

        # Map numeric cell_id (1..N) to intensity for quick lookup
        cell_intensity_map = dict(zip(valid_cell_ids, intensity))

        shown_cells = 0
        intensity_values_shown = []  # Track intensity values of shown cells for colorbar

        # Draw all shapes: non-measured in light gray, measured in color (on top)
        for cell_id, pts in shapes_idx.items():
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            # Close polygon if needed
            if x_coords.size >= 3:
                if x_coords[0] != x_coords[-1] or y_coords[0] != y_coords[-1]:
                    x_coords = np.append(x_coords, x_coords[0])
                    y_coords = np.append(y_coords, y_coords[0])

            if cell_id in cell_intensity_map and not np.isnan(cell_intensity_map.get(cell_id, np.nan)):
                # Measured cell: colored by normalized intensity
                cell_intensity = float(cell_intensity_map[cell_id])
                color = cmap(cell_intensity)
                ax.plot(x_coords, y_coords, color="lightgray", linewidth=3, alpha=1, zorder=2)
                ax.fill(x_coords, y_coords, color=color, alpha=1, zorder=2)
                shown_cells += 1
                intensity_values_shown.append(cell_intensity)
            else:
                # Not measured for the selected protein: draw as light gray polygon
                ax.plot(x_coords, y_coords, color='lightgray', linewidth=3.0, alpha=1, zorder=1)
                ax.fill(x_coords, y_coords, color='lightgray', alpha=0.7, zorder=1)

        # Add colorbar
        if shown_cells > 0:
            # Create a ScalarMappable for the colorbar
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            
            norm = Normalize(vmin=0, vmax=1)  # Since we normalized intensity to 0-1
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            
            # Add colorbar to the right of the plot
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
            # Keep label compact and add more padding to avoid overlap with tick labels
            cbar.set_label('Normalized Intensity', fontsize=12, rotation=270, labelpad=30, va='center')
            # Slightly reduce tick label size for extra clearance
            cbar.ax.tick_params(labelsize=10)
            
            # Add some statistics to the plot
            min_intensity = np.min(intensity_values_shown)
            max_intensity = np.max(intensity_values_shown)
            mean_intensity = np.mean(intensity_values_shown)
            
            # Removed stats textbox on the right plot for aesthetics

        # Customize plot
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('', fontsize=12)
        ax.set_title(f'Cell Shapes - {individual_id} ({individual_group_name}) | Protein: {protein_name}', 
                fontsize=14, fontweight='bold')

        # Remove all ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(labelsize=0)

        # Invert y-axis to match typical microscopy coordinates (crop-local coordinates preserve this orientation)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting shapes for {individual_id}: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Cell Shapes - {individual_id} ({individual_group_name})', fontweight='bold')
    
    return fig

def main():
    # Use default Streamlit sidebar width
    # Load data
    with st.spinner("Loading data..."):
        intensity, scores, controls_results, disease_results = load_data()
    
    if intensity is None:
        st.error("Failed to load data. Please check file paths.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Title Page", "Page 1: Volcano Plot & Protein Expression", "Page 2: Individual Trajectory Analysis"]
    )
    
    if page == "Title Page":
        st.title("Data Dashboard accompanying the manuscript:")
        st.markdown("---")
        # Title Page with paper information
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; font-size: 2.5rem; margin-bottom: 1rem;">
                Single cell spatial proteomics maps human liver zonation patterns and their vulnerability to disruption in tissue architecture
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Abstract section
        st.header("Abstract")
        st.markdown("""
        <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #1f77b4;">
        Understanding protein distribution patterns across tissue architecture is crucial for deciphering organ function in health and disease. Here, we applied single-cell Deep Visual Proteomics to perform spatially-resolved proteome analysis of individual cells in native tissue. We built a robust framework comprising  strategic cell selection and  continuous protein gradient mapping allowing the investigation of larger clinical cohorts. We generated a comprehensive spatial map of the human hepatic proteome by analyzing hundreds of isolated hepatocytes from 18 individuals. Among the 2,500 proteins identified per cell about half exhibited zonated expression patterns. Cross-species comparison with male mice revealed conserved metabolic functions and human-specific features of liver zonation. Analysis of samples with disrupted liver architecture demonstrated widespread loss of protein zonation, with pericentral proteins being particularly susceptible. Our study provides a comprehensive resource of human liver organization while establishing a broadly applicable framework for spatial proteomics analyses along tissue gradients.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Authors and affiliations
        st.markdown(
            """
            <div style="font-size:1.0rem; line-height:1.6; padding: 0.5rem 0;">
                <p style="margin-bottom: 0.6rem;"><strong>Authors:</strong> Caroline A. M. Weiss<sup>1</sup>, Lauryn A. Brown<sup>2</sup>, Lucas Miranda<sup>3</sup>, Paolo Pellizzoni<sup>3</sup>, Shani Ben-Moshe<sup>1</sup>, Sophia Steigerwald<sup>1</sup>, Kirsten Remmert<sup>4</sup>, Jonathan Hernandez<sup>4</sup>, Karsten Borgwardt<sup>3</sup>, Florian A. Rosenberger<sup>1,5</sup>, Natalie Porat-Shliom<sup>2,5</sup>, Matthias Mann<sup>1,5</sup></p>
                <ol style="margin-top: 0.2rem;">
                    <li>Proteomics and Signal Transduction, Max Planck Institute of Biochemistry, Martinsried, Germany.</li>
                    <li>Cell Biology and Imaging Sections, Thoracic and GI Malignancies Branch, Center for Cancer Research, National Cancer Institute, National Institutes of Health, Bethesda, MD, USA.</li>
                    <li>Machine Learning and Systems Biology, Max Planck Institute of Biochemistry, Martinsried, Germany.</li>
                    <li>Surgical Oncology Program, National Cancer Institute (NCI), National Institutes of Health (NIH), Bethesda, Maryland, USA.</li>
                    <li>Co-corresponding authors: <a href="mailto:rosenberger@biochem.mpg.de">rosenberger@biochem.mpg.de</a>, <a href="mailto:natalie.porat-shliom@nih.gov">natalie.porat-shliom@nih.gov</a>, <a href="mailto:mmann@biochem.mpg.de">mmann@biochem.mpg.de</a></li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Preprint link
        st.markdown(
            """
            <div style="text-align: center; margin: 0.5rem 0 1.0rem 0;">
                <a href="https://www.biorxiv.org/content/10.1101/2025.04.13.648568v1.full.pdf" target="_blank" style="font-size:1.05rem; font-weight:600; text-decoration:none;">📄 View preprint on bioRxiv (PDF)</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    elif page == "Page 1: Volcano Plot & Protein Expression":
        st.title("🧬 DVP Data Dashboard")
        st.markdown("---")
        st.header("Page 1: Volcano Plot & Protein Expression Analysis")
        
        # Group selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Individual Group Selection")
        group_options = {
            "Healthy Tissue": 1,
            "Disrupted Tissue": 2
        }
        
        selected_group_name = st.sidebar.selectbox(
            "Select individual group:",
            list(group_options.keys()),
            index=0,
            help="Choose between healthy or disrupted tissue individuals"
        )
        
        selected_group = group_options[selected_group_name]
        group_name = selected_group_name.split(" (")[0]  # Extract just the name part
        
        # Select appropriate volcano results
        if selected_group == 1:
            volcano_results = controls_results
        else:
            volcano_results = disease_results
        
        # Filter data for selected group
        filtered_intensity, filtered_scores = filter_data_by_group(intensity, scores, selected_group)
        
        # Protein selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Protein Selection")
        
        # Get all available proteins from the filtered intensity data
        all_available_proteins = set(filtered_intensity['protein'].unique())
        
        # Rank proteins by their zonation significance (q-value and effect size)
        # First, get proteins that have volcano results
        proteins_with_results = volcano_results[volcano_results['index'].isin(all_available_proteins)].copy()
        
        # Sort by q-value (ascending) then by absolute coefficient (descending)
        proteins_with_results['abs_coefficient'] = proteins_with_results['coefficient'].abs()
        proteins_with_results_sorted = proteins_with_results.sort_values(
            ['q_value', 'abs_coefficient'], 
            ascending=[True, False]
        )
        
        # Get proteins without results (not in volcano plot)
        proteins_without_results = sorted(list(all_available_proteins - set(proteins_with_results['index'])))
        
        # Create final ranked list: significant proteins first, then others alphabetically
        ranked_proteins = proteins_with_results_sorted['index'].tolist() + proteins_without_results
        
        # Set ASS1 as default if available, otherwise use the most zonated protein
        if "ASS1" in ranked_proteins:
            default_protein = "ASS1"
        else:
            default_protein = ranked_proteins[0] if ranked_proteins else "No proteins available"
        
        default_index = ranked_proteins.index(default_protein) if default_protein in ranked_proteins else 0
        
        # Searchable selectbox for protein (now ranked by zonation)
        selected_protein = st.sidebar.selectbox(
            "Select a protein:",
            ranked_proteins,
            index=default_index,
            help="Proteins ranked by zonation significance (most zonated first). Type to search for a specific protein."
        )
        
        # Page description in a blue-accent box (like the landing-page abstract)
        st.markdown(
            """
            <div style="text-align: justify; font-size: 1.05rem; line-height: 1.6; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #1f77b4;">
            This page allows exploration of protein zonation patterns in the human liver across two groups: Healthy Individuals (healthy liver architecture) and Individuals with disrupted tissue architecture.
            <br><br>
            The two main visualizations are:
            <br><br>
            <p><strong>Volcano plot</strong>: Each dot represents a protein. The x-axis shows the zonation coefficient (effect size), where negative values indicate enrichment toward the central vein and positive values indicate enrichment toward the portal vein. The y-axis shows statistical evidence as −log10(q). Points are deemed significant only when q < 0.05 and |zonation coefficient| > 1. Vertical dotted lines mark ±1 to denote the effect-size threshold, and the currently selected protein is highlighted with a golden star.</p>
            <p><strong>Protein expression</strong>: For the selected protein and cohort, each dot is a cell with its z-scored protein intensity plotted against the spatial score (Central → Portal). A LOWESS smoothing curve is overlaid to summarize the trend across the gradient while keeping the point cloud unbiased.</p>
            
            Use the sidebar to select the individual group and protein of interest. Proteins are ranked by their zonation significance, with the most zonated proteins listed first. ASS1 is pre-selected as it is a well-known zonated protein.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Volcano Plot - {group_name}")
            volcano_fig = create_volcano_plot(volcano_results, f"Volcano Plot - {group_name}", selected_protein)
            st.pyplot(volcano_fig)
            plt.close()
            
            # Display some statistics
            st.markdown("**Volcano Plot Statistics:**")
            total_proteins = len(volcano_results)
            significant_proteins = ((volcano_results['q_value'] < 0.05) & (np.abs(volcano_results['coefficient']) > 1)).sum()
            st.write(f"- Significant proteins (q < 0.05 AND |zonation coef| > 1): {significant_proteins}")
            st.write(f"- Percentage significant: {significant_proteins/total_proteins*100:.1f}%")
        
        with col2:
            st.subheader(f"Protein Expression - {group_name}")
            expression_fig = create_protein_expression_plot(
                filtered_intensity, filtered_scores, selected_protein, group_name
            )
            st.pyplot(expression_fig)
            plt.close()
            
            # Display protein information
            st.markdown(f"**Selected Protein: {selected_protein}**")
            if selected_protein in volcano_results['index'].values:
                protein_info = volcano_results[volcano_results['index'] == selected_protein].iloc[0]
                st.write(f"- Zonation coefficient: {protein_info['coefficient']:.4f}")
                st.write(f"- p-value: {protein_info['p_value']:.2e}")
                st.write(f"- q-value (FDR adjusted): {protein_info['q_value']:.2e}")
            else:
                st.write("- No statistical results available for this protein")
        
        # Additional information
        st.markdown("---")
        st.subheader("Dataset Information")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Proteins (union over cells)", len(all_available_proteins))

        with col2:
            st.metric("Median proteins per cell", 2539)
            
        with col3:
            st.metric("Tested proteins", 1741)
        
        with col4:
            st.metric("Total Cells for selected protein", len(filtered_intensity.columns) - 1)  # Exclude protein column
        
        with col5:
            st.metric("Total Individuals", 18)
    
    elif page == "Page 2: Individual Trajectory Analysis":
        st.title("🧬 DVP Data Dashboard")
        st.markdown("---")
        st.header("Page 2: Individual Trajectory Analysis")
        
        # Use explicit ordering provided by the user
        all_individuals = ORDERED_INDIVIDUAL_IDS
            
        # Sidebar for individual and protein selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Individual Selection")

        # Build display labels and keep the explicit order
        id_to_label = build_individual_label_mapping()
        ordered_individuals = all_individuals

        # Default to NML16 (Individual 9) if available, else first in ordered list
        default_individual = "NML16" if "NML16" in ordered_individuals else (ordered_individuals[0] if ordered_individuals else None)
        default_individual_index = ordered_individuals.index(default_individual) if default_individual else 0

        selected_individual = st.sidebar.selectbox(
            "Select an individual:",
            ordered_individuals,
            index=default_individual_index,
            format_func=lambda _id: id_to_label.get(_id, _id),
            help="Choose a specific individual to analyze their protein trajectory"
        )
        
        # Determine which group the selected individual belongs to
        individual_group = 1 if selected_individual in GROUP_DICT[1] else 2
        individual_group_name = "Healthy" if individual_group == 1 else "Disrupted Tissue Architecture"
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Protein Selection")
        
        # Get all available proteins from the intensity data
        all_available_proteins = set(intensity['protein'].unique())

        # Choose volcano results matching the individual's group
        volcano_results = controls_results if individual_group == 1 else disease_results

        # Rank proteins: first by significance (q asc), then by |coef| desc; then others alphabetically
        proteins_with_results = volcano_results[volcano_results['index'].isin(all_available_proteins)].copy()
        proteins_with_results['abs_coefficient'] = proteins_with_results['coefficient'].abs()
        proteins_with_results_sorted = proteins_with_results.sort_values(
            ['q_value', 'abs_coefficient'], ascending=[True, False]
        )
        proteins_without_results = sorted(list(all_available_proteins - set(proteins_with_results['index'])))
        ranked_proteins = proteins_with_results_sorted['index'].tolist() + proteins_without_results

        # Keep ASS1 as default if available
        if "ASS1" in ranked_proteins:
            default_protein = "ASS1"
        else:
            default_protein = ranked_proteins[0] if ranked_proteins else "No proteins available"
        default_index = ranked_proteins.index(default_protein) if default_protein in ranked_proteins else 0

        selected_protein = st.sidebar.selectbox(
            "Select a protein:",
            ranked_proteins,
            index=default_index,
            help="Proteins ranked by zonation significance (most zonated first). Type to search for a specific protein."
        )
        
        # Page description in a blue-accent box (like the landing-page abstract)
        st.markdown(
            """
            <div style="text-align: justify; font-size: 1.05rem; line-height: 1.6; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #1f77b4;">
            This page enables detailed exploration of protein expression trajectories within individual liver samples. Select an individual and protein of interest to visualize how that protein's abundance varies across spatial zones (Central → Portal) in that specific person.
            <br><br>
            The two main visualizations are:
            <br><br>
            <p><strong>Individual trajectory</strong>: For the selected individual, each dot is a sampled cell showing the z-scored intensity of the chosen protein versus its spatial score (Central → Portal). This plot highlights person-specific zonation patterns.</p>
            <p><strong>Cell shapes</strong>: Cell polygons colored by protein expression intensity are shown over the corresponding tissue image. Cells where the selected protein was not detected are still shown in grey.</p>
            Use the sidebar to select the individual and protein of interest.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)
        
        display_name = build_individual_label_mapping().get(selected_individual, selected_individual)

        with col1:
            st.subheader(f"Individual Trajectory - {display_name}")
            trajectory_fig = create_individual_trajectory_plot(
                intensity, scores, selected_protein, selected_individual, individual_group_name
            )
            # Override title in the figure to use display label
            try:
                trajectory_fig.axes[0].set_title(
                    f"{selected_protein} Trajectory - {display_name}",
                    fontsize=14, fontweight='bold'
                )
            except Exception:
                pass
            st.pyplot(trajectory_fig)
            plt.close()
            
            # Count samples for this individual
            individual_samples = [col for col in intensity.columns if col.startswith(selected_individual + "_")]
        
        with col2:
            st.subheader(f"Cell Shapes - {display_name}")
            shapes_fig = create_cell_shapes_plot(intensity, selected_individual, selected_protein, individual_group_name)
            # Override title in the figure to use display label
            try:
                shapes_fig.axes[0].set_title(
                    f"Cell Shapes - {display_name} | Protein: {selected_protein}",
                    fontsize=14, fontweight='bold'
                )
            except Exception:
                pass
            st.pyplot(shapes_fig)
            plt.close()
            
            # Shapes are loaded directly from pickles within the plotting function
        
        # Additional information
        st.markdown("---")
        st.subheader("Dataset Information")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Proteins (union over cells)", len(all_available_proteins))

        with col2:
            st.metric("Median proteins per cell", 2539)
            
        with col3:
            st.metric("Tested proteins", 1741)
        
        with col4:
            st.metric("Total Individuals", 18)

if __name__ == "__main__":
    main()
