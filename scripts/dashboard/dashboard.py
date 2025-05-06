import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mlflow
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import lightgbm as lgb
from datetime import datetime

print(f"[{datetime.now()}] Launching Credit Default Prediction Dashboard...")

# Page configuration
st.set_page_config(
    page_title="Credit Default Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove plotly modebar 
st.markdown("""
<style>
    .stPlotlyChart .modebar {display: none !important}
</style>
""", unsafe_allow_html=True)

# Paths
METRICS_PATH = Path("data/metrics/summary.parquet")
MODEL_DIR = Path("data/model")
METRICS_DIR = Path("data/metrics")
MLFLOW_TRACKING_URI = "data/mlruns"

# Connect to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Helper functions
def load_metrics_data():
    """Load metrics summary data"""
    if not METRICS_PATH.exists():
        st.error(f"Metrics file not found: {METRICS_PATH}")
        return None
    
    try:
        return pd.read_parquet(METRICS_PATH)
    except Exception as e:
        st.error(f"Error loading metrics data: {e}")
        return None

def compute_ranks(df, metrics):
    """Compute ranks for each metric (1 = best)"""
    ranks_df = pd.DataFrame()
    
    for metric, _ in metrics:
        if metric in df.columns:
            # Higher is better, except brier_score and latency_or_model_size
            ascending = metric in ['brier_score', 'latency_or_model_size']
            ranks_df[f"{metric}_rank"] = df[metric].rank(ascending=ascending)
    
    return ranks_df

def compute_weighted_score(metrics_ranks, metrics_weights):
    """Compute weighted score based on ranks and weights"""
    weight_dict = dict(metrics_weights)
    # Make sure rank columns appear in same order
    rank_cols = [f"{m}_rank" for m, _ in metrics_weights if f"{m}_rank" in metrics_ranks.columns]
    
    return metrics_ranks[rank_cols].mul(
        [weight_dict[m.split('_rank')[0]] for m in rank_cols],
        axis=1
    ).sum(axis=1)

def style_leaderboard(df):
    """Apply conditional formatting to the leaderboard"""
    # Define a function for styling numeric columns
    def highlight_cols(s):
        # For metric values
        if s.name in [m for m, _ in metrics_weights]:
            # For most metrics higher is better
            is_higher_better = s.name not in ['brier_score', 'latency_or_model_size']
            
            # Calculate normalized values (0-1 range)
            if s.max() != s.min():  # Avoid division by zero
                normalized = (s - s.min()) / (s.max() - s.min())
                if not is_higher_better:
                    normalized = 1 - normalized
            else:
                normalized = [0.5] * len(s)  # Default to middle value if all values are the same
            
            # Create color gradient from red to yellow to green
            def get_color(val):
                # Red to yellow to green gradient
                if val < 0.5:
                    # Red to yellow (0-0.5)
                    r = 255
                    g = int(255 * (val * 2))
                    b = 0
                else:
                    # Yellow to green (0.5-1)
                    r = int(255 * (1 - (val - 0.5) * 2))
                    g = 255
                    b = 0
                return f'background-color: rgba({r}, {g}, {b}, 0.3)'
            
            return [get_color(v) for v in normalized]
        
        return [''] * len(s)
    
    # Apply the style
    return df.style.apply(highlight_cols)

def plot_roc_curve(fpr, tpr, auc_value):
    """Plot ROC curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_value:.3f})',
        line=dict(color='royalblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.01, y=0.99),
        width=600,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_pr_curve(precision, recall, auc_pr):
    """Plot Precision-Recall curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {auc_pr:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(x=0.01, y=0.01),
        width=600,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_learning_curve(evals_result):
    """Plot learning curve from evals_result"""
    if not evals_result or 'valid_0' not in evals_result:
        return None
    
    metrics = list(evals_result['valid_0'].keys())
    if not metrics:
        return None
    
    fig = make_subplots(rows=len(metrics), cols=1, 
                        subplot_titles=[m.upper() for m in metrics],
                        vertical_spacing=0.1)
    
    for i, metric in enumerate(metrics):
        if 'valid_0' in evals_result and metric in evals_result['valid_0']:
            valid_values = evals_result['valid_0'][metric]
            fig.add_trace(
                go.Scatter(
                    y=valid_values,
                    mode='lines',
                    name=f'Validation {metric.upper()}',
                    line=dict(color='royalblue')
                ),
                row=i+1, col=1
            )
        
        if 'train' in evals_result and metric in evals_result['train']:
            train_values = evals_result['train'][metric]
            fig.add_trace(
                go.Scatter(
                    y=train_values,
                    mode='lines',
                    name=f'Training {metric.upper()}',
                    line=dict(color='darkorange')
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=300 * len(metrics),
        title='Learning Curves',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    for i in range(len(metrics)):
        fig.update_xaxes(title_text='Iterations' if i == len(metrics)-1 else '', row=i+1, col=1)
    
    return fig

def plot_confusion_matrix(cm, class_names=['Non-Default', 'Default']):
    """Plot confusion matrix"""
    z = cm
    x = class_names
    y = class_names
    
    # Calculate derived values
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    misclass = 1 - accuracy
    
    # Create annotations
    annotations = []
    for i, row in enumerate(z):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=str(value),
                    font=dict(color='white' if value > total/4 else 'black'),
                    showarrow=False
                )
            )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        showscale=False
    ))
    
    # Add annotations
    fig.update_layout(
        title=f'Confusion Matrix (Accuracy: {accuracy:.3f})',
        annotations=annotations,
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label'),
        width=500,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def get_model_artifacts(model_name):
    """Get model artifacts from files or MLflow"""
    fpr = tpr = precision = recall = evals_result = None
    model_base_name = model_name.replace('.txt', '')
    metrics_dir = METRICS_DIR
    
    # Try to get ROC curve data from files
    try:
        roc_file_path = metrics_dir / f"{model_base_name}_roc.npz"
        if roc_file_path.exists():
            roc_data = np.load(roc_file_path)
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
        else:
            # Fallback to individual files
            fpr_path = metrics_dir / f"{model_base_name}_fpr.npy"
            tpr_path = metrics_dir / f"{model_base_name}_tpr.npy"
            if fpr_path.exists() and tpr_path.exists():
                fpr = np.load(fpr_path)
                tpr = np.load(tpr_path)
    except Exception as e:
        st.error(f"Could not load ROC data from files: {e}")
        return None, None, None, None, None
    
    # Try to get PR curve data from files
    try:
        pr_file_path = metrics_dir / f"{model_base_name}_pr.npz"
        if pr_file_path.exists():
            pr_data = np.load(pr_file_path)
            precision = pr_data['precision']
            recall = pr_data['recall']
        else:
            # Fallback to individual files
            precision_path = metrics_dir / f"{model_base_name}_precision.npy"
            recall_path = metrics_dir / f"{model_base_name}_recall.npy"
            if precision_path.exists() and recall_path.exists():
                precision = np.load(precision_path)
                recall = np.load(recall_path)
    except Exception as e:
        st.error(f"Could not load PR data from files: {e}")
    
    # If files don't exist, try MLflow
    if fpr is None or tpr is None or precision is None or recall is None:
        try:
            # Try to find the run with the model name
            runs = mlflow.search_runs(filter_string=f"tag.run_name = '{model_base_name}'")
            
            if not runs.empty:
                run_id = runs.iloc[0]['run_id']
                client = mlflow.tracking.MlflowClient()
                
                # Try to get ROC curve data
                if fpr is None or tpr is None:
                    try:
                        fpr_path = client.download_artifacts(run_id, "roc/fpr.npy")
                        tpr_path = client.download_artifacts(run_id, "roc/tpr.npy")
                        fpr = np.load(fpr_path)
                        tpr = np.load(tpr_path)
                    except Exception as e:
                        st.debug(f"Could not load ROC data from MLflow: {e}")
                
                # Try to get PR curve data
                if precision is None or recall is None:
                    try:
                        precision_path = client.download_artifacts(run_id, "pr/precision.npy")
                        recall_path = client.download_artifacts(run_id, "pr/recall.npy")
                        precision = np.load(precision_path)
                        recall = np.load(recall_path)
                    except Exception as e:
                        st.debug(f"Could not load PR data from MLflow: {e}")
                
                # Try to get evals_result
                try:
                    evals_result_path = client.download_artifacts(run_id, "evals_result.json")
                    with open(evals_result_path, 'r') as f:
                        evals_result = json.load(f)
                except Exception as e:
                    st.debug(f"Could not load evals_result from MLflow: {e}")
        except Exception as e:
            st.debug(f"Error accessing MLflow: {e}")
    
    return fpr, tpr, precision, recall, evals_result

# Define metrics weights
metrics_weights = [
    ("auc_roc", 0.35),
    ("ks", 0.25),
    ("auc_pr", 0.20),
    ("f1_score", 0.10),
    ("brier_score", 0.05),
    ("latency_or_model_size", 0.05)
]

# Main dashboard
st.title('Credit Default Prediction Dashboard')

# Load metrics data
metrics_df = load_metrics_data()

if metrics_df is not None:
    # Add any missing metrics columns with NaN values
    for metric, _ in metrics_weights:
        if metric not in metrics_df.columns:
            metrics_df[metric] = np.nan
    
    # Compute ranks
    ranks_df = compute_ranks(metrics_df, metrics_weights)
    
    # Create leaderboard dataframe
    leaderboard_df = pd.concat([metrics_df, ranks_df], axis=1)
    
    # Compute weighted score
    leaderboard_df['weighted_score'] = compute_weighted_score(ranks_df, metrics_weights)
    
    # Add overall rank and sort
    leaderboard_df['overall_rank'] = leaderboard_df['weighted_score'].rank()
    leaderboard_df = leaderboard_df.sort_values('overall_rank')
    
    # Display metrics and ranks
    st.header('Model Leaderboard')
    
    # Create a simplified ranking dataframe with only model names and ranks
    ranking_df = pd.DataFrame({
        'Rank': range(1, min(len(leaderboard_df), 6) + 1),
        'Model': leaderboard_df['model'].values[:6]
    })
    
    # Display the simplified ranking
    st.subheader('Model Ranking')
    st.dataframe(ranking_df, hide_index=True)
    
    # Add a toggle for detailed metrics
    if st.checkbox('Show detailed metrics'):
        # Select columns for detailed view
        detailed_cols = ['model']
        for metric, _ in metrics_weights:
            if metric in metrics_df.columns:
                detailed_cols.append(metric)
        
        # Display the detailed metrics
        st.subheader('Detailed Metrics')
        st.dataframe(style_leaderboard(leaderboard_df[detailed_cols]), hide_index=True)
    
    # Create tabs for each model with formatted names
    st.header('Model Details')
    
    # Format model names for tabs
    def format_model_name(model_name):
        # Extract model type (gbdt, dart, goss) and whether it's baseline or tuned
        parts = model_name.replace('.txt', '').split('_')
        model_type = parts[0].capitalize()
        is_tuned = parts[1] if len(parts) > 1 else ''
        return f"{model_type} {'tuned' if is_tuned == 'tuned' else 'baseline'}"
    
    formatted_names = [format_model_name(model) for model in leaderboard_df['model'].tolist()]
    tabs = st.tabs(formatted_names)
    
    for i, tab in enumerate(tabs):
        with tab:
            model_name = leaderboard_df.iloc[i]['model']
            
            # Display metrics in columns
            cols = st.columns(3)
            metric_idx = 0
            
            for metric, _ in metrics_weights:
                if metric in leaderboard_df.columns and not pd.isna(leaderboard_df.iloc[i][metric]):
                    with cols[metric_idx % 3]:
                        st.metric(
                            label=metric.upper(),
                            value=f"{leaderboard_df.iloc[i][metric]:.4f}",
                            delta=f"Rank: {int(leaderboard_df.iloc[i][f'{metric}_rank'])}"
                        )
                        metric_idx += 1
            
            # Get confusion matrix data if available
            cm_data = None
            if all(col in leaderboard_df.columns for col in ['tp', 'fp', 'tn', 'fn']):
                cm_data = np.array([
                    [leaderboard_df.iloc[i]['tn'], leaderboard_df.iloc[i]['fp']],
                    [leaderboard_df.iloc[i]['fn'], leaderboard_df.iloc[i]['tp']]
                ])
            
            # Get model artifacts from MLflow
            fpr, tpr, precision, recall, evals_result = get_model_artifacts(model_name)
            
            # Create two columns for plots
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC curve
                if fpr is not None and tpr is not None:
                    auc_value = leaderboard_df.iloc[i]['auc_roc']
                    st.plotly_chart(plot_roc_curve(fpr, tpr, auc_value))
                else:
                    st.warning("ROC curve data not available")
                
                # Confusion matrix
                if cm_data is not None:
                    st.plotly_chart(plot_confusion_matrix(cm_data))
                else:
                    st.warning("Confusion matrix data not available")
            
            with col2:
                # PR curve
                if precision is not None and recall is not None:
                    auc_pr = leaderboard_df.iloc[i]['auc_pr']
                    st.plotly_chart(plot_pr_curve(precision, recall, auc_pr))
                else:
                    st.warning("Precision-Recall curve data not available")
                
                # Learning curve
                if evals_result is not None:
                    learning_curve = plot_learning_curve(evals_result)
                    if learning_curve:
                        st.plotly_chart(learning_curve)
                else:
                    st.warning("Learning curve data not available")
            
            # Model details
            with st.expander("Model Details"):
                # Try to load model and show feature importance
                model_path = MODEL_DIR / model_name
                if model_path.exists():
                    try:
                        booster = lgb.Booster(model_file=str(model_path))
                        importance = pd.DataFrame({
                            'Feature': booster.feature_name(),
                            'Importance': booster.feature_importance()
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig = px.bar(
                            importance.head(20), 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title='Feature Importance (top 20)',
                            hover_data=['Feature', 'Importance'],
                            labels={'Feature': '', 'Importance': 'Importance Score'}
                        )
                        
                        # Improve layout for better readability
                        fig.update_layout(
                            height=600,  # Make the graph taller
                            margin=dict(l=10, r=10, t=40, b=10),  # Adjust margins
                            yaxis={
                                'categoryorder': 'total ascending',
                                'showticklabels': False  # Hide y-axis labels
                            },
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=12,
                                font_family="Arial",
                                font_color="black"  # Set hover text color to black for better readability
                            )
                        )
                        
                        # Add a note about hovering
                        st.info("Hover over the bars to see feature names and importance values")
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show model parameters in a more compact format
                        st.subheader("Model Parameters")
                        params = booster.params
                        
                        # Convert params to a more readable format
                        formatted_params = {}
                        for key, value in params.items():
                            try:
                                # Try to convert string numbers to actual numbers
                                if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                    formatted_params[key] = float(value) if '.' in value else int(value)
                                else:
                                    formatted_params[key] = value
                            except:
                                formatted_params[key] = value
                        
                        # Display with indent=1 for more compact view
                        st.json(formatted_params, expanded=False)
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                else:
                    st.warning(f"Model file not found: {model_path}")
else:
    st.error("No metrics data available. Please run the evaluation script first.")
