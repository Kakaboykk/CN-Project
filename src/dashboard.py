"""
Interactive Streamlit Dashboard for Cerebrum-IDS
Provides visualization and quick testing capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys
import json
import requests
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyvis.network import Network
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ag_utils import Corpus, parse_ag_file, parse_node_properties
from models import NN, GCN, GAT, GCN_EW
from model_utils import evaluate_performance, predict_prob
from ensemble_model import EnsembleModel, evaluate_ensemble, predict_ensemble_prob
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

# Page configuration
st.set_page_config(
    page_title="Cerebrum-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rounded style and theme support
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: var(--background-color);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stSelectbox, .stTextInput, .stTextArea {
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
    }
    h1, h2, h3 {
        color: var(--text-color);
    }
    .info-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_data_and_models():
    """Load attack graph, dataset, and initialize models"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load attack graph
    attack_graph_path = os.path.join(project_root, 'mulval_attack_graph', 'AttackGraph.dot')
    nodes, edges, node_properties = parse_ag_file(attack_graph_path)
    node_dict = parse_node_properties(nodes, node_properties)
    corpus = Corpus(node_dict)
    node_features = corpus.get_node_features()
    action_nodes = corpus.get_action_nodes()
    action_node_idx = list(action_nodes.keys())
    action_mask = action_node_idx
    
    # Build edge index
    adj_matrix = torch.zeros(len(nodes), len(nodes))
    for edge in edges:
        source_node, target_node = edge
        source_index = nodes.index(source_node)
        target_index = nodes.index(target_node)
        adj_matrix[source_index][target_index] = 1
    edge_index = adj_matrix.nonzero().t().contiguous()
    
    # Load dataset
    data_path = os.path.join(project_root, 'datasets', 'synt', '')
    sample_method = 'synthetic'
    X_test = torch.load(data_path + f'X_test-{sample_method}.pth')
    Y_test = torch.load(data_path + f'Y_test-{sample_method}.pth')
    
    # Initialize models
    in_dim = X_test.shape[-1]
    rt_meas_dim = 78
    hidden_dim = 20
    out_dim = 1
    
    models = {}
    models['NN'] = NN(rt_meas_dim, hidden_dim, out_dim)
    models['GCN'] = GCN(in_dim, hidden_dim, out_dim)
    models['GCN-EW'] = GCN_EW(in_dim, hidden_dim, out_dim, edge_index)
    models['GAT'] = GAT(hidden_channels=hidden_dim, heads=4, in_dim=in_dim, out_dim=out_dim)
    
    # Set model properties
    for name, model in models.items():
        model.name = name
        model.action_mask = action_mask

    # Attempt to load trained weights/statistics
    artifacts_dir = Path(project_root) / 'artifacts'
    models_trained = False
    alpha = 0.6

    if artifacts_dir.exists():
        state_files = {
            'NN': artifacts_dir / 'nn.pt',
            'GCN': artifacts_dir / 'gcn.pt',
            'GCN-EW': artifacts_dir / 'gcn_ew.pt',
            'GAT': artifacts_dir / 'gat.pt'
        }

        loaded_models = []
        for name, path in state_files.items():
            if path.exists():
                try:
                    models[name].load_state_dict(torch.load(path, map_location='cpu'))
                    loaded_models.append(name)
                except Exception as exc:
                    st.warning(f"⚠️ Failed to load weights for {name}: {exc}")

        models_trained = len(loaded_models) == len(state_files)

        # Load training statistics if available
        stats_path = artifacts_dir / 'training_stats.json'
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    stats_data = json.load(f)
                for name, model in models.items():
                    stat_entry = stats_data.get(name, {})
                    model.stat = {
                        'loss_train': stat_entry.get('loss_train', []),
                        'loss_val': stat_entry.get('loss_val', []),
                        'acc_train': stat_entry.get('acc_train', []),
                        'acc_val': stat_entry.get('acc_val', [])
                    }
            except Exception as exc:
                st.warning(f"⚠️ Failed to load training statistics: {exc}")

        ensemble_meta = artifacts_dir / 'ensemble_meta.pt'
        if ensemble_meta.exists():
            try:
                alpha_info = torch.load(ensemble_meta, map_location='cpu')
                alpha = float(alpha_info.get('alpha', alpha))
            except Exception as exc:
                st.warning(f"⚠️ Failed to load ensemble metadata: {exc}")

    # Create ensemble model (even if weights missing, will use default alpha)
    ensemble_model = EnsembleModel(
        gcn_model=models['GCN'],
        gat_model=models['GAT'],
        alpha=alpha,
        learnable_alpha=False
    )
    ensemble_model.name = 'Ensemble'
    ensemble_model.action_mask = action_mask

    # Ensure models have stat attribute to avoid attribute errors
    for model in models.values():
        if not hasattr(model, 'stat'):
            model.stat = {
                'loss_train': [],
                'loss_val': [],
                'acc_train': [],
                'acc_val': []
            }
    
    return {
        'nodes': nodes,
        'edges': edges,
        'node_dict': node_dict,
        'corpus': corpus,
        'edge_index': edge_index,
        'action_mask': action_mask,
        'X_test': X_test,
        'Y_test': Y_test,
        'models': models,
        'rt_meas_dim': rt_meas_dim,
        'ensemble_model': ensemble_model,
        'models_trained': models_trained
    }

def create_attack_graph_visualization(nodes, edges, node_dict):
    """Create interactive attack graph using PyVis"""
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    
    # Add nodes with different colors based on type
    for node_id, node_info in node_dict.items():
        node_label = f"{node_id}: {node_info['predicate']}"
        shape = node_info.get('shape', 'box')
        
        # Color coding
        if shape == 'diamond':
            color = '#ff6b6b'  # Red for action nodes
            size = 30
        elif shape == 'box':
            color = '#4ecdc4'  # Teal for privilege nodes
            size = 25
        else:
            color = '#95e1d3'  # Light teal for other nodes
            size = 20
        
        net.add_node(
            node_id,
            label=node_label,
            color=color,
            size=size,
            title=f"Predicate: {node_info['predicate']}<br>Attributes: {', '.join(node_info['attributes'])}"
        )
    
    # Add edges
    for edge in edges:
        source_node, target_node = edge
        source_index = nodes.index(source_node)
        target_index = nodes.index(target_node)
        net.add_edge(source_index, target_index, width=2)
    
    # Save to temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

def plot_roc_curves(models, X_test, Y_test, edge_index, rt_meas_dim, ensemble_model=None):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, (name, model) in enumerate(models.items()):
        prob = predict_prob(model, X_test, edge_index, rt_meas_dim)
        y_probs = prob.view(-1, 2)
        
        fpr, tpr, thresholds = roc_curve(Y_test.view(-1).numpy(), y_probs[:, 1].numpy())
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{name} (AUC = {roc_auc:.4f})',
            line=dict(color=colors[idx], width=3),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash'),
        hovertemplate='Random<extra></extra>'
    ))

    # Ensemble curve
    if ensemble_model is not None:
        ensemble_prob = predict_ensemble_prob(ensemble_model, X_test, edge_index, rt_meas_dim)
        ensemble_y_probs = ensemble_prob.view(-1, 2)
        fpr, tpr, _ = roc_curve(Y_test.view(-1).numpy(), ensemble_y_probs[:, 1].numpy())
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'Ensemble (AUC = {roc_auc:.4f})',
            line=dict(color='#9b59b6', width=4, dash='dot'),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='ROC Curves - All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='closest',
        template='plotly_white',
        height=500,
        legend=dict(x=0.7, y=0.2)
    )
    
    return fig

def plot_precision_recall_curves(models, X_test, Y_test, edge_index, rt_meas_dim, ensemble_model=None):
    """Plot Precision-Recall curves for all models"""
    fig = go.Figure()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, (name, model) in enumerate(models.items()):
        prob = predict_prob(model, X_test, edge_index, rt_meas_dim)
        y_probs = prob.view(-1, 2)
        
        precision, recall, _ = precision_recall_curve(
            Y_test.view(-1).numpy(), 
            y_probs[:, 1].numpy()
        )
        pr_auc = auc(recall, precision)
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{name} (AUC = {pr_auc:.4f})',
            line=dict(color=colors[idx], width=3),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Precision-Recall Curves - All Models',
        xaxis_title='Recall',
        yaxis_title='Precision',
        hovermode='closest',
        template='plotly_white',
        height=500,
        legend=dict(x=0.7, y=0.2)
    )

    if ensemble_model is not None:
        ensemble_prob = predict_ensemble_prob(ensemble_model, X_test, edge_index, rt_meas_dim)
        ensemble_y_probs = ensemble_prob.view(-1, 2)
        precision, recall, _ = precision_recall_curve(
            Y_test.view(-1).numpy(),
            ensemble_y_probs[:, 1].numpy()
        )
        pr_auc = auc(recall, precision)
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'Ensemble (AUC = {pr_auc:.4f})',
            line=dict(color='#9b59b6', width=4, dash='dot'),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))
    
    return fig

def plot_f1_scores(models, X_test, Y_test, edge_index, rt_meas_dim, ensemble_model=None):
    """Plot F1 scores for different thresholds"""
    fig = go.Figure()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    thresholds = np.linspace(0.1, 0.9, 50)
    
    for idx, (name, model) in enumerate(models.items()):
        prob = predict_prob(model, X_test, edge_index, rt_meas_dim)
        y_probs = prob.view(-1, 2)[:, 1].numpy()
        y_true = Y_test.view(-1).numpy()
        
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name=name,
            line=dict(color=colors[idx], width=3),
            hovertemplate='Threshold: %{x:.3f}<br>F1 Score: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='F1 Score vs Threshold - All Models',
        xaxis_title='Threshold',
        yaxis_title='F1 Score',
        hovermode='closest',
        template='plotly_white',
        height=500,
        legend=dict(x=0.7, y=0.2)
    )

    if ensemble_model is not None:
        thresholds = np.linspace(0.1, 0.9, 50)
        ensemble_prob = predict_ensemble_prob(ensemble_model, X_test, edge_index, rt_meas_dim)
        y_probs = ensemble_prob.view(-1, 2)[:, 1].numpy()
        y_true = Y_test.view(-1).numpy()
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name='Ensemble',
            line=dict(color='#9b59b6', width=4, dash='dot'),
            hovertemplate='Threshold: %{x:.3f}<br>F1 Score: %{y:.3f}<extra></extra>'
        ))
    
    return fig

def predict_from_json(json_data, models, edge_index, rt_meas_dim, action_mask, ensemble_model=None):
    """Make prediction from JSON input"""
    try:
        # Parse JSON input
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # Convert to tensor format
        # Expected format: {"features": [[...], [...], ...]} or similar
        if 'features' in data:
            features = torch.tensor(data['features'], dtype=torch.float32)
        else:
            # Try to infer format
            features = torch.tensor(list(data.values())[0], dtype=torch.float32)
        
        # Ensure correct shape (num_samples, num_nodes, feature_dim)
        if len(features.shape) == 2:
            # Assume it's (num_nodes, feature_dim) - add batch dimension
            features = features.unsqueeze(0)
        
        results = {}
        for name, model in models.items():
            prob = predict_prob(model, features, edge_index, rt_meas_dim)
            prob_1 = prob[:, :, 1]  # Probability of class 1 (attack)
            
            results[name] = {
                'probabilities': prob_1.tolist(),
                'predictions': (prob_1 > 0.5).int().tolist(),
                'max_prob': prob_1.max().item(),
                'mean_prob': prob_1.mean().item()
            }
        
        if ensemble_model is not None:
            ensemble_prob = predict_ensemble_prob(ensemble_model, features, edge_index, rt_meas_dim)
            prob_1 = ensemble_prob[:, :, 1]
            results['Ensemble'] = {
                'probabilities': prob_1.tolist(),
                'predictions': (prob_1 > 0.5).int().tolist(),
                'max_prob': prob_1.max().item(),
                'mean_prob': prob_1.mean().item(),
                'alpha': ensemble_model.get_alpha()
            }
        
        return results
    except Exception as e:
        return {'error': str(e)}

def predict_from_api(api_url, json_data):
    """Fetch prediction from FastAPI endpoint"""
    try:
        response = requests.post(api_url, json=json_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {'error': str(e)}

# Main app
def main():
    # Title
    st.title("🛡️ Cerebrum-IDS Dashboard")
    st.markdown("Interactive visualization and testing platform for Graph Neural Network-based Intrusion Detection")
    
    # Load data (cached)
    with st.spinner("Loading models and data..."):
        data = load_data_and_models()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
        st.markdown("---")
        st.header("📊 Quick Stats")
        st.metric("Total Nodes", len(data['nodes']))
        st.metric("Total Edges", len(data['edges']))
        st.metric("Action Nodes", len(data['action_mask']))
        st.metric("Test Samples", len(data['X_test']))
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Attack Graph", "📊 Model Metrics", "🔮 Live Inference"])
    
    # Tab 1: Attack Graph
    with tab1:
        st.header("Attack Graph Visualization")
        st.markdown("Interactive visualization of the network attack graph structure")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Graph Structure")
            with st.spinner("Generating graph visualization..."):
                graph_html = create_attack_graph_visualization(
                    data['nodes'], 
                    data['edges'], 
                    data['node_dict']
                )
                with open(graph_html, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=600)
        
        with col2:
            st.subheader("Node Information")
            selected_node = st.selectbox("Select Node", range(len(data['nodes'])))
            
            if selected_node in data['node_dict']:
                node_info = data['node_dict'][selected_node]
                st.markdown(f"""
                **Node ID:** {selected_node}
                
                **Predicate:** {node_info['predicate']}
                
                **Attributes:** {', '.join(node_info['attributes'])}
                
                **Shape:** {node_info.get('shape', 'N/A')}
                
                **Compromise Probability:** {node_info.get('possibility', 'N/A')}
                """)
            
            st.markdown("---")
            st.subheader("Graph Statistics")
            st.json({
                "Total Nodes": len(data['nodes']),
                "Total Edges": len(data['edges']),
                "Action Nodes": len(data['action_mask']),
                "Node Features": data['corpus'].get_node_features().shape[1]
            })
    
    # Tab 2: Model Metrics
    with tab2:
        st.header("Model Performance Metrics")
        
        # Check if models are trained
        models_trained = data.get('models_trained', False)
        
        if not models_trained:
            st.warning("⚠️ Models are not trained yet. Metrics shown are for untrained models. For accurate results, train the models first using `run_simple.py`")
            st.info("💡 **Note**: Training/validation metrics will show as 0.0000 for untrained models. Test set metrics are still calculated.")
        
        # Calculate metrics
        with st.spinner("Calculating metrics..."):
            metrics = evaluate_performance(
                data['models'], 
                data['X_test'], 
                data['Y_test'], 
                data['edge_index']
            )
            ensemble_model = data.get('ensemble_model')
            ensemble_metrics = None
            if ensemble_model is not None:
                ensemble_metrics = evaluate_ensemble(
                    ensemble_model,
                    data['X_test'],
                    data['Y_test'],
                    data['edge_index']
                )
                metrics.append(ensemble_metrics)
            metrics_df = pd.DataFrame(metrics)
        
        # Metrics table
        st.subheader("Performance Summary")
        st.dataframe(metrics_df, width='stretch')
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curves")
            roc_fig = plot_roc_curves(
                data['models'], 
                data['X_test'], 
                data['Y_test'], 
                data['edge_index'],
                data['rt_meas_dim'],
                data.get('ensemble_model')
            )
            st.plotly_chart(roc_fig, use_container_width=True)  # plotly_chart doesn't have width parameter yet
        
        with col2:
            st.subheader("Precision-Recall Curves")
            pr_fig = plot_precision_recall_curves(
                data['models'], 
                data['X_test'], 
                data['Y_test'], 
                data['edge_index'],
                data['rt_meas_dim'],
                data.get('ensemble_model')
            )
            st.plotly_chart(pr_fig, use_container_width=True)  # plotly_chart doesn't have width parameter yet
        
        st.subheader("F1 Score vs Threshold")
        f1_fig = plot_f1_scores(
            data['models'], 
            data['X_test'], 
            data['Y_test'], 
            data['edge_index'],
            data['rt_meas_dim'],
            data.get('ensemble_model')
        )
        st.plotly_chart(f1_fig, use_container_width=True)  # plotly_chart doesn't have width parameter yet
        
        # Detailed metrics
        st.subheader("Detailed Metrics by Model")
        selected_model = st.selectbox("Select Model", metrics_df['model'].tolist())
        
        model_metrics = metrics_df[metrics_df['model'] == selected_model].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{float(model_metrics['accuracy']):.4f}")
        with col2:
            st.metric("Precision", f"{float(model_metrics['precision']):.4f}")
        with col3:
            st.metric("Recall", f"{float(model_metrics['recall']):.4f}")
        with col4:
            st.metric("F1 Score", f"{float(model_metrics['f1']):.4f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUC-ROC", f"{float(model_metrics['auc']):.4f}")
        with col2:
            st.metric("True Positives", int(model_metrics['TP']))
        with col3:
            st.metric("True Negatives", int(model_metrics['TN']))
        with col4:
            st.metric("False Positives", int(model_metrics['FP']))
    
    # Tab 3: Live Inference
    with tab3:
        st.header("Live Inference")
        st.markdown("Upload JSON input or use API endpoint for real-time predictions")
        
        inference_method = st.radio(
            "Inference Method",
            ["Local Models", "FastAPI Endpoint"],
            horizontal=True
        )
        
        if inference_method == "Local Models":
            st.subheader("JSON Input")
            
            # Example JSON
            example_json = {
                "features": [[0.5] * 135 for _ in range(26)]  # 26 nodes, 135 features each
            }
            
            with st.expander("📝 Example JSON Format"):
                st.json(example_json)
                st.code(json.dumps(example_json, indent=2))
            
            # Input options
            input_method = st.radio(
                "Input Method",
                ["Upload JSON File", "Paste JSON"],
                horizontal=True
            )
            
            json_data = None
            
            if input_method == "Upload JSON File":
                uploaded_file = st.file_uploader("Choose JSON file", type=['json'])
                if uploaded_file is not None:
                    json_data = json.load(uploaded_file)
            
            else:
                json_text = st.text_area("Paste JSON here", height=200)
                if json_text:
                    try:
                        json_data = json.loads(json_text)
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
            
            if json_data and st.button("🔮 Predict", type="primary"):
                with st.spinner("Making predictions..."):
                    results = predict_from_json(
                        json_data,
                        data['models'],
                        data['edge_index'],
                        data['rt_meas_dim'],
                        data['action_mask'],
                        data.get('ensemble_model')
                    )
                
                if 'error' in results:
                    st.error(f"Error: {results['error']}")
                else:
                    st.success("Predictions completed!")
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    for model_name, result in results.items():
                        with st.expander(f"📊 {model_name} Results"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Max Probability", f"{result['max_prob']:.4f}")
                            with col2:
                                st.metric("Mean Probability", f"{result['mean_prob']:.4f}")
                            if 'alpha' in result:
                                st.metric("GCN Weight (alpha)", f"{result['alpha']:.2f}")
                            
                            st.write("**Probabilities by Action Node:**")
                            prob_df = pd.DataFrame({
                                'Action Node': data['action_mask'],
                                'Probability': result['probabilities'][0] if isinstance(result['probabilities'][0], list) else result['probabilities']
                            })
                            st.dataframe(prob_df, width='stretch')
                            
                            st.write("**Predictions (Threshold = 0.5):**")
                            pred_df = pd.DataFrame({
                                'Action Node': data['action_mask'],
                                'Prediction': ['Attack' if p == 1 else 'Normal' for p in (result['predictions'][0] if isinstance(result['predictions'][0], list) else result['predictions'])]
                            })
                            st.dataframe(pred_df, width='stretch')
        
        else:  # FastAPI Endpoint
            st.subheader("FastAPI Endpoint")
            api_url = st.text_input(
                "API URL",
                value="http://localhost:8000/predict",
                help="Enter the FastAPI /predict endpoint URL"
            )
            
            json_text = st.text_area("JSON Input", height=200)
            
            if st.button("🚀 Fetch from API", type="primary") and json_text:
                try:
                    json_data = json.loads(json_text)
                    with st.spinner("Fetching from API..."):
                        results = predict_from_api(api_url, json_data)
                    
                    if 'error' in results:
                        st.error(f"API Error: {results['error']}")
                    else:
                        st.success("API Response received!")
                        st.json(results)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

if __name__ == "__main__":
    main()

