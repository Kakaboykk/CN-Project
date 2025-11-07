"""
Ensemble Model combining GCN and GAT predictions
Uses weighted averaging to improve accuracy and robustness
"""

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from model_utils import predict_prob


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines GCN and GAT predictions using weighted averaging
    
    Args:
        gcn_model: Trained GCN model
        gat_model: Trained GAT model
        alpha: Weight for GCN (1-alpha is weight for GAT). Default 0.6
        learnable_alpha: If True, alpha is a learnable parameter. Default False
    """
    def __init__(self, gcn_model, gat_model, alpha=0.6, learnable_alpha=False):
        super(EnsembleModel, self).__init__()
        self.gcn_model = gcn_model
        self.gat_model = gat_model
        self.learnable_alpha = learnable_alpha
        
        if learnable_alpha:
            # Make alpha a learnable parameter (constrained to [0, 1])
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            # Fixed alpha value
            self.register_buffer('alpha', torch.tensor(alpha))
        
        # Ensure models are in eval mode
        self.gcn_model.eval()
        self.gat_model.eval()
    
    def forward(self, x, edge_index):
        """
        Forward pass: weighted combination of GCN and GAT outputs
        
        Args:
            x: Input node features
            edge_index: Graph edge indices
            
        Returns:
            Combined output predictions
        """
        # Get predictions from both models
        with torch.no_grad():
            gcn_output = self.gcn_model(x, edge_index)
            gat_output = self.gat_model(x, edge_index)
        
        # Ensure alpha is in [0, 1] if learnable
        if self.learnable_alpha:
            alpha = torch.sigmoid(self.alpha)
        else:
            alpha = self.alpha
        
        # Weighted combination
        ensemble_output = alpha * gcn_output + (1 - alpha) * gat_output
        
        return ensemble_output
    
    def get_alpha(self):
        """Get current alpha value (in [0, 1])"""
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha).item()
        else:
            return self.alpha.item()


def predict_ensemble_prob(ensemble_model, X, edge_index, rt_meas_dim=78, device='cpu'):
    """
    Get probability predictions from ensemble model
    
    Args:
        ensemble_model: EnsembleModel instance
        X: Input features
        edge_index: Graph edge indices
        rt_meas_dim: Real-time measurement dimension
        device: Device to run on
        
    Returns:
        Probability predictions in shape (batch, num_action_nodes, 2)
    """
    ensemble_model.eval()
    mask = ensemble_model.gcn_model.action_mask  # Use GCN's action_mask
    
    prob = torch.zeros((len(X), len(mask), 2), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Get probabilities from both models
        gcn_prob = predict_prob(ensemble_model.gcn_model, X, edge_index, rt_meas_dim, device)
        gat_prob = predict_prob(ensemble_model.gat_model, X, edge_index, rt_meas_dim, device)
        
        # Get alpha value
        alpha = ensemble_model.get_alpha()
        
        # Weighted combination of probabilities
        ensemble_prob = alpha * gcn_prob + (1 - alpha) * gat_prob
        
        prob = ensemble_prob
    
    return prob


def evaluate_ensemble(ensemble_model, X, y, edge_index, device='cpu'):
    """
    Evaluate ensemble model performance
    
    Args:
        ensemble_model: EnsembleModel instance
        X: Test features
        y: Test labels
        edge_index: Graph edge indices
        device: Device to run on
        
    Returns:
        Dictionary with performance metrics
    """
    ensemble_model.eval()
    mask = ensemble_model.gcn_model.action_mask
    
    # Get predictions
    prob = predict_ensemble_prob(ensemble_model, X, edge_index, device=device)
    pred_ts = torch.argmax(prob, dim=2)
    
    # Calculate accuracy
    accuracy = (pred_ts == y).sum().item() / (y.shape[0] * y.shape[1])
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y.flatten().numpy(), pred_ts.flatten().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y.view(-1).numpy(), pred_ts.view(-1).numpy(), average='macro', zero_division=0.0
    )
    
    TN, FP, FN, TP = conf_matrix.ravel()
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    # ROC curve
    y_probs = prob.view(-1, 2)
    fpr, tpr, thresholds = roc_curve(y.view(-1).numpy(), y_probs[:, 1].numpy())
    roc_auc = auc(fpr, tpr)
    
    # Get alpha value
    alpha = ensemble_model.get_alpha()
    
    metrics = {
        'model': 'Ensemble (GCN+GAT)',
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'TP': int(TP),
        'precision': f'{precision:.4f}',
        'recall': f'{recall:.4f}',
        'f1': f'{f1:.4f}',
        'auc': f'{roc_auc:.4f}',
        'fpr': f'{FPR:.4f}',
        'fnr': f'{FNR:.4f}',
        'accuracy': f'{accuracy:.4f}',
        'alpha': f'{alpha:.4f}',
        'gcn_weight': f'{alpha:.4f}',
        'gat_weight': f'{1-alpha:.4f}'
    }
    
    return metrics

