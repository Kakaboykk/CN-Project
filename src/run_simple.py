#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple script to run Cerebrum-IDS with clear terminal output
"""
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("Cerebrum-IDS - Starting Execution")
print("="*80)
print("\nImporting libraries...")
sys.stdout.flush()

import torch
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
print("✓ PyTorch imported")
sys.stdout.flush()

from ag_utils import Corpus, parse_ag_file, parse_node_properties
from models import NN, GCN, GAT, GCN_EW
from model_utils import train, evaluate_performance, predict_prob
from ensemble_model import EnsembleModel, evaluate_ensemble, predict_ensemble_prob
from sklearn.metrics import roc_curve, auc
print("✓ Project modules imported")
sys.stdout.flush()

print("\n" + "="*80)
print("STEP 1: Loading Attack Graph")
print("="*80)
sys.stdout.flush()

# Get the script directory and construct absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
attack_graph_path = os.path.join(project_root, 'mulval_attack_graph', 'AttackGraph.dot')
nodes, edges, node_properties = parse_ag_file(attack_graph_path)
node_dict = parse_node_properties(nodes, node_properties)
corpus = Corpus(node_dict)
node_features = corpus.get_node_features()
action_nodes = corpus.get_action_nodes()
action_node_idx = list(action_nodes.keys())
action_mask = action_node_idx

print(f"✓ Attack graph loaded: {len(nodes)} nodes, {len(edges)} edges")
print(f"✓ Action nodes: {action_node_idx}")
sys.stdout.flush()

print("\n" + "="*80)
print("STEP 2: Building Graph Structure")
print("="*80)
sys.stdout.flush()

adj_matrix = torch.zeros(len(nodes), len(nodes))
for edge in edges:
    source_node, target_node = edge
    source_index = nodes.index(source_node)
    target_index = nodes.index(target_node)
    adj_matrix[source_index][target_index] = 1
edge_index = adj_matrix.nonzero().t().contiguous()
print("✓ Graph structure built")
sys.stdout.flush()

print("\n" + "="*80)
print("STEP 3: Loading Dataset")
print("="*80)
sys.stdout.flush()

data_path = os.path.join(project_root, 'datasets', 'synt', '')
sample_method = 'synthetic'
X_train = torch.load(data_path+'X_train-{}.pth'.format(sample_method))
X_val   = torch.load(data_path+'X_val-{}.pth'.format(sample_method))
X_test  = torch.load(data_path+'X_test-{}.pth'.format(sample_method))
Y_train = torch.load(data_path+'Y_train-{}.pth'.format(sample_method))
Y_val   = torch.load(data_path+'Y_val-{}.pth'.format(sample_method))
Y_test  = torch.load(data_path+'Y_test-{}.pth'.format(sample_method))

print(f"✓ Dataset loaded:")
print(f"  Training: {X_train.shape}, Labels: {Y_train.shape}")
print(f"  Validation: {X_val.shape}, Labels: {Y_val.shape}")
print(f"  Test: {X_test.shape}, Labels: {Y_test.shape}")
sys.stdout.flush()

print("\n" + "="*80)
print("STEP 4: Initializing Models")
print("="*80)
sys.stdout.flush()

in_dim = X_train.shape[-1]
rt_meas_dim = 78
hidden_dim = 20
out_dim = 1
lr = 0.001
device = 'cpu'
num_epochs = 2

models = {}
models['NN'] = NN(rt_meas_dim, hidden_dim, out_dim)
models['GCN'] = GCN(in_dim, hidden_dim, out_dim)
models['GCN-EW'] = GCN_EW(in_dim, hidden_dim, out_dim, edge_index)
models['GAT'] = GAT(hidden_channels=hidden_dim, heads=4, in_dim=in_dim, out_dim=out_dim)

print("✓ Models initialized: NN, GCN, GCN-EW, GAT")
sys.stdout.flush()

print("\n" + "="*80)
print("STEP 5: Training Models")
print("="*80)
sys.stdout.flush()

training_results = {}

for name, model in models.items():
    model.name = name
    model.action_mask = action_mask
    print(f"\n{'='*80}")
    print(f"Training {name} Model...")
    print('-'*80)
    sys.stdout.flush()
    
    train(model, lr, num_epochs, X_train, Y_train, X_val, Y_val, edge_index, rt_meas_dim, device)
    
    training_results[name] = {
        'train_acc': model.stat["acc_train"][-1],
        'val_acc': model.stat["acc_val"][-1],
        'train_loss': model.stat["loss_train"][-1],
        'val_loss': model.stat["loss_val"][-1]
    }
    
    print(f"\n{name} Training Results:")
    print(f"  Training Accuracy:   {model.stat['acc_train'][-1]:.4f} ({model.stat['acc_train'][-1]*100:.2f}%)")
    print(f"  Validation Accuracy: {model.stat['acc_val'][-1]:.4f} ({model.stat['acc_val'][-1]*100:.2f}%)")
    print(f"  Training Loss:       {model.stat['loss_train'][-1]:.4f}")
    print(f"  Validation Loss:     {model.stat['loss_val'][-1]:.4f}")
    sys.stdout.flush()

print("\n" + "="*80)
print("TRAINING SUMMARY TABLE")
print("="*80)
print(f"\n{'Model':<10} {'Train Acc':<15} {'Val Acc':<15} {'Train Loss':<15} {'Val Loss':<15}")
print('-'*80)
for name, results in training_results.items():
    print(f'{name:<10} {results["train_acc"]:<15.4f} {results["val_acc"]:<15.4f} '
          f'{results["train_loss"]:<15.4f} {results["val_loss"]:<15.4f}')
sys.stdout.flush()

print("\n" + "="*80)
print("STEP 6: Evaluating Performance on Test Set")
print("="*80)
sys.stdout.flush()

metrics = evaluate_performance(models, X_test, Y_test, edge_index, device)

print("\nPerformance Metrics Table (Individual Models):")
print('-'*80)
df = pd.DataFrame(metrics)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.to_string(index=False))
sys.stdout.flush()

print("\n" + "="*80)
print("DETAILED RESULTS BY MODEL")
print("="*80)

for idx, row in df.iterrows():
    print(f"\n{row['model']} Model:")
    print(f"  Accuracy:    {float(row['accuracy']):.4f} ({float(row['accuracy'])*100:.2f}%)")
    print(f"  Precision:   {float(row['precision']):.4f} ({float(row['precision'])*100:.2f}%)")
    print(f"  Recall:      {float(row['recall']):.4f} ({float(row['recall'])*100:.2f}%)")
    print(f"  F1-Score:    {float(row['f1']):.4f}")
    print(f"  AUC-ROC:     {float(row['auc']):.4f}")
    print(f"  TP: {row['TP']}, TN: {row['TN']}, FP: {row['FP']}, FN: {row['FN']}")
    sys.stdout.flush()

print("\n" + "="*80)
print("ENSEMBLE MODEL EVALUATION")
print("="*80)
sys.stdout.flush()

# Create ensemble model (GCN + GAT)
print("\nCreating Ensemble Model (GCN + GAT)...")
ensemble_alpha = 0.6  # 60% GCN, 40% GAT
ensemble_model = EnsembleModel(
    gcn_model=models['GCN'],
    gat_model=models['GAT'],
    alpha=ensemble_alpha,
    learnable_alpha=False
)
ensemble_model.name = 'Ensemble'
ensemble_model.action_mask = action_mask

print(f"✓ Ensemble model created with alpha={ensemble_alpha} (GCN weight: {ensemble_alpha:.2f}, GAT weight: {1-ensemble_alpha:.2f})")

# Evaluate ensemble
print("Evaluating ensemble model...")
ensemble_metrics = evaluate_ensemble(ensemble_model, X_test, Y_test, edge_index, device)
print(f"✓ Ensemble evaluation completed")

# Add ensemble to metrics
all_metrics = metrics + [ensemble_metrics]
df_all = pd.DataFrame(all_metrics)

print("\nUpdated Performance Metrics Table (Including Ensemble):")
print('-'*80)
print(df_all.to_string(index=False))
sys.stdout.flush()

print("\n" + "="*80)
print("BEST MODELS (Including Ensemble)")
print("="*80)
best_acc = max(df_all.iterrows(), key=lambda x: float(x[1]['accuracy']))
best_auc = max(df_all.iterrows(), key=lambda x: float(x[1]['auc']))
best_prec = max(df_all.iterrows(), key=lambda x: float(x[1]['precision']))
best_recall = max(df_all.iterrows(), key=lambda x: float(x[1]['recall']))

print(f"✓ Best Accuracy:    {best_acc[1]['model']} ({best_acc[1]['accuracy']})")
print(f"✓ Best AUC-ROC:     {best_auc[1]['model']} ({best_auc[1]['auc']})")
print(f"✓ Best Precision:   {best_prec[1]['model']} ({best_prec[1]['precision']})")
print(f"✓ Best Recall:      {best_recall[1]['model']} ({best_recall[1]['recall']})")

# Show ensemble results
print("\n" + "="*80)
print("ENSEMBLE MODEL RESULTS")
print("="*80)
print(f"Ensemble (GCN+GAT) Performance:")
print(f"  Accuracy:    {float(ensemble_metrics['accuracy']):.4f} ({float(ensemble_metrics['accuracy'])*100:.2f}%)")
print(f"  Precision:   {float(ensemble_metrics['precision']):.4f} ({float(ensemble_metrics['precision'])*100:.2f}%)")
print(f"  Recall:      {float(ensemble_metrics['recall']):.4f} ({float(ensemble_metrics['recall'])*100:.2f}%)")
print(f"  F1-Score:    {float(ensemble_metrics['f1']):.4f}")
print(f"  AUC-ROC:     {float(ensemble_metrics['auc']):.4f}")
print(f"  GCN Weight:  {float(ensemble_metrics['gcn_weight']):.4f}")
print(f"  GAT Weight:  {float(ensemble_metrics['gat_weight']):.4f}")
print(f"  TP: {ensemble_metrics['TP']}, TN: {ensemble_metrics['TN']}, FP: {ensemble_metrics['FP']}, FN: {ensemble_metrics['FN']}")
sys.stdout.flush()

# Save trained models and statistics for dashboard reuse
print("\nSaving models and training statistics...")
artifact_dir = Path('../artifacts').resolve()
artifact_dir.mkdir(parents=True, exist_ok=True)

torch.save(models['NN'].state_dict(), artifact_dir / 'nn.pt')
torch.save(models['GCN'].state_dict(), artifact_dir / 'gcn.pt')
torch.save(models['GCN-EW'].state_dict(), artifact_dir / 'gcn_ew.pt')
torch.save(models['GAT'].state_dict(), artifact_dir / 'gat.pt')
torch.save({'alpha': ensemble_model.get_alpha()}, artifact_dir / 'ensemble_meta.pt')

def _to_float_list(values):
    return [float(v) for v in values] if values else []

stats_payload = {}
for name, model in models.items():
    stat = getattr(model, 'stat', {})
    stats_payload[name] = {
        'loss_train': _to_float_list(stat.get('loss_train', [])),
        'loss_val': _to_float_list(stat.get('loss_val', [])),
        'acc_train': _to_float_list(stat.get('acc_train', [])),
        'acc_val': _to_float_list(stat.get('acc_val', []))
    }

# Store ensemble metrics as well
ensemble_stats_payload = {}
for key, value in ensemble_metrics.items():
    try:
        ensemble_stats_payload[key] = float(value)
    except (TypeError, ValueError):
        ensemble_stats_payload[key] = value
stats_payload['Ensemble'] = ensemble_stats_payload

with open(artifact_dir / 'training_stats.json', 'w') as f:
    json.dump(stats_payload, f, indent=2)

print(f"✓ Saved artifacts to {artifact_dir}")
sys.stdout.flush()

print("\n" + "="*80)
print("STEP 7: Generating Visualizations")
print("="*80)
sys.stdout.flush()

# Plot ROC Curve
print("\nGenerating ROC Curve...")
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    prob = predict_prob(model, X_test, edge_index)
    y_probs = prob.view(-1, 2)
    
    fpr, tpr, thresholds = roc_curve(Y_test.view(-1), y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, label='{} (AUC = {:.4f})'.format(model.name, roc_auc), linewidth=2)
    
# Add ensemble to ROC curve
ensemble_prob = predict_ensemble_prob(ensemble_model, X_test, edge_index, rt_meas_dim, device)
ensemble_y_probs = ensemble_prob.view(-1, 2)
ensemble_fpr, ensemble_tpr, _ = roc_curve(Y_test.view(-1).numpy(), ensemble_y_probs[:, 1].numpy())
ensemble_roc_auc = auc(ensemble_fpr, ensemble_tpr)
ax.plot(ensemble_fpr, ensemble_tpr, label='Ensemble (AUC = {:.4f})'.format(ensemble_roc_auc), 
        linewidth=3, linestyle='-.', color='purple')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
print("✓ ROC curve saved to: roc_curve.png")
print("  → ROC curve window opened. Close it when done viewing.")
plt.show(block=False)  # Open in non-blocking mode so both can be open

# Plot Confusion Matrix Components
print("Generating Confusion Matrix Components...")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

bar_width = 0.6
labels = ['TN', 'FP', 'FN', 'TP']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']  # Blue, Green, Red, Orange, Purple

# Include ensemble in the plot
all_model_names = list(models.keys()) + ['Ensemble']

for i, label in enumerate(labels):
    for j, name in enumerate(all_model_names):
        value = int(all_metrics[j][label])
        model_color = colors[j] if j < len(colors) else '#9b59b6'  # Purple for ensemble
        axs[i].bar(j, value, width=bar_width, label=name, color=model_color, alpha=0.7)
        # Add value labels on bars
        axs[i].text(j, value, str(int(value)), ha='center', va='bottom', fontsize=9)
    axs[i].set_xticks(np.arange(len(all_model_names)))
    axs[i].set_xticklabels(all_model_names, fontsize=10)
    axs[i].set_title(f'{label} (True Negatives)' if label == 'TN' else 
                     f'{label} (False Positives)' if label == 'FP' else
                     f'{label} (False Negatives)' if label == 'FN' else
                     f'{label} (True Positives)', fontsize=11, fontweight='bold')
    axs[i].set_ylabel('Count', fontsize=10)
    axs[i].grid(True, alpha=0.3, axis='y')

plt.suptitle('Confusion Matrix Components by Model', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('confusion_matrix_components.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix components saved to: confusion_matrix_components.png")
print("  → Confusion matrix window opened. Close it when done viewing.")
plt.show(block=False)  # Open in non-blocking mode so both can be open

print("\n" + "="*80)
print("EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)
print("✓ All graphs have been displayed and saved")
print("\nNote: Both graph windows are open. Close them when you're done viewing.")
print("      Graphs are also saved as PNG files in the current directory.")
sys.stdout.flush()

# Keep the script running so graphs stay open
# User can close the graph windows manually
input("\nPress Enter to exit (this will close any open graph windows)...")
plt.close('all')  # Close all figures when user presses Enter

