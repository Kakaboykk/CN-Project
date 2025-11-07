# How to Train Models for Dashboard

## Quick Steps

### Step 1: Train the Models
Open a terminal and run:
```bash
cd src
python -u run_simple.py
```

This will:
- Load the attack graph
- Load the dataset
- Train all 4 models (NN, GCN, GCN-EW, GAT)
- Create and evaluate the Ensemble model
- Display all results

### Step 2: Wait for Training to Complete
The script will:
- Train each model (takes a few minutes)
- Evaluate performance
- Generate visualizations
- Show completion message

### Step 3: Refresh the Dashboard
After training completes:
1. Go back to your dashboard browser tab
2. Click the **"Rerun"** button (top right) or press `R`
3. The warnings will disappear and you'll see accurate metrics!

## What Happens During Training

The script will show:
- Training progress for each model
- Training and validation accuracy
- Performance metrics on test set
- Ensemble model evaluation
- ROC curves and confusion matrices

## Expected Output

You'll see output like:
```
Training NN Model...
NN Training Results:
  Training Accuracy:   0.7447 (74.47%)
  Validation Accuracy: 0.7424 (74.24%)

Training GCN Model...
GCN Training Results:
  Training Accuracy:   0.9027 (90.27%)
  Validation Accuracy: 0.9027 (90.27%)

... (and so on for all models)

ENSEMBLE MODEL EVALUATION
Creating Ensemble Model (GCN + GAT)...
✓ Ensemble model created with alpha=0.6
```

## After Training

Once training is complete:
- Models will have the `stat` attribute with training history
- Dashboard will show accurate metrics
- All visualizations will work properly
- Ensemble model will be available

## Note

- Training takes a few minutes (depends on your system)
- You only need to train once (models stay in memory during dashboard session)
- If you restart the dashboard, you may need to train again (or save/load model checkpoints)

