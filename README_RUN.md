# How to Run Cerebrum-IDS from Terminal

## Quick Start

### Step 1: Open Terminal
- In VS Code: Press `` Ctrl+` `` (backtick) or go to **Terminal → New Terminal**
- Or open PowerShell/Command Prompt in the project folder

### Step 2: Navigate to src directory
```bash
cd src
```

### Step 3: Run the script
Choose one of the following options:

#### Option A: Simple Terminal Output (Recommended)
```bash
python -u run_simple.py
```
This displays all results directly in the terminal.

#### Option B: With Visualization Files
```bash
python run_gnn_ids_dataset1.py
```
This creates PNG files for ROC curve and confusion matrix.

---

## Complete Command Sequence

Copy and paste these commands one by one:

```bash
cd src
python -u run_simple.py
```

Or run from the root directory:

```bash
python -u src/run_simple.py
```

---

## What You'll See

The terminal will display:
1. ✓ Attack graph loading
2. ✓ Dataset loading
3. ✓ Model training progress for each model (NN, GCN, GCN-EW, GAT)
4. ✓ Training and validation accuracy
5. ✓ Performance metrics table
6. ✓ Detailed results for each model
7. ✓ Best model summary

---

## Troubleshooting

### "python: command not found"
Try:
```bash
python3 -u run_simple.py
```

### "Module not found" errors
Install dependencies:
```bash
pip install torch torch-geometric scikit-learn matplotlib pandas numpy
```

### Want to see output in real-time?
Use the `-u` flag for unbuffered output:
```bash
python -u run_simple.py
```

---

## Expected Output Example

```
================================================================================
Cerebrum-IDS - Starting Execution
================================================================================

Importing libraries...
✓ PyTorch imported
✓ Project modules imported

================================================================================
STEP 1: Loading Attack Graph
================================================================================
✓ Attack graph loaded: 26 nodes, 26 edges
✓ Action nodes: [0, 2, 4, 7, 9, 12, 14]

... (training results) ...

================================================================================
TRAINING SUMMARY TABLE
================================================================================

Model      Train Acc       Val Acc         Train Loss      Val Loss
--------------------------------------------------------------------------------
NN         0.7447          0.7424          0.9820          0.9970
GCN        0.9027          0.9027          1.2363          1.2379
GCN-EW     0.8911          0.8910          1.2281          1.2307
GAT        0.7128          0.7139          0.7822          0.7978

... (performance metrics) ...
```

---

That's it! Just navigate to `src` and run the script.

