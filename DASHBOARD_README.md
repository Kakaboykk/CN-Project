# Cerebrum-IDS Dashboard

Interactive Streamlit dashboard for visualization and quick testing of the Cerebrum-IDS system.

## Features

### 📈 Attack Graph Tab
- **Interactive Visualization**: Uses PyVis to create an interactive network graph
- **Node Information**: Click on nodes to see details
- **Graph Statistics**: View graph structure information
- **Color Coding**: 
  - Red (Diamond): Action nodes
  - Teal (Box): Privilege nodes
  - Light Teal: Other nodes

### 📊 Model Metrics Tab
- **Performance Summary Table**: All metrics for all models
- **ROC Curves**: Interactive Plotly visualization
- **Precision-Recall Curves**: Compare model performance
- **F1 Score vs Threshold**: Find optimal threshold
- **Detailed Metrics**: Per-model breakdown with confusion matrix components

### 🔮 Live Inference Tab
- **Local Models**: Upload JSON or paste JSON for predictions
- **FastAPI Integration**: Connect to `/predict` endpoint
- **Real-time Results**: See probabilities and predictions
- **Multiple Models**: Compare predictions across all models

## Installation

1. Install dashboard dependencies:
```bash
pip install -r requirements_dashboard.txt
```

Or install individually:
```bash
pip install streamlit plotly pyvis pandas numpy torch torch-geometric scikit-learn requests
```

## Running the Dashboard

From the `src` directory:
```bash
streamlit run dashboard.py
```

Or from the project root:
```bash
streamlit run src/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## JSON Input Format

For Live Inference, use the following JSON format:

```json
{
  "features": [
    [0.5, 0.3, 0.2, ...],  // Node 0 features (135 features)
    [0.4, 0.6, 0.1, ...],  // Node 1 features (135 features)
    ...
    [0.3, 0.5, 0.7, ...]   // Node 25 features (135 features)
  ]
}
```

**Note**: 
- 26 nodes total (matching attack graph)
- 135 features per node (57 node features + 78 real-time measurements)

## FastAPI Integration

If you have a FastAPI server running with a `/predict` endpoint:

1. Start your FastAPI server
2. Enter the API URL in the dashboard (e.g., `http://localhost:8000/predict`)
3. Paste your JSON input
4. Click "Fetch from API"

## Theme Support

The dashboard supports light and dark themes (selectable in sidebar). The visual style uses:
- Rounded corners on all components
- Smooth transitions
- Consistent color scheme
- Responsive layout

## Troubleshooting

### PyVis not displaying
- Make sure `pyvis` is installed: `pip install pyvis`
- Check browser console for errors

### Models not loading
- Ensure you've run `run_simple.py` at least once to generate data
- Check that dataset files exist in `datasets/synt/`

### API connection errors
- Verify FastAPI server is running
- Check the API URL is correct
- Ensure CORS is enabled on the API server

## Features Overview

✅ Interactive attack graph visualization  
✅ ROC, Precision-Recall, and F1 curves  
✅ Real-time inference with JSON input  
✅ FastAPI endpoint integration  
✅ Dark/Light theme support  
✅ Rounded visual style  
✅ Responsive layout  
✅ Multiple model comparison  

