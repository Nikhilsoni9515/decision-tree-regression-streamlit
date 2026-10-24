# 🌳 Decision Tree Regressor – Streamlit UI

Interactive UI for the regression workflow in `DecisionTreeReg.ipynb`
(`cardekho_data.csv` → predict `Selling_Price`).

## Features
- Upload any CSV (or keep `cardekho_data.csv` next to `app.py` to load it automatically)
- EDA: distribution, correlation heatmap, average/count by category, scatter plots
- Text columns are label-encoded automatically
- **Manual** training (criterion, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes) or **RandomizedSearchCV**
- Metrics: R² (train/test), MAE, RMSE, actual-vs-predicted, residuals, feature importance, tree plot
- Predict form generated from your dataset's columns

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy free
Push to GitHub, then on [share.streamlit.io](https://share.streamlit.io) choose the repo, branch `main`, main file `app.py`.
