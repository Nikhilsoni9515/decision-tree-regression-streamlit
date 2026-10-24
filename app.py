"""Decision Tree Regressor - Streamlit UI
Based on DecisionTreeReg.ipynb (cardekho_data.csv -> predict Selling_Price).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree

DEFAULT_CSV = "cardekho_data.csv"
DEFAULT_TARGET = "Selling_Price"

st.set_page_config(page_title="Decision Tree Regressor", page_icon="🌳", layout="wide")
st.title("🌳 Decision Tree Regressor")
st.caption("Upload a dataset, pick a numeric target, train a tree, and predict values.")


@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = load_csv(uploaded)
    elif os.path.exists(DEFAULT_CSV):
        df = load_csv(DEFAULT_CSV)
        st.info(f"Using `{DEFAULT_CSV}`")
    else:
        st.warning("Upload a CSV to get started.")
        st.stop()

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.error("No numeric column found to use as a target.")
        st.stop()
    target = st.selectbox(
        "Target column (number to predict)", numeric_cols,
        index=numeric_cols.index(DEFAULT_TARGET) if DEFAULT_TARGET in numeric_cols else len(numeric_cols) - 1,
    )
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.01)
    random_state = st.number_input("Random state", value=42, step=1)

    st.header("2. Model")
    mode = st.radio("Training mode", ["Manual", "RandomizedSearchCV"])
    criteria = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
    if mode == "Manual":
        criterion = st.selectbox("criterion", criteria)
        unlimited = st.checkbox("max_depth = None (unlimited)", value=True)
        max_depth = None if unlimited else st.slider("max_depth", 1, 20, 5)
        min_samples_split = st.slider("min_samples_split", 2, 50, 2)
        min_samples_leaf = st.slider("min_samples_leaf", 1, 50, 1)
        max_features = st.selectbox("max_features", [None, "sqrt", "log2"])
        limit_leaves = st.checkbox("Limit max_leaf_nodes", value=False)
        max_leaf_nodes = st.slider("max_leaf_nodes", 2, 100, 20) if limit_leaves else None
    else:
        n_iter = st.slider("n_iter (random combinations)", 10, 200, 100)
        cv = st.slider("cv (folds)", 2, 10, 3)
        st.caption("Search space: criterion (4), max_depth 1-19, min_samples_split 2-19, "
                   "min_samples_leaf 1-19, max_features [None, sqrt, log2], max_leaf_nodes 2-19. "
                   "Scoring: R².")

    train_clicked = st.button("🚀 Train model", type="primary")


# ---------------------------------------------------------------- preprocessing
def preprocess(df, target):
    """Label-encode text feature columns (like the notebook does)."""
    data = df.copy()
    filled = []
    for c in data.columns:
        if data[c].isnull().any():
            if pd.api.types.is_numeric_dtype(data[c]):
                data[c] = data[c].fillna(data[c].median())
            else:
                data[c] = data[c].fillna(data[c].mode()[0])
            filled.append(c)
    encoders = {}
    for c in data.columns.drop(target):
        if not pd.api.types.is_numeric_dtype(data[c]):
            le = LabelEncoder()
            data[c] = le.fit_transform(data[c].astype(str))
            encoders[c] = le
    return data.drop(columns=[target]), data[target], encoders, filled


# ---------------------------------------------------------------- training
if train_clicked:
    X, y, encoders, filled = preprocess(df, target)
    allowed = criteria if y.min() >= 0 else [c for c in criteria if c != "poisson"]
    if mode == "Manual" and criterion not in allowed:
        st.error("`poisson` needs a non-negative target. Choose another criterion.")
        st.stop()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state))

    with st.spinner("Training..."):
        best_params = None
        if mode == "Manual":
            model = DecisionTreeRegressor(
                criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf, max_features=max_features,
                max_leaf_nodes=max_leaf_nodes, random_state=int(random_state))
            model.fit(X_train, y_train)
        else:
            params = {
                "criterion": allowed,
                "max_depth": list(range(1, 20)),
                "min_samples_split": list(range(2, 20)),
                "min_samples_leaf": list(range(1, 20)),
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": list(range(2, 20)),
            }
            search = RandomizedSearchCV(
                DecisionTreeRegressor(random_state=int(random_state)), params,
                n_iter=n_iter, cv=cv, random_state=int(random_state), n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = {k: (v.item() if hasattr(v, "item") else v)
                           for k, v in search.best_params_.items()}
            best_params["cv_r2"] = round(float(search.best_score_), 4)

    st.session_state["res"] = dict(
        model=model, X=X, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        y_pred=model.predict(X_test), encoders=encoders, target=target,
        best_params=best_params, filled=filled, raw=df.drop(columns=[target]),
    )

# ---------------------------------------------------------------- tabs
tab_data, tab_res, tab_tree, tab_pred = st.tabs(["📊 Data & EDA", "📈 Results", "🌲 Tree", "🔮 Predict"])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))
    st.subheader("Top 5 rows")
    st.dataframe(df.head())
    st.subheader("Last 5 rows")
    st.dataframe(df.tail())
    st.subheader("Null values per column")
    st.dataframe(df.isnull().sum().rename("nulls"))

    st.subheader(f"EDA on `{target}`")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                or df[c].nunique() <= 5]
    cat_cols = [c for c in cat_cols if c != target and df[c].nunique() <= 30]
    num_feats = [c for c in numeric_cols if c != target]

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df[target].dropna(), bins=30, color="#2e7d32", edgecolor="white")
        ax.set_title(f"{target} distribution"); ax.set_xlabel(target)
        st.pyplot(fig)
    with b:
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
            ax.set_title("Correlation heatmap")
            fig.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig)
    if cat_cols:
        cat = st.selectbox("Categorical column for charts", cat_cols)
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df.groupby(cat)[target].mean().sort_values().plot.bar(ax=ax, color="#1565c0")
            ax.set_title(f"Average {target} by {cat}"); ax.set_ylabel(f"mean {target}")
            st.pyplot(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df[cat].value_counts().plot.bar(ax=ax, color="#ef6c00")
            ax.set_title(f"Count by {cat}")
            st.pyplot(fig)
    if num_feats:
        xcol = st.selectbox("X-axis for scatter plot", num_feats,
                            index=num_feats.index("Present_Price") if "Present_Price" in num_feats else 0)
        fig, ax = plt.subplots(figsize=(7, 4))
        if cat_cols:
            for k, g in df.groupby(cat):
                ax.scatter(g[xcol], g[target], label=str(k), alpha=0.7, s=18)
            ax.legend(title=cat, fontsize=7)
        else:
            ax.scatter(df[xcol], df[target], alpha=0.7, s=18)
        ax.set_xlabel(xcol); ax.set_ylabel(target); ax.set_title(f"{target} vs {xcol}")
        st.pyplot(fig)

res = st.session_state.get("res")

with tab_res:
    if not res:
        st.info("Set options in the sidebar and click **Train model**.")
    else:
        m, y_test, y_pred = res["model"], res["y_test"], res["y_pred"]
        test_r2 = r2_score(y_test, y_pred)
        train_r2 = r2_score(res["y_train"], m.predict(res["X_train"]))
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test R²", f"{test_r2:.4f}")
        c2.metric("Train R²", f"{train_r2:.4f}")
        c3.metric("MAE", f"{mae:.3f}")
        c4.metric("RMSE", f"{rmse:.3f}")
        c5, c6, _, _ = st.columns(4)
        c5.metric("Tree depth", m.get_depth())
        c6.metric("Leaves", m.get_n_leaves())
        if train_r2 - test_r2 > 0.10:
            st.warning("Train R² is much higher than test R² - the tree is likely overfitting. "
                       "Try a smaller max_depth, larger min_samples_leaf or max_leaf_nodes.")
        if res["best_params"]:
            st.success(f"Best parameters: {res['best_params']}")
        if res["filled"]:
            st.caption(f"Missing values were filled in: {', '.join(res['filled'])}")

        left, right = st.columns(2)
        with left:
            st.subheader("Actual vs predicted (test set)")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, y_pred, alpha=0.7, s=18, color="#2e7d32")
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.legend()
            st.pyplot(fig)
        with right:
            st.subheader("Residuals (actual − predicted)")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(y_test - y_pred, bins=25, color="#1565c0", edgecolor="white")
            ax.axvline(0, color="red", linestyle="--")
            ax.set_xlabel("Residual")
            st.pyplot(fig)

        st.subheader("Feature importance")
        imp = pd.Series(m.feature_importances_, index=res["X"].columns).sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        imp.plot.barh(ax=ax, color="#2e7d32"); ax.set_xlabel("Importance")
        st.pyplot(fig)

with tab_tree:
    if not res:
        st.info("Train a model to see the tree.")
    else:
        m = res["model"]
        depth = max(1, m.get_depth())
        shown = st.slider("Levels to display", 1, depth, min(3, depth))
        fig, ax = plt.subplots(figsize=(max(12, 3 * shown + 6), 3 * shown + 2))
        plot_tree(m, feature_names=list(res["X"].columns), filled=True, rounded=True,
                  max_depth=shown, fontsize=8, ax=ax)
        st.pyplot(fig)

with tab_pred:
    if not res:
        st.info("Train a model first, then enter values here to get a prediction.")
    else:
        m, raw, encoders = res["model"], res["raw"], res["encoders"]
        st.subheader("Enter feature values")
        values = {}
        cols_ui = st.columns(3)
        for i, c in enumerate(res["X"].columns):
            with cols_ui[i % 3]:
                if c in encoders:
                    values[c] = st.selectbox(c, list(encoders[c].classes_), key=f"in_{c}")
                else:
                    s = raw[c].dropna()
                    lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                    if pd.api.types.is_integer_dtype(s):
                        values[c] = st.number_input(c, int(lo), int(hi), int(med), step=1, key=f"in_{c}")
                    else:
                        values[c] = st.number_input(c, lo, hi, med, key=f"in_{c}")
        if st.button("Predict", type="primary"):
            row = pd.DataFrame([values])
            for c, le in encoders.items():
                row[c] = le.transform(row[c].astype(str))
            pred = float(m.predict(row[res["X"].columns])[0])
            st.success(f"Predicted **{res['target']}**: {pred:,.3f}")
