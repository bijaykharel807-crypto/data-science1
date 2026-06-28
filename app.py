import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes   # <-- 10‑feature dataset
from sklearn.model_selection import train_test_split
import os

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Model Dashboard", layout="wide")

# -------------------------------
# Helper functions
# -------------------------------
@st.cache_resource
def load_models():
    """Load all .pkl models from the 'saved_models' folder."""
    model_dir = "saved_models"
    models = {}
    if not os.path.exists(model_dir):
        st.warning(f"Folder '{model_dir}' not found. Please create it and place your .pkl files there.")
        return models
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            try:
                models[file.replace(".pkl", "")] = joblib.load(os.path.join(model_dir, file))
            except Exception as e:
                st.error(f"Failed to load {file}: {e}")
    return models

@st.cache_data
def load_data():
    """
    Load dataset: either from uploaded CSV or use Diabetes (10 features) as fallback.
    Returns: (DataFrame, feature_columns_list, target_column_name)
    """
    uploaded_file = st.session_state.get("uploaded_file", None)
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Assume the last column is the target
            target_col = df.columns[-1]
            feature_cols = [col for col in df.columns if col != target_col]
            return df, feature_cols, target_col
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None, None, None
    
    # Fallback: Diabetes dataset (10 features)
    data = load_diabetes(as_frame=True)
    df = data.frame
    target_col = "target"
    feature_cols = [col for col in df.columns if col != target_col]
    return df, feature_cols, target_col

@st.cache_data
def prepare_data(df, target_col):
    """Split data into X and y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Session state initialisation
# -------------------------------
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Details", "Dataset", "EDA", "Model Comparison", "Prediction"]
)

# -------------------------------
# Load models (once)
# -------------------------------
models = load_models()

if not models:
    st.sidebar.warning("No models found. Please place .pkl files in 'saved_models/'.")

# -------------------------------
# Page: Project Details
# -------------------------------
if page == "Project Details":
    st.title("📊 Project Overview")
    st.markdown("""
    This dashboard is built to compare multiple regression models and make predictions.

    **Models available:**
    - Linear Regression (best performing)
    - Lasso
    - Ridge
    - Random Forest
    - Linear Regression (standard)

    All models were trained on a dataset with **10 features** (like the Diabetes dataset).  
    If your models used a different dataset, please upload the correct CSV on the **Dataset** page.

    Use the sidebar to navigate through the sections:
    - **Dataset**: view and explore the data.
    - **EDA**: visualise distributions and correlations.
    - **Model Comparison**: see performance metrics side‑by‑side.
    - **Prediction**: input new data and get predictions from any model.
    """)

# -------------------------------
# Page: Dataset
# -------------------------------
elif page == "Dataset":
    st.title("📁 Dataset")
    
    uploaded = st.file_uploader("Upload your own CSV (optional)", type=["csv"])
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded
        st.success("Dataset loaded from upload!")
    
    # Load data
    data_result = load_data()
    if data_result is None or data_result[0] is None:
        st.warning("Could not load dataset. Please upload a CSV or ensure Diabetes is available.")
        st.stop()
    
    df, feature_cols, target_col = data_result
    st.write(f"**Target column:** `{target_col}`")
    st.write(f"**Number of rows:** {df.shape[0]}   **Features:** {df.shape[1]-1}")
    st.dataframe(df.head(10))
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# -------------------------------
# Page: EDA
# -------------------------------
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")
    
    data_result = load_data()
    if data_result is None or data_result[0] is None:
        st.warning("No data available. Please upload a CSV.")
        st.stop()
    
    df, feature_cols, target_col = data_result
    
    st.subheader("Distributions")
    fig, axes = plt.subplots(nrows=(len(df.columns)+2)//3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------
# Page: Model Comparison
# -------------------------------
elif page == "Model Comparison":
    st.title("📊 Model Performance Comparison")
    
    if not models:
        st.error("No models loaded. Please place .pkl files in 'saved_models/'.")
        st.stop()
    
    data_result = load_data()
    if data_result is None or data_result[0] is None:
        st.warning("No data available. Please upload a CSV.")
        st.stop()
    
    df, feature_cols, target_col = data_result
    X_train, X_test, y_train, y_test = prepare_data(df, target_col)
    
    results = []
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append({
                "Model": name,
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R²": r2
            })
        except Exception as e:
            st.error(f"Error evaluating {name}: {e}")
    
    if not results:
        st.warning("No models could be evaluated.")
        st.stop()
    
    df_results = pd.DataFrame(results).sort_values("R²", ascending=False)
    st.subheader("Performance Metrics")
    st.dataframe(df_results.style.highlight_max(subset=["R²"], color="lightgreen").highlight_min(subset=["MSE", "RMSE"], color="salmon"))
    
    st.subheader("R² Score Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_results["Model"], df_results["R²"], color="skyblue")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.3f}", ha="center", va="bottom")
    st.pyplot(fig)

# -------------------------------
# Page: Prediction
# -------------------------------
elif page == "Prediction":
    st.title("🔮 Make a Prediction")
    
    if not models:
        st.error("No models loaded. Please place .pkl files in 'saved_models/'.")
        st.stop()
    
    data_result = load_data()
    if data_result is None or data_result[0] is None:
        st.warning("No data available. Please upload a CSV to infer feature names.")
        st.stop()
    
    df, feature_cols, target_col = data_result
    
    model_names = list(models.keys())
    selected_model_name = st.selectbox("Choose a model", model_names)
    model = models[selected_model_name]
    
    st.subheader("Enter feature values")
    input_values = {}
    cols_per_row = 3
    cols = st.columns(cols_per_row)
    for i, feature in enumerate(feature_cols):
        with cols[i % cols_per_row]:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].mean())
            input_values[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=0.01,
                format="%.4f"
            )
    
    if st.button("Predict", type="primary"):
        try:
            X_input = np.array([input_values[f] for f in feature_cols]).reshape(1, -1)
            prediction = model.predict(X_input)[0]
            st.success(f"**Predicted {target_col}:** {prediction:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit")