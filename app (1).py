import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Try importing XGBoost (optional) ──────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Supply Chain Defect Dashboard",
    page_icon="🏭",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & MODEL TRAINING  (cached so it only runs once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Encode features, train all four models, return models + encoders."""
    supply_df = df.copy()

    # Create binary target
    mean_defect = supply_df["Defect rates"].mean()
    supply_df["High_Defect"] = (supply_df["Defect rates"] > mean_defect).astype(int)

    features = ["Supplier name", "Transportation modes", "Costs", "Lead time", "Routes", "Product type"]
    cat_features = ["Supplier name", "Transportation modes", "Routes", "Product type"]

    X = supply_df[features].copy()
    y = supply_df["High_Defect"]

    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, solver="liblinear"),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, use_label_encoder=False
        )

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        results[name] = {
            "model":        mdl,
            "accuracy":     accuracy_score(y_test, y_pred),
            "report":       classification_report(y_test, y_pred, output_dict=True),
            "conf_matrix":  confusion_matrix(y_test, y_pred),
            "y_test":       y_test,
            "y_pred":       y_pred,
        }

    # Feature importance from Random Forest
    rf_model = results["Random Forest"]["model"]
    importance_df = pd.DataFrame({
        "Feature":    X_train.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return results, encoders, X_train.columns.tolist(), importance_df, mean_defect


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — confusion matrix plot
# ══════════════════════════════════════════════════════════════════════════════
CMAP_MAP = {
    "Random Forest":       "Blues",
    "Logistic Regression": "Greens",
    "Decision Tree":       "Purples",
    "XGBoost":             "Oranges",
}

def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=CMAP_MAP.get(model_name, "Blues"),
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏭 Supply Chain Defect Prediction Dashboard")
st.markdown("Upload your supply chain Excel file to explore insights and predict defect risk.")

# ── File upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload `cleaned_supply_chain_data.xlsx`",
    type=["xlsx"],
    help="The same Excel file used in the original analysis.",
)

if uploaded_file is None:
    st.info("👆 Please upload the Excel file to get started.")
    st.stop()

# Load & train
df = load_data(uploaded_file)
results, encoders, feature_cols, importance_df, mean_defect = train_models(df)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Insights",
    "🤖 Model Performance",
    "🔍 Feature Importance",
    "🎯 Predict Defect Risk",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Exploratory Data Analysis")

    # KPI row
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Records",    len(df))
    k2.metric("Avg Defect Rate",  f"{df['Defect rates'].mean():.2f}%")
    k3.metric("Unique Suppliers", df["Supplier name"].nunique())

    st.divider()

    # Q1 — Supplier defect rates
    st.subheader("Q1 · Which suppliers have the highest defect rates?")
    supplier_defect = (
        df.groupby("Supplier name")["Defect rates"]
        .mean()
        .reset_index()
        .sort_values("Defect rates", ascending=False)
    )
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x="Supplier name", y="Defect rates",
        data=supplier_defect.head(5),
        hue="Supplier name", legend=False,
        palette="viridis", ax=ax1
    )
    ax1.set_title("Top 5 Suppliers by Average Defect Rate")
    ax1.set_xlabel("Supplier Name")
    ax1.set_ylabel("Avg Defect Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)

    st.divider()

    # Q2 — Lead time vs defects
    st.subheader("Q2 · Does manufacturing lead time affect defect rates?")
    corr_lt = df["Manufacturing lead time"].corr(df["Defect rates"])
    st.metric("Correlation (Lead Time vs Defects)", f"{corr_lt:.2f}")
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    sns.regplot(
        x="Manufacturing lead time", y="Defect rates",
        data=df, scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"}, ax=ax2
    )
    ax2.set_title("Manufacturing Lead Time vs Defect Rates")
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # Q3 — Transportation mode
    st.subheader("Q3 · Does transportation mode affect defect risk?")
    transport_defect = (
        df.groupby("Transportation modes")["Defect rates"]
        .mean()
        .reset_index()
        .sort_values("Defect rates", ascending=False)
    )
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="Transportation modes", y="Defect rates",
        data=transport_defect,
        hue="Transportation modes", legend=False,
        palette="coolwarm", ax=ax3
    )
    ax3.set_title("Average Defect Rate by Transportation Mode")
    plt.tight_layout()
    st.pyplot(fig3)

    st.divider()

    # Q4 — Shipping cost vs defects
    st.subheader("Q4 · Does shipping cost influence defect rates?")
    corr_sc = df["Shipping costs"].corr(df["Defect rates"])
    st.metric("Correlation (Shipping Cost vs Defects)", f"{corr_sc:.2f}")
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    sns.regplot(
        x="Shipping costs", y="Defect rates",
        data=df, scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"}, ax=ax4
    )
    ax4.set_title("Shipping Costs vs Defect Rates")
    plt.tight_layout()
    st.pyplot(fig4)

    st.divider()

    # Q5 — Product type
    st.subheader("Q5 · Are certain product categories more defect-prone?")
    product_defect = (
        df.groupby("Product type")["Defect rates"]
        .mean()
        .reset_index()
        .sort_values("Defect rates", ascending=False)
    )
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="Product type", y="Defect rates",
        data=product_defect,
        hue="Product type", legend=False,
        palette="coolwarm", ax=ax5
    )
    ax5.set_title("Average Defect Rate by Product Type")
    plt.tight_layout()
    st.pyplot(fig5)

    st.divider()

    # Correlation heatmap & raw data
    st.subheader("Correlation Heatmap")
    fig6, ax6 = plt.subplots(figsize=(11, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax6)
    plt.tight_layout()
    st.pyplot(fig6)

    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Model Performance Comparison")

    # Accuracy summary table
    acc_data = {
        name: round(res["accuracy"] * 100, 1)
        for name, res in results.items()
    }
    acc_df = pd.DataFrame(acc_data.items(), columns=["Model", "Accuracy (%)"])
    best_model = acc_df.loc[acc_df["Accuracy (%)"].idxmax(), "Model"]
    st.dataframe(acc_df.set_index("Model"), use_container_width=True)
    st.success(f"🏆 Best accuracy: **{best_model}** ({acc_data[best_model]}%)")

    st.divider()

    # Per-model details
    for name, res in results.items():
        with st.expander(f"{name}  —  Accuracy: {res['accuracy']*100:.1f}%"):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                report_df = pd.DataFrame(res["report"]).transpose().round(2)
                st.dataframe(report_df, use_container_width=True)
            with col_b:
                st.pyplot(plot_confusion_matrix(res["conf_matrix"], name))


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Feature Importance (Random Forest)")
    st.markdown(
        "Shows which factors most influence whether a product batch is predicted "
        "as **high defect risk**."
    )

    fig7, ax7 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance", y="Feature",
        data=importance_df,
        hue="Feature", legend=False,
        palette="magma", ax=ax7
    )
    ax7.set_title("Feature Importances for Defect Risk Prediction")
    ax7.set_xlabel("Relative Importance")
    plt.tight_layout()
    st.pyplot(fig7)

    st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — PREDICT
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🎯 Predict Defect Risk for a New Batch")

    # Retrieve unique values from dataset for dropdowns
    supplier_options       = sorted(df["Supplier name"].unique().tolist())
    transport_options      = sorted(df["Transportation modes"].unique().tolist())
    route_options          = sorted(df["Routes"].unique().tolist())
    product_type_options   = sorted(df["Product type"].unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        supplier      = st.selectbox("Supplier Name",        supplier_options)
        transport     = st.selectbox("Transportation Mode",  transport_options)
        route         = st.selectbox("Route",                route_options)
        product_type  = st.selectbox("Product Type",         product_type_options)

    with col2:
        costs      = st.number_input("Costs ($)",      min_value=0.0, value=500.0,  step=10.0)
        lead_time  = st.number_input("Lead Time (days)", min_value=0.0, value=10.0, step=1.0)
        model_choice = st.selectbox("Model to use", list(results.keys()))

    st.divider()

    if st.button("⚡ Predict Defect Risk", type="primary"):
        # Encode categorical inputs using the fitted encoders
        def safe_encode(encoder, value):
            classes = list(encoder.classes_)
            if value in classes:
                return encoder.transform([value])[0]
            return 0  # fallback for unseen labels

        input_data = pd.DataFrame([{
            "Supplier name":        safe_encode(encoders["Supplier name"], supplier),
            "Transportation modes": safe_encode(encoders["Transportation modes"], transport),
            "Costs":                costs,
            "Lead time":            lead_time,
            "Routes":               safe_encode(encoders["Routes"], route),
            "Product type":         safe_encode(encoders["Product type"], product_type),
        }])

        chosen_model = results[model_choice]["model"]
        prediction   = chosen_model.predict(input_data)[0]
        proba        = None

        if hasattr(chosen_model, "predict_proba"):
            proba = chosen_model.predict_proba(input_data)[0]

        if prediction == 1:
            st.error("⚠️ **High Defect Risk Predicted**")
            st.write("**Recommendation:** Review supplier quality and production process. "
                     "Consider switching to air transport or auditing the selected route.")
        else:
            st.success("✅ **Low Defect Risk Predicted**")
            st.write("Production batch appears safe. Continue with standard QC procedures.")

        if proba is not None:
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Probability — Low Risk",  f"{proba[0]*100:.1f}%")
            p_col2.metric("Probability — High Risk", f"{proba[1]*100:.1f}%")

        st.caption(f"Model used: **{model_choice}** | Mean defect threshold: {mean_defect:.2f}%")
