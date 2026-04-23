import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Supply Chain Defect Dashboard",
    page_icon="🏭",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# EXACT COLUMN NAMES FROM YOUR EXCEL FILE
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = [
    "Product type", "Supplier name", "Transportation modes",
    "Routes", "Costs", "Lead time", "Defect rates",
    "Manufacturing lead time", "Shipping costs",
]

MODEL_FEATURES   = ["Supplier name", "Transportation modes", "Costs",
                    "Lead time", "Routes", "Product type"]
CAT_FEATURES     = ["Supplier name", "Transportation modes", "Routes", "Product type"]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    # Force numeric columns
    for col in ["Defect rates", "Shipping costs", "Manufacturing lead time",
                "Costs", "Lead time", "Manufacturing costs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Defect rates"], inplace=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(_df):
    supply_df = _df.copy()
    mean_defect = supply_df["Defect rates"].mean()
    supply_df["High_Defect"] = (supply_df["Defect rates"] > mean_defect).astype(int)

    X = supply_df[MODEL_FEATURES].copy()
    y = supply_df["High_Defect"]

    encoders = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, solver="liblinear", max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        results[name] = {
            "model":       mdl,
            "accuracy":    accuracy_score(y_test, y_pred),
            "report":      classification_report(y_test, y_pred, output_dict=True),
            "conf_matrix": confusion_matrix(y_test, y_pred),
        }

    rf = results["Random Forest"]["model"]
    importance_df = pd.DataFrame({
        "Feature":    X_train.columns,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return results, encoders, importance_df, float(mean_defect)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════
CMAP_MAP = {
    "Random Forest":       "Blues",
    "Logistic Regression": "Greens",
    "Decision Tree":       "Purples",
}

def make_cm_fig(cm, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=CMAP_MAP.get(name, "Blues"),
                xticklabels=["Low Risk", "High Risk"],
                yticklabels=["Low Risk", "High Risk"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    return fig

def safe_encode(encoder, value):
    classes = list(encoder.classes_)
    return int(encoder.transform([value])[0]) if value in classes else 0

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏭 Supply Chain Defect Prediction Dashboard")
st.markdown("Upload your **cleaned_supply_chain_data.xlsx** to start.")

uploaded_file = st.file_uploader("📂 Upload Excel file", type=["xlsx"])

if uploaded_file is None:
    st.info("👆 Please upload the Excel file to get started.")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

# ── Validate columns ───────────────────────────────────────────────────────
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    st.error(f"❌ Missing columns in your Excel file: `{missing_cols}`")
    st.write("**Columns found in your file:**")
    st.write(list(df.columns))
    st.stop()

# ── Train models ───────────────────────────────────────────────────────────
with st.spinner("Training models..."):
    try:
        results, encoders, importance_df, mean_defect = train_models(df)
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        st.stop()

st.success(f"✅ Loaded {len(df):,} rows — models trained successfully!")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Insights",
    "🤖 Model Performance",
    "🔍 Feature Importance",
    "🎯 Predict Defect Risk",
])

# ─────────────────────────────────────────
# TAB 1 ── EDA
# ─────────────────────────────────────────
with tab1:
    st.header("Exploratory Data Analysis")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records",       f"{len(df):,}")
    k2.metric("Avg Defect Rate",      f"{df['Defect rates'].mean():.2f}%")
    k3.metric("Unique Suppliers",     df["Supplier name"].nunique())
    k4.metric("Unique Product Types", df["Product type"].nunique())

    st.divider()

    # Q1 — Supplier defect rates
    st.subheader("Q1 · Which suppliers have the highest defect rates?")
    sup_def = (df.groupby("Supplier name")["Defect rates"]
               .mean().reset_index()
               .sort_values("Defect rates", ascending=False))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Supplier name", y="Defect rates",
                data=sup_def.head(5), hue="Supplier name",
                legend=False, palette="viridis", ax=ax)
    ax.set_title("Top 5 Suppliers by Average Defect Rate")
    ax.set_xlabel("Supplier Name"); ax.set_ylabel("Avg Defect Rate (%)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    st.pyplot(fig); plt.close(fig)
    st.dataframe(sup_def, use_container_width=True)

    st.divider()

    # Q2 — Lead time vs defects
    st.subheader("Q2 · Does manufacturing lead time affect defect rates?")
    corr_lt = df["Manufacturing lead time"].corr(df["Defect rates"])
    st.metric("Correlation", f"{corr_lt:.2f}",
              help="Close to 0 = no relationship")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(x="Manufacturing lead time", y="Defect rates",
                data=df, scatter_kws={"alpha": 0.4},
                line_kws={"color": "red"}, ax=ax)
    ax.set_title("Manufacturing Lead Time vs Defect Rates")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.divider()

    # Q3 — Transport mode
    st.subheader("Q3 · Does transportation mode affect defect risk?")
    tran_def = (df.groupby("Transportation modes")["Defect rates"]
                .mean().reset_index()
                .sort_values("Defect rates", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Transportation modes", y="Defect rates",
                data=tran_def, hue="Transportation modes",
                legend=False, palette="coolwarm", ax=ax)
    ax.set_title("Average Defect Rate by Transportation Mode")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(tran_def, use_container_width=True)

    st.divider()

    # Q4 — Shipping cost
    st.subheader("Q4 · Does shipping cost influence defect rates?")
    corr_sc = df["Shipping costs"].corr(df["Defect rates"])
    st.metric("Correlation", f"{corr_sc:.2f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(x="Shipping costs", y="Defect rates",
                data=df, scatter_kws={"alpha": 0.4},
                line_kws={"color": "red"}, ax=ax)
    ax.set_title("Shipping Costs vs Defect Rates")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.divider()

    # Q5 — Product type
    st.subheader("Q5 · Are certain product types more defect-prone?")
    prod_def = (df.groupby("Product type")["Defect rates"]
                .mean().reset_index()
                .sort_values("Defect rates", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Product type", y="Defect rates",
                data=prod_def, hue="Product type",
                legend=False, palette="coolwarm", ax=ax)
    ax.set_title("Average Defect Rate by Product Type")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(prod_def, use_container_width=True)

    st.divider()

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True,
                fmt=".2f", cmap="coolwarm", ax=ax)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# ─────────────────────────────────────────
# TAB 2 ── MODEL PERFORMANCE
# ─────────────────────────────────────────
with tab2:
    st.header("Model Performance Comparison")

    acc_rows   = [(n, round(r["accuracy"]*100, 1)) for n, r in results.items()]
    acc_df     = pd.DataFrame(acc_rows, columns=["Model", "Accuracy (%)"])
    best_model = acc_df.loc[acc_df["Accuracy (%)"].idxmax(), "Model"]
    st.dataframe(acc_df.set_index("Model"), use_container_width=True)
    st.success(f"🏆 Best: **{best_model}** ({acc_df['Accuracy (%)'].max()}%)")

    st.divider()

    for name, res in results.items():
        with st.expander(f"{name}  —  Accuracy: {res['accuracy']*100:.1f}%"):
            ca, cb = st.columns(2)
            with ca:
                rpt = pd.DataFrame(res["report"]).transpose().round(2)
                st.dataframe(rpt, use_container_width=True)
            with cb:
                fig_cm = make_cm_fig(res["conf_matrix"], name)
                st.pyplot(fig_cm); plt.close(fig_cm)

# ─────────────────────────────────────────
# TAB 3 ── FEATURE IMPORTANCE
# ─────────────────────────────────────────
with tab3:
    st.header("Feature Importance (Random Forest)")
    st.markdown("Factors that most influence **High Defect Risk** prediction.")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature",
                data=importance_df, hue="Feature",
                legend=False, palette="magma", ax=ax)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Relative Importance")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.dataframe(importance_df, use_container_width=True)

# ─────────────────────────────────────────
# TAB 4 ── PREDICT
# ─────────────────────────────────────────
with tab4:
    st.header("🎯 Predict Defect Risk for a New Batch")

    col1, col2 = st.columns(2)

    with col1:
        supplier     = st.selectbox("Supplier Name",       sorted(df["Supplier name"].unique()))
        transport    = st.selectbox("Transportation Mode", sorted(df["Transportation modes"].unique()))
        route        = st.selectbox("Route",               sorted(df["Routes"].unique()))
        product_type = st.selectbox("Product Type",        sorted(df["Product type"].unique()))

    with col2:
        costs        = st.number_input("Costs ($)",        min_value=0.0,
                                       value=float(df["Costs"].median()), step=10.0)
        lead_time    = st.number_input("Lead Time (days)", min_value=0.0,
                                       value=float(df["Lead time"].median()), step=1.0)
        model_choice = st.selectbox("Model",               list(results.keys()))

    st.divider()

    if st.button("⚡ Predict Defect Risk", type="primary"):
        input_data = pd.DataFrame([{
            "Supplier name":        safe_encode(encoders["Supplier name"],        supplier),
            "Transportation modes": safe_encode(encoders["Transportation modes"], transport),
            "Costs":                costs,
            "Lead time":            lead_time,
            "Routes":               safe_encode(encoders["Routes"],               route),
            "Product type":         safe_encode(encoders["Product type"],         product_type),
        }])

        chosen = results[model_choice]["model"]
        pred   = chosen.predict(input_data)[0]

        st.divider()
        if pred == 1:
            st.error("⚠️ **High Defect Risk Predicted**")
            st.markdown(
                "**Recommendations:**\n"
                "- Audit supplier quality processes\n"
                "- Consider switching to Air transport\n"
                "- Review the selected route for risk factors"
            )
        else:
            st.success("✅ **Low Defect Risk Predicted**")
            st.markdown("Production batch appears safe. Continue standard QC procedures.")

        if hasattr(chosen, "predict_proba"):
            proba = chosen.predict_proba(input_data)[0]
            p1, p2 = st.columns(2)
            p1.metric("🟢 Prob — Low Risk",  f"{proba[0]*100:.1f}%")
            p2.metric("🔴 Prob — High Risk", f"{proba[1]*100:.1f}%")

        st.caption(f"Model: **{model_choice}** | Defect threshold: {mean_defect:.2f}%")
