# 🏭 Supply Chain Defect Prediction Dashboard

An interactive Streamlit app that predicts product defect risk using machine learning on supply chain data.

## 🚀 Live App
👉 [Click here to open the app](https://share.streamlit.io)  
*(Replace this link with your actual Streamlit Cloud URL after deploying)*

---

## 📋 Features

- **EDA & Insights** — Visual analysis of defect rates by supplier, transport mode, product type, and more
- **Model Performance** — Compare Random Forest, Logistic Regression, Decision Tree, and XGBoost
- **Feature Importance** — Understand what factors drive defect risk most
- **Predict Defect Risk** — Enter batch details and get an instant High/Low risk prediction

---

## 🗂️ Project Structure

```
supply_chain_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files to exclude from Git
└── README.md                       # This file
```

> ⚠️ The Excel file (`cleaned_supply_chain_data.xlsx`) is uploaded directly in the app — it is **not** stored in this repo.

---

## 🛠️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/supply-chain-app.git
cd supply-chain-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → select this repo → set main file to `app.py`
4. Click **Deploy**

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | 1.32.0 |
| pandas | 2.2.1 |
| numpy | 1.26.4 |
| matplotlib | 3.8.3 |
| seaborn | 0.13.2 |
| scikit-learn | 1.4.1 |
| openpyxl | 3.1.2 |
| xgboost | 2.0.3 |
