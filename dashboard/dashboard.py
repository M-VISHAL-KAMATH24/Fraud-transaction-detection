import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import matplotlib.pyplot as plt

# -----------------------------
# Database connection details
# -----------------------------
conn_params = {
    'dbname': 'fraud_db',
    'user': 'postgres',
    'password': 'vil100sr',
    'host': 'localhost',
    'port': '5432'
}

# -----------------------------
# Cache data
# -----------------------------
@st.cache_data(ttl=30)
def load_transactions():
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_id, transaction_data, offline_pred, online_pred, geo_flag, final_pred
                    FROM transactions
                    ORDER BY id DESC LIMIT 200;
                """)
                rows = cur.fetchall()
                return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# -----------------------------
# Safe JSON normalize
# -----------------------------
def safe_json_normalize(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            x = {}
    return pd.json_normalize(x)

# -----------------------------
# App layout
# -----------------------------
def main():
    # ---- Custom Page Config ----
    st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

    # ---- Custom CSS Styling ----
    st.markdown("""
        <style>
            body {background-color: #f9fafb;}
            .main {background-color: #ffffff; padding: 20px; border-radius: 12px;}
            h1, h2, h3 {color: #333333;}
            /* Style for metric boxes */
            div[data-testid="stMetricValue"] {
                background-color: #eef2f7;
                color: #000000;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("💳 Fraud Detection Dashboard")
    st.caption("Monitor, analyze, and detect fraud in real-time.")

    # ---- Refresh Button ----
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

    # ---- Load Data ----
    df = load_transactions()
    if df.empty:
        st.warning("⚠️ No transactions to display.")
        return

    # ---- Normalize JSON ----
    df['transaction_data'] = df['transaction_data'].apply(safe_json_normalize)
    df_expanded = pd.json_normalize(df['transaction_data'])
    df = pd.concat([df.drop('transaction_data', axis=1), df_expanded], axis=1)

    # ---- Filters ----
    with st.container():
        st.subheader("🔍 Filters")
        col1, col2 = st.columns(2)
        with col1:
            user_filter = st.multiselect("Filter by User ID:", options=df['user_id'].unique(), default=None)
        with col2:
            fraud_filter = st.multiselect(
                "Filter by Fraud Status (Final Prediction):",
                options=[0, 1],
                format_func=lambda x: "No Fraud" if x == 0 else "Fraud",
                default=[0, 1]
            )

    if user_filter:
        df = df[df['user_id'].isin(user_filter)]
    if fraud_filter:
        df = df[df['final_pred'].isin(fraud_filter)]

    # ---- Transactions Table ----
    st.subheader(f"📋 Transactions (Showing {len(df)})")
    st.dataframe(df, use_container_width=True)

    # ---- Summary Metrics ----
    st.subheader("📊 Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Transactions Displayed", len(df))
    with col2:
        st.metric("Flagged as Fraud", int(df['final_pred'].sum()))

    # ---- Fraud Distribution ----
    st.subheader("📈 Fraud Distribution")
    fraud_counts = df['final_pred'].value_counts().sort_index()
    pie_labels = ['Non-Fraud', 'Fraud']
    colors = ['#2ca02c', '#d62728']
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))  # Slightly smaller chart
    ax1.pie(
        fraud_counts.reindex([0, 1]).fillna(0),
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white'}
    )
    ax1.axis('equal')
    st.pyplot(fig1)

    # ---- Fraud by Transaction Type ----
    st.subheader("💼 Fraud by Transaction Type")
    if 'type' in df.columns:
        fraud_by_type = df[df['final_pred'] == 1]['type'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3))  # Reduced size
        fraud_by_type.plot(kind='bar', color='crimson', ax=ax2, edgecolor='black')
        ax2.set_ylabel("Count of Fraudulent Transactions")
        ax2.set_xlabel("Transaction Type")
        ax2.set_title("Fraudulent Transactions by Type", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("ℹ️ Transaction type data not available.")

if __name__ == "__main__":
    main()
