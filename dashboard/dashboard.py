import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import matplotlib.pyplot as plt

# Database connection details - update accordingly
conn_params = {
    'dbname': 'fraud_db',
    'user': 'postgres',
    'password': 'vil100sr',
    'host': 'localhost',
    'port': '5432'
}

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
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def safe_json_normalize(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            x = {}
    return pd.json_normalize(x)

def main():
    st.title("Fraud Detection Dashboard")

    if st.button("Refresh Data"):
        st.cache_data.clear()

    df = load_transactions()
    if df.empty:
        st.write("No transactions to display.")
        return

    # Normalize transaction_data JSON column
    df['transaction_data'] = df['transaction_data'].apply(safe_json_normalize)
    df_expanded = pd.json_normalize(df['transaction_data'])
    df = pd.concat([df.drop('transaction_data', axis=1), df_expanded], axis=1)

    # Filters
    user_filter = st.multiselect("Filter by User ID:", options=df['user_id'].unique(), default=None)
    fraud_filter = st.multiselect("Filter by Fraud Status (Final Prediction):", options=[0, 1], format_func=lambda x: "No Fraud" if x == 0 else "Fraud", default=[0, 1])

    if user_filter:
        df = df[df['user_id'].isin(user_filter)]
    if fraud_filter:
        df = df[df['final_pred'].isin(fraud_filter)]

    # Show latest transactions
    st.subheader(f"Transactions (Showing {len(df)})")
    st.dataframe(df)

    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Transactions Displayed", len(df))
    with col2:
        st.metric("Flagged as Fraud", int(df['final_pred'].sum()))

    # Pie chart of fraud vs non-fraud
    st.subheader("Fraud Distribution")
    fraud_counts = df['final_pred'].value_counts().sort_index()
    pie_labels = ['Non-Fraud', 'Fraud']
    colors = ['#2ca02c', '#d62728']
    fig1, ax1 = plt.subplots()
    ax1.pie(
        fraud_counts.reindex([0, 1]).fillna(0),
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'black'})
    ax1.axis('equal')
    st.pyplot(fig1)

    # Bar chart: fraud count by transaction type
    st.subheader("Fraud by Transaction Type")
    if 'type' in df.columns:
        fraud_by_type = df[df['final_pred'] == 1]['type'].value_counts()
        fig2, ax2 = plt.subplots()
        fraud_by_type.plot(kind='bar', color='crimson', ax=ax2)
        ax2.set_ylabel("Count of Fraudulent Transactions")
        ax2.set_xlabel("Transaction Type")
        st.pyplot(fig2)
    else:
        st.info("Transaction type data not available.")

if __name__ == "__main__":
    main()
