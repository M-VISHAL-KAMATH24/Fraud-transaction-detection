import streamlit as st
import requests
import time
import json
import random

st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")

st.title("💳 Real-Time Transaction Fraud Detection")
st.write("Submit a transaction to see if it's flagged as fraudulent by our live AI model.")

# --- API Configuration ---
API_URL = "http://127.0.0.1:5000"

# --- Form for User Input ---
with st.form("transaction_form"):
    st.header("Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.text_input("User ID", value=f"user_{random.randint(1, 15)}")
        amount = st.number_input("Amount", min_value=0.01, value=150.75, step=10.0, format="%.2f")
        transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])

    with col2:
        old_balance_org = st.number_input("Sender's Old Balance", value=10000.0)
        new_balance_orig = st.number_input("Sender's New Balance", value=9849.25)
        old_balance_dest = st.number_input("Receiver's Old Balance", value=2000.0)
        new_balance_dest = st.number_input("Receiver's New Balance", value=2150.75)
    
    # The submit button is now correctly placed as the main action for the form.
    submitted = st.form_submit_button("Submit for Fraud Check")

# --- Form Submission Logic ---
if submitted:
    # Here we build the full data payload, including the "hidden" values
    transaction_data = {
        "user_id": user_id,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        
        # These values are now added directly, not as UI elements
        "step": 1,
        "isFlaggedFraud": 0,
        "tx_lat": 40.7128, # Default to NYC for demo
        "tx_long": -74.0060, # Default to NYC for demo
    }

    st.subheader("Processing...")
    
    with st.spinner("1. Submitting transaction to the processing queue..."):
        try:
            submit_response = requests.post(f"{API_URL}/submit_transaction", json=transaction_data, timeout=5)
            if submit_response.status_code == 202:
                transaction_id = submit_response.json().get("transaction_id")
                st.success(f"Transaction received! Processing ID: `{transaction_id}`")
            else:
                st.error(f"API Error: {submit_response.status_code} - {submit_response.text}")
                st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API. Is `api.py` running? Error: {e}")
            st.stop()

    with st.spinner(f"2. Waiting for AI model to analyze transaction `{transaction_id}`..."):
        prediction_result = None
        for i in range(15): # Poll for up to 15 seconds
            try:
                poll_response = requests.get(f"{API_URL}/get_prediction/{transaction_id}", timeout=2)
                if poll_response.status_code == 200 and poll_response.json().get("status") == "completed":
                    prediction_result = poll_response.json()
                    break
                time.sleep(1) # Wait 1 second before polling again
            except requests.exceptions.RequestException:
                st.error("API connection lost while waiting for result.")
                st.stop()
        
    st.subheader("Result")

    if prediction_result:
        is_fraud = prediction_result.get("is_fraud")
        details = prediction_result.get("details", {})
        
        if is_fraud:
            st.error("🔴 **FRAUD DETECTED!** This transaction has been flagged and blocked.", icon="🚨")
        else:
            st.success("🟢 **Transaction Approved.** This transaction appears to be legitimate.", icon="✅")
        
        st.write("Prediction Details:")
        st.json(details)
    else:
        st.warning("Prediction timed out. The system is likely under high load. Please check the consumer logs.", icon="⏳")

