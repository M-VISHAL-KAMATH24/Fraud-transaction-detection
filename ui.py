import streamlit as st
import requests
import time

st.set_page_config(page_title="Mobile Money Transfer", layout="centered")

# --- API Configuration ---
API_URL = "http://127.0.0.1:5000"

# --- Data Fetching Function ---
@st.cache_data(ttl=10) # Cache user data for 10 seconds
def get_users():
    """Fetches the list of users and their balances from the API."""
    try:
        response = requests.get(f"{API_URL}/get_users", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return []
    return []

# --- Main App ---
st.title("💸 Mobile Money Transfer")

users = get_users()

if not users:
    st.error("Could not connect to the API to fetch users. Is `api.py` running?")
    st.stop()
else:
    user_dict = {u['user_name']: u for u in users}
    user_names = sorted(user_dict.keys())

    st.header("Create a New Transaction")

    # --- User and Amount Selection ---
    col1, col2 = st.columns(2)
    with col1:
        sender_name = st.selectbox("Select Sender", user_names, index=0, key="sender_select")
        # Get the full sender object
        sender = user_dict[sender_name]
        sender_balance = float(sender['current_balance'])
        # Get the sender's email
        sender_email = sender.get('email', 'N/A') # Use .get() for safety
        
        st.info(f"Current Balance: **${sender_balance:,.2f}**")
        # Display the email
        st.caption(f"Email: {sender_email}")


    with col2:
        available_receivers = [name for name in user_names if name != sender_name]
        receiver_name = st.selectbox("Select Receiver", available_receivers, index=min(1, len(available_receivers)-1), key="receiver_select")
        # Get the full receiver object
        receiver = user_dict[receiver_name]
        receiver_balance = float(receiver['current_balance'])
        # Get the receiver's email
        receiver_email = receiver.get('email', 'N/A')
        
        st.info(f"Current Balance: **${receiver_balance:,.2f}**")
        # Display the email
        st.caption(f"Email: {receiver_email}")

    amount = st.number_input("Amount to Transfer", min_value=0.01, value=100.00, step=50.0, format="%.2f")

    # --- Submission Logic ---
    if st.button("Send Money", use_container_width=True):
        if amount > sender_balance:
            st.error("Transfer amount cannot exceed sender's balance.")
        else:
            payload = {
                "sender_id": sender['user_id'],
                "receiver_id": receiver['user_id'],
                "amount": amount
            }
            
            with st.spinner("Submitting transaction for fraud analysis..."):
                submit_response = requests.post(f"{API_URL}/submit_transaction", json=payload)

                if submit_response.status_code != 202:
                    st.error(f"Failed to submit: {submit_response.json().get('error', 'Unknown API error')}")
                else:
                    transaction_id = submit_response.json()['transaction_id']
                    st.info(f"Transaction submitted (ID: {transaction_id}). Waiting for result...")

                    for _ in range(20):
                        time.sleep(1)
                        try:
                            poll_response = requests.get(f"{API_URL}/get_prediction/{transaction_id}")
                            if poll_response.status_code == 200 and poll_response.json()['status'] == 'completed':
                                result = poll_response.json()
                                if result['is_fraud']:
                                    st.error("🔴 **Transaction Blocked!** This activity was flagged as potentially fraudulent.", icon="🚨")
                                else:
                                    st.success("🟢 **Transaction Approved and Completed!** Balances updated.", icon="✅")
                                    st.balloons()
                                
                                time.sleep(2)
                                st.rerun()
                                break
                        except requests.RequestException:
                            st.error("Lost connection to API while polling for result.")
                            break
                    else:
                        st.warning("Prediction timed out. The system may be busy. Please check the consumer logs.")
