import streamlit as st
import requests
import time

st.set_page_config(page_title="Mobile Money Transfer", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-image: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }

    /* Title styling */
    h1 {
        font-family: 'Arial Black', Gadget, sans-serif;
        color: #f7b733; /* Gold color for title */
        text-align: center;
    }

    /* Header styling */
    h2 {
        font-family: 'Arial', sans-serif;
        color: #fc4a1a; /* A vibrant orange for headers */
        border-bottom: 2px solid #fc4a1a;
        padding-bottom: 5px;
    }

    /* Custom card for user info */
    .user-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    }
    
    .user-card .balance {
        font-size: 1.5rem;
        font-weight: bold;
        color: #42f59e; /* Bright green for balance */
    }
    
    .user-card .email {
        font-size: 0.9rem;
        color: #cccccc;
        font-style: italic;
    }
    
    /* Style for the submit button */
    div.stButton > button:first-child {
        background-image: linear-gradient(to right, #fc4a1a 0%, #f7b733 51%, #fc4a1a 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 15px 45px;
        text-transform: uppercase;
        font-size: 1rem;
        transition: 0.5s;
        background-size: 200% auto;
        box-shadow: 0 0 20px #eee;
    }
    
    div.stButton > button:hover {
        background-position: right center; /* change the direction of the change here */
        color: #fff;
        text-decoration: none;
    }

</style>
""", unsafe_allow_html=True)


# --- API Configuration ---
API_URL = "http://127.0.0.1:5000"

# --- Data Fetching Function ---
@st.cache_data(ttl=10)
def get_users():
    try:
        response = requests.get(f"{API_URL}/get_users", timeout=5)
        return response.json() if response.status_code == 200 else []
    except requests.RequestException: return []

# --- Main App ---
st.title("💸 Modern Money Transfer")

users = get_users()

if not users:
    st.error("Could not connect to the API to fetch users. Is `api.py` running?")
else:
    user_dict = {u['user_name']: u for u in users}
    user_names = sorted(user_dict.keys())

    st.header("Create a New Transaction")

    col1, col2 = st.columns(2)
    with col1:
        sender_name = st.selectbox("Select Sender", user_names, index=0, key="sender_select")
        sender = user_dict[sender_name]
        sender_balance = float(sender['current_balance'])
        sender_email = sender.get('email', 'N/A')
        
        st.markdown(f"""
        <div class="user-card">
            <h4>Sender: {sender_name}</h4>
            <p class="balance">${sender_balance:,.2f}</p>
            <p class="email">{sender_email}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        available_receivers = [name for name in user_names if name != sender_name]
        receiver_name = st.selectbox("Select Receiver", available_receivers, index=min(1, len(available_receivers)-1), key="receiver_select")
        receiver = user_dict[receiver_name]
        receiver_balance = float(receiver['current_balance'])
        receiver_email = receiver.get('email', 'N/A')
        
        st.markdown(f"""
        <div class="user-card">
            <h4>Receiver: {receiver_name}</h4>
            <p class="balance">${receiver_balance:,.2f}</p>
            <p class="email">{receiver_email}</p>
        </div>
        """, unsafe_allow_html=True)

    amount = st.number_input("Amount to Transfer", min_value=0.01, value=100.00, step=50.0, format="%.2f")

    if st.button("Send Money", use_container_width=True):
        if amount > sender_balance:
            st.error("Transfer amount cannot exceed sender's balance.")
        else:
            payload = {"sender_id": sender['user_id'], "receiver_id": receiver['user_id'], "amount": amount}
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
