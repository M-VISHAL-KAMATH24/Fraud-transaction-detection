import streamlit as st
import requests
import time

st.set_page_config(page_title="Mobile Money Transfer", layout="wide")

# --- Custom CSS (unchanged) ---
st.markdown("""
<style>
    .stApp { background-image: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%); color: white; }
    h1 { font-family: 'Arial Black', Gadget, sans-serif; color: #f7b733; text-align: center; }
    h2 { font-family: 'Arial', sans-serif; color: #fc4a1a; border-bottom: 2px solid #fc4a1a; padding-bottom: 5px; }
    .user-card { background-color: rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 1rem; margin-bottom: 1rem; border: 1px solid rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1); }
    .user-card .balance { font-size: 1.5rem; font-weight: bold; color: #42f59e; }
    .user-card .email { font-size: 0.9rem; color: #cccccc; font-style: italic; }
    div.stButton > button:first-child { background-image: linear-gradient(to right, #fc4a1a 0%, #f7b733 51%, #fc4a1a 100%); color: white; font-weight: bold; border-radius: 10px; border: none; padding: 15px 45px; text-transform: uppercase; font-size: 1rem; transition: 0.5s; background-size: 200% auto; box-shadow: 0 0 20px #eee; }
    div.stButton > button:hover { background-position: right center; color: #fff; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# --- API and Data Fetching (unchanged) ---
API_URL = "http://127.0.0.1:5000"
@st.cache_data(ttl=10)
def get_users():
    try:
        response = requests.get(f"{API_URL}/get_users", timeout=5)
        return response.json() if response.ok else []
    except: return []

# --- Main App ---
st.title("💸 Secure Money Transfer")

# Initialize session state for OTP
if 'otp_verified' not in st.session_state:
    st.session_state.otp_verified = False
if 'transaction_payload' not in st.session_state:
    st.session_state.transaction_payload = None

users = get_users()
if not users:
    st.error("Could not connect to the API.")
else:
    user_dict = {u['user_name']: u for u in users}
    user_names = sorted(user_dict.keys())

    st.header("Create a New Transaction")

    col1, col2 = st.columns(2)
    with col1:
        sender_name = st.selectbox("Sender", user_names, key="sender")
        sender = user_dict[sender_name]
        sender_balance = float(sender['current_balance'])
        sender_email = sender.get('email', 'N/A')
        st.markdown(f'<div class="user-card"><h4>Sender: {sender_name}</h4><p class="balance">${sender_balance:,.2f}</p><p class="email">{sender_email}</p></div>', unsafe_allow_html=True)
    with col2:
        receivers = [n for n in user_names if n != sender_name]
        receiver_name = st.selectbox("Receiver", receivers, index=min(1, len(receivers)-1))
        receiver = user_dict[receiver_name]
        receiver_balance = float(receiver['current_balance'])
        receiver_email = receiver.get('email', 'N/A')
        st.markdown(f'<div class="user-card"><h4>Receiver: {receiver_name}</h4><p class="balance">${receiver_balance:,.2f}</p><p class="email">{receiver_email}</p></div>', unsafe_allow_html=True)

    amount = st.number_input("Amount to Transfer", min_value=0.01, step=50.0, format="%.2f")
    drain_ratio = amount / sender_balance if sender_balance > 0 else 0

    # --- NEW OTP LOGIC ---
    otp_placeholder = st.empty()
    
    # Scenario 1: High-risk, requires OTP
    if 0.7 <= drain_ratio < 1.0:
        if not st.session_state.otp_verified:
            with otp_placeholder.container():
                st.warning("⚠️ **High-Risk Transaction Detected!** Please verify with an OTP to proceed.", icon="❗")
                otp_input = st.text_input("Enter OTP (for testing, use 5555)", type="password")
                if st.button("Verify OTP"):
                    if otp_input == "5555":
                        st.session_state.otp_verified = True
                        st.success("OTP Verified! You can now send the money.")
                        time.sleep(1)
                        st.rerun() # Rerun to hide the OTP box
                    else:
                        st.error("Invalid OTP.")
    # If OTP was just verified, show a success message
    elif st.session_state.otp_verified:
         with otp_placeholder.container():
            st.success("OTP Verified! You can now send the money.")


    # Scenario 2: Instant block
    if drain_ratio >= 1.0:
        st.error("❌ **Transaction Blocked!** Attempting to transfer the full account balance is not permitted.", icon="🚫")
        st.stop()


    # Determine if the main button should be disabled
    send_button_disabled = (0.7 <= drain_ratio < 1.0) and not st.session_state.otp_verified

    if st.button("Send Money", use_container_width=True, disabled=send_button_disabled):
        # Prepare payload and send
        payload = {"sender_id": sender['user_id'], "receiver_id": receiver['user_id'], "amount": amount}
        with st.spinner("Submitting transaction..."):
            resp = requests.post(f"{API_URL}/submit_transaction", json=payload)
            if not resp.ok:
                st.error(f"API Error: {resp.text}")
            else:
                tx_id = resp.json()['transaction_id']
                st.info(f"Transaction submitted (ID: {tx_id}). Awaiting final confirmation...")
                # Poll for result...
                for _ in range(20):
                    time.sleep(1)
                    poll_resp = requests.get(f"{API_URL}/get_prediction/{tx_id}")
                    if poll_resp.ok and poll_resp.json().get('status') == 'completed':
                        res = poll_resp.json()
                        if res['is_fraud']:
                            st.error("🔴 **Transaction Blocked by Backend!** This was flagged as fraudulent.", icon="🚨")
                        else:
                            st.success("🟢 **Transaction Approved & Completed!**", icon="✅")
                            st.balloons()
                        # Reset OTP state and rerun
                        st.session_state.otp_verified = False
                        time.sleep(3)
                        st.rerun()
                        break
                else:
                    st.warning("Prediction timed out.")
