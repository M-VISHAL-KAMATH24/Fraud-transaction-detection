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


# --- API and Data Fetching ---
API_URL = "http://127.0.0.1:5000"
@st.cache_data(ttl=10)
def get_users():
    try:
        response = requests.get(f"{API_URL}/get_users", timeout=5)
        return response.json() if response.ok else []
    except: return []

# --- Helper to reset OTP state ---
def reset_otp_flow():
    st.session_state.otp_requested = False
    st.session_state.otp_verified = False

# --- Main App ---
st.title("💸 Secure Money Transfer")

# Initialize session state for the multi-step OTP process
if 'otp_requested' not in st.session_state:
    st.session_state.otp_requested = False
if 'otp_verified' not in st.session_state:
    st.session_state.otp_verified = False

users = get_users()
if not users:
    st.error("Could not connect to the API.")
else:
    user_dict = {u['user_name']: u for u in users}
    user_names = sorted(user_dict.keys())

    st.header("Create a New Transaction")

    col1, col2 = st.columns(2)
    with col1:
        sender_name = st.selectbox("Sender", user_names, key="sender", on_change=reset_otp_flow)
        sender = user_dict[sender_name]
        sender_balance = float(sender['current_balance'])
        sender_email = sender.get('email', 'N/A')
        st.markdown(f'<div class="user-card"><h4>Sender: {sender_name}</h4><p class="balance">${sender_balance:,.2f}</p><p class="email">{sender_email}</p></div>', unsafe_allow_html=True)
    with col2:
        receivers = [n for n in user_names if n != sender_name]
        receiver_name = st.selectbox("Receiver", receivers, index=min(1, len(receivers)-1), on_change=reset_otp_flow)
        receiver = user_dict[receiver_name]
        st.markdown(f'<div class="user-card"><h4>Receiver: {receiver_name}</h4><p class="balance">${float(receiver["current_balance"]):,.2f}</p><p class="email">{receiver.get("email", "N/A")}</p></div>', unsafe_allow_html=True)

    amount = st.number_input("Amount to Transfer", min_value=0.01, step=50.0, format="%.2f", on_change=reset_otp_flow)
    
    # --- NEW OTP FLOW LOGIC ---
    send_button_disabled = False
    if sender_balance > 0:
        drain_ratio = amount / sender_balance

        # Scenario 1: High-risk transaction (70%-99.9% drain), requires OTP
        if 0.7 <= drain_ratio < 1.0:
            st.warning("⚠️ **High-Risk Transaction Detected!** An OTP is required to proceed.", icon="❗")
            
            # Step 1: Show button to request an OTP via email
            if not st.session_state.otp_requested:
                if st.button("Email me an OTP"):
                    with st.spinner("Requesting OTP..."):
                        try:
                            response = requests.post(f"{API_URL}/request_otp", json={"user_id": sender['user_id']})
                            if response.ok:
                                st.session_state.otp_requested = True
                                st.success(f"An OTP has been sent to {sender_email}.")
                                st.rerun() # Rerun to show the next step
                            else:
                                st.error(f"Failed to send OTP: {response.json().get('error', 'Unknown error')}")
                        except requests.RequestException as e:
                            st.error(f"API Error: Could not connect to the API to request OTP.")

            # Step 2: If OTP has been requested, show the verification input
            if st.session_state.otp_requested and not st.session_state.otp_verified:
                otp_input = st.text_input("Enter OTP from email", type="password")
                if st.button("Verify OTP"):
                    with st.spinner("Verifying..."):
                        try:
                            response = requests.post(f"{API_URL}/verify_otp", json={"user_id": sender['user_id'], "otp": otp_input})
                            if response.ok:
                                st.session_state.otp_verified = True
                                st.success("OTP Verified! You can now send the money.")
                                st.rerun() # Rerun to reflect the verified state
                            else:
                                st.error(f"Verification Failed: {response.json().get('error', 'Invalid OTP')}")
                        except requests.RequestException as e:
                            st.error(f"API Error: Could not connect to the API to verify OTP.")
            
            # Keep the main button disabled until the OTP flow is complete
            if not st.session_state.otp_verified:
                send_button_disabled = True

        # Scenario 2: Instant block for 100% drain
        elif drain_ratio >= 1.0:
            st.error("❌ **Transaction Blocked!** Attempting to transfer the full account balance is not permitted.", icon="🚫")
            st.stop()
    
    # --- Send Money Button ---
    if st.button("Send Money", use_container_width=True, disabled=send_button_disabled):
        if amount > sender_balance:
            st.error("❌ **Transaction Blocked!** Insufficient funds.", icon="💳")
        else:
            payload = {"sender_id": sender['user_id'], "receiver_id": receiver['user_id'], "amount": amount}
            with st.spinner("Submitting transaction..."):
                resp = requests.post(f"{API_URL}/submit_transaction", json=payload)
                if not resp.ok:
                    st.error(f"API Error: {resp.text}")
                else:
                    tx_id = resp.json()['transaction_id']
                    st.info(f"Transaction submitted (ID: {tx_id}). Awaiting final confirmation...")
                    for _ in range(20):
                        time.sleep(1)
                        poll_resp = requests.get(f"{API_URL}/get_prediction/{tx_id}")
                        if poll_resp.ok and poll_resp.json().get('status') == 'completed':
                            res = poll_resp.json()
                            if res['is_fraud']:
                                st.error(f"🔴 **Transaction Blocked by Backend!** Reason: {res.get('details', {}).get('rule', 'Suspicious Activity')}", icon="🚨")
                            else:
                                st.success("🟢 **Transaction Approved & Completed!**", icon="✅")
                                st.balloons()
                            reset_otp_flow()
                            time.sleep(4)
                            st.rerun()
                            break
                    else:
                        st.warning("Prediction timed out.")
