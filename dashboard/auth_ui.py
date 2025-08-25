# dashboard/auth_ui.py
import streamlit as st
import requests

BACKEND_BASE = st.secrets.get("backend_base", "http://127.0.0.1:8000")

st.set_page_config(page_title="Auth", page_icon="🔐")
st.title("User Authentication")

tab_signup, tab_login = st.tabs(["Signup", "Login"])  # returns two tab containers [web:186]

with tab_signup:  # use the tab container, not the list itself [web:186]
    st.subheader("Create account")
    with st.form("signup_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign up")
    if submit:
        try:
            r = requests.post(
                f"{BACKEND_BASE}/signup",
                json={"name": name, "email": email, "password": pwd},
                timeout=10,
            )
            if r.status_code == 201:
                st.success("Signup successful. Please log in.")
            else:
                st.error(r.json().get("error", f"Signup failed: {r.status_code}"))
        except Exception as e:
            st.error(f"Request failed: {e}")

with tab_login:
    st.subheader("Log in")
    with st.form("login_form"):
        email_l = st.text_input("Email", key="login_email")
        pwd_l = st.text_input("Password", type="password", key="login_pwd")
        submit_l = st.form_submit_button("Log in")
    if submit_l:
        try:
            r = requests.post(
                f"{BACKEND_BASE}/login",
                json={"email": email_l, "password": pwd_l},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                st.session_state["token"] = data["access_token"]
                st.session_state["user_email"] = email_l
                st.success("Logged in")
            else:
                st.error(r.json().get("error", f"Login failed: {r.status_code}"))
        except Exception as e:
            st.error(f"Request failed: {e}")

if "token" in st.session_state:
    st.info(f"Token present. User: {st.session_state.get('user_email')}")
    if st.button("Logout"):
        st.session_state.pop("token", None)
        st.session_state.pop("user_email", None)
        st.rerun()
