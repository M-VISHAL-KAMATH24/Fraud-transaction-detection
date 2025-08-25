# backend/run.py
from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
import jwt
import os

JWT_SECRET = os.environ.get("JWT_SECRET", "dev_secret_change_me")
JWT_ALG = "HS256"

# in-memory "db" for demo; swap with Postgres later
USERS = {}  # key: email, value: {name, email, password_hash, created_at}

def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok", service="fraud-backend"), 200

    @app.post("/signup")
    def signup():
        data = request.get_json(force=True)
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        if not name or not email or not password:
            return jsonify(error="name, email, password are required"), 400
        if email in USERS:
            return jsonify(error="user already exists"), 409
        USERS[email] = {
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        return jsonify(message="signup ok"), 201

    @app.post("/login")
    def login():
        data = request.get_json(force=True)
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        user = USERS.get(email)
        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify(error="invalid credentials"), 401
        exp = datetime.now(timezone.utc) + timedelta(minutes=30)
        token = jwt.encode({"sub": email, "exp": exp}, JWT_SECRET, algorithm=JWT_ALG)
        return jsonify(access_token=token, token_type="Bearer", expires_at=exp.isoformat()), 200

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
