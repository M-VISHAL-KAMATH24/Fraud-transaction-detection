# seed_database.py
import psycopg2
from faker import Faker
import random

# --- PostgreSQL Connection Configuration ---
DB_CONFIG = {
    "dbname": "fraud_db",
    "user": "postgres",
    "password": "vil100sr", # <-- IMPORTANT: Use your actual password
    "host": "localhost",
    "port": "5432"
}

def seed_users(num_users=20):
    """Connects to the database and inserts fake users."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    fake = Faker()

    # Clear existing users to avoid duplicates on re-run
    cursor.execute("DELETE FROM users;")
    print("Cleared existing users.")

    for i in range(1, num_users + 1):
        user_id = f"user_{i}"
        user_name = fake.name()
        # Give users a random starting balance between 1,000 and 500,000
        balance = round(random.uniform(1000, 500000), 2)
        
        cursor.execute(
            "INSERT INTO users (user_id, user_name, current_balance) VALUES (%s, %s, %s)",
            (user_id, user_name, balance)
        )
        print(f"Inserted {user_id}: {user_name} with balance {balance}")

    conn.commit()
    cursor.close()
    conn.close()
    print("\nDatabase seeding complete!")

if __name__ == '__main__':
    seed_users()

