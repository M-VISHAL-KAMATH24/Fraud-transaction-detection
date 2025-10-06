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

# --- The pool of emails to be assigned randomly ---
EMAIL_POOL = [
    'vishalkamath69@gmail.com',
    'kamathvishal26@gmail.com',
    '22k20.vishal@sjec.ac.in',
    '22k37.rohan@sjec.ac.in',
    '22k57.mahiba@sjec.ac.in',
    'vishalssss06@gmail.com'
]

def seed_users(num_users=20):
    """Connects to the database and inserts fake users with random emails."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        fake = Faker()

        # Clear existing users to start fresh
        cursor.execute("TRUNCATE TABLE users RESTART IDENTITY;")
        print("Cleared existing users.")

        # Create new users and assign emails in one loop
        for i in range(1, num_users + 1):
            user_id = f"user_{i}"
            user_name = fake.name()
            balance = round(random.uniform(1000, 500000), 2)
            # Pick a random email from your list
            assigned_email = random.choice(EMAIL_POOL)
            
            cursor.execute(
                "INSERT INTO users (user_id, user_name, current_balance, email) VALUES (%s, %s, %s, %s)",
                (user_id, user_name, balance, assigned_email)
            )
            print(f"Inserted {user_id}: {user_name} with email {assigned_email}")

        conn.commit()
        print("\nDatabase seeding with emails complete!")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database error: {error}")
    finally:
        if conn is not None:
            cursor.close()
            conn.close()

if __name__ == '__main__':
    seed_users()
