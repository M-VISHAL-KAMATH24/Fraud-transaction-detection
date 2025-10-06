import psycopg2
from faker import Faker
import random

DB_CONFIG = {"dbname": "fraud_db", "user": "postgres", "password": "vil100sr", "host": "localhost", "port": "5432"}

def seed_users(num_users=20):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    fake = Faker()
    # This ensures a clean slate every time you run it
    cursor.execute("TRUNCATE TABLE users RESTART IDENTITY;")
    print("Cleared existing users.")

    for i in range(1, num_users + 1):
        user_id = f"user_{i}"
        user_name = fake.name()
        # Generate a random INTEGER balance
        balance = random.randint(10000, 500000)
        
        cursor.execute(
            "INSERT INTO users (user_id, user_name, current_balance) VALUES (%s, %s, %s)",
            (user_id, user_name, balance)
        )
    conn.commit()
    cursor.close()
    conn.close()
    print("\nDatabase seeding with integer balances complete!")

if __name__ == '__main__':
    seed_users()
