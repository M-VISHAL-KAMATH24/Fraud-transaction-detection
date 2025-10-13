import psycopg2
import os
from dotenv import load_dotenv

# Load database configuration from your .env file
load_dotenv()

# Use the same DB config as your other scripts
DB_CONFIG = {
    "dbname": "fraud_db",
    "user": "postgres",
    "password": "vil100sr",
    "host": "localhost",
    "port": "5432"
}

# Sample locations for your users
user_locations = {
    'user_1': {'name': 'Delhi', 'lat': 28.6139, 'long': 77.2090},
    'user_2': {'name': 'New York', 'lat': 40.7128, 'long': -74.0060},
    'user_3': {'name': 'London', 'lat': 51.5074, 'long': -0.1278},
    'user_4': {'name': 'Paris', 'lat': 48.8566, 'long': 2.3522},
    'user_5': {'name': 'Tokyo', 'lat': 35.6895, 'long': 139.6917},
    'user_6': {'name': 'Toronto', 'lat': 43.6510, 'long': -79.3470},
    'user_7': {'name': 'Sydney', 'lat': -33.8688, 'long': 151.2093},
    'user_8': {'name': 'Moscow', 'lat': 55.7558, 'long': 37.6173},
    'user_9': {'name': 'São Paulo', 'lat': -23.5505, 'long': -46.6333},
    'user_10': {'name': 'Johannesburg', 'lat': -26.2041, 'long': 28.0473},
    'user_11': {'name': 'Mumbai', 'lat': 19.0760, 'long': 72.8777},
    'user_12': {'name': 'Los Angeles', 'lat': 34.0522, 'long': -118.2437},
    'user_13': {'name': 'Edinburgh', 'lat': 55.9533, 'long': -3.1883},
    'user_14': {'name': 'Chennai', 'lat': 13.0827, 'long': 80.2707},
    'user_15': {'name': 'Melbourne', 'lat': -37.8136, 'long': 144.9631},
}

def seed_database():
    """Connects to the DB and updates user locations."""
    conn = None
    try:
        print("Connecting to the database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("Connection successful. Seeding locations...")

        for user_id, loc in user_locations.items():
            print(f"Updating {user_id} to location: {loc['name']}")
            cursor.execute(
                """
                UPDATE users
                SET billing_lat = %s, billing_long = %s
                WHERE user_id = %s;
                """,
                (loc['lat'], loc['long'], user_id)
            )
        
        conn.commit()
        cursor.close()
        print("\nDatabase seeding complete! Your users now have billing locations.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while connecting to or seeding PostgreSQL: {error}")
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    seed_database()

