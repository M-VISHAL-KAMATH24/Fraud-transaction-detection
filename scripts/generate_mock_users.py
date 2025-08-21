import pandas as pd
from faker import Faker
import random

fake = Faker()

# Generate 100 mock users
users = []
for _ in range(100):
    user_id = fake.uuid4()  # Unique ID
    billing_country = fake.country_code()  # e.g., 'US'
    billing_lat = fake.latitude()  # For distance checks
    billing_long = fake.longitude()
    users.append({
        'user_id': user_id,
        'billing_country': billing_country,
        'billing_lat': billing_lat,
        'billing_long': billing_long
    })

df_users = pd.DataFrame(users)
df_users.to_csv('mock_users.csv', index=False)  # Save as CSV
print("Mock user DB created: mock_users.csv")
