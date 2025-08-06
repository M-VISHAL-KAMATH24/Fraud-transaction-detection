# scripts/eda.py - Exploratory Data Analysis for PaySim Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

# Ensure outputs folder exists
os.makedirs('outputs', exist_ok=True)

# Load dataset
df = pd.read_csv('data/paysim.csv')  # Adjust path if needed
print("Dataset Head:\n", df.head())
print("\nDataset Description:\n", df.describe())
print("\nFraud Distribution:\n", df['isFraud'].value_counts())

# Visualize and save fraud distribution plot
sns.countplot(x='isFraud', data=df)
plt.title('Fraud Distribution')
plt.savefig('outputs/fraud_distribution.png')
plt.close()
print("Fraud distribution plot saved to outputs/fraud_distribution.png")

# Handle imbalance preview (SMOTE on a subset)
X = df.drop('isFraud', axis=1)  # Features (preprocess fully later)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE (demo on numeric features; encode categoricals in preprocess.py)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train.select_dtypes('number'), y_train)
print("\nResampled Fraud Distribution:\n", pd.Series(y_resampled).value_counts())

# Save basic stats (for milestone)
df.describe().to_csv('outputs/basic_stats.csv')
print("Basic stats saved to outputs/basic_stats.csv")
