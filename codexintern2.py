import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === Load dataset ===
file_path = r"C:/Users/GEO/Downloads/August_workshop_python/data.csv"
df = pd.read_csv(file_path)

print("=== Columns in Dataset ===")
print(df.columns)

print("\n=== First 5 Rows ===")
print(df.head())

# === Preprocessing ===
# One-hot encode 'Location' since it's categorical
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Features (independent variables)
X = df.drop('Rent', axis=1)

# Target (dependent variable)
y = df['Rent']

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Linear Regression Model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Predictions ===
y_pred = model.predict(X_test)

# === Evaluation ===
print("\nModel Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# === Visualization ===
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Actual vs Predicted Rent")
plt.show()
