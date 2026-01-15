# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# ----------------------------
# 1. Create synthetic dataset
# ----------------------------
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'hours_studied': np.random.uniform(0, 20, n_samples),
    'attendance': np.random.uniform(50, 100, n_samples),
    'previous_grade': np.random.randint(50, 100, n_samples),
})

# Simple rule: higher hours, higher attendance, higher previous grade = higher final grade
data['final_grade'] = (
    data['hours_studied'] * 2 + 
    data['attendance'] * 0.3 + 
    data['previous_grade'] * 0.5 + 
    np.random.normal(0, 5, n_samples)
).clip(0, 100).round(2)

# ----------------------------
# 2. Prepare features & target
# ----------------------------
X = data[['hours_studied', 'attendance', 'previous_grade']]
y = data['final_grade']

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------
# 4. Train Linear Regression
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# 5. Save model & scaler
# ----------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("Model and scaler saved to model.pkl")
