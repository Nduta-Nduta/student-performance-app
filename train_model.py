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
n_samples = 300

data = pd.DataFrame({
    'hours_studied': np.random.uniform(0, 20, n_samples),
    'attendance': np.random.uniform(50, 100, n_samples),
    'previous_grade': np.random.randint(50, 100, n_samples),
    'sleep_hours': np.random.uniform(4, 10, n_samples),
    'study_group': np.random.randint(0, 10, n_samples),
    'assignment_completion': np.random.uniform(50, 100, n_samples),
    'test_prep_days': np.random.randint(0, 14, n_samples),
})

# Weighted formula: higher values generally lead to better grades
data['final_grade'] = (
    data['hours_studied'] * 1.5 + 
    data['attendance'] * 0.4 + 
    data['previous_grade'] * 0.6 +
    data['sleep_hours'] * 0.8 +
    data['study_group'] * 0.5 +
    data['assignment_completion'] * 0.3 +
    data['test_prep_days'] * 0.7 +
    np.random.normal(0, 3, n_samples)
).clip(0, 100).round(2)

# ----------------------------
# 2. Prepare features & target
# ----------------------------
X = data[['hours_studied', 'attendance', 'previous_grade', 'sleep_hours', 'study_group', 'assignment_completion', 'test_prep_days']]
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
