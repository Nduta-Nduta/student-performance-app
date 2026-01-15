# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# ----------------------------
# 1. Create synthetic dataset
# ----------------------------
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'practice_hours': np.random.uniform(0, 20, n_samples),
    'lessons_taken': np.random.randint(1, 15, n_samples),
    'theory_score': np.random.randint(40, 100, n_samples),
    'mock_test_score': np.random.randint(30, 100, n_samples),
    'missed_lessons': np.random.randint(0, 5, n_samples),
})

# Simple rule: higher practice, higher scores, fewer missed lessons = Pass
data['pass'] = ((data['practice_hours'] > 10) & 
                (data['theory_score'] > 60) & 
                (data['mock_test_score'] > 50) & 
                (data['missed_lessons'] < 3)).astype(int)

# ----------------------------
# 2. Prepare features & target
# ----------------------------
X = data[['practice_hours', 'lessons_taken', 'theory_score', 'mock_test_score', 'missed_lessons']]
y = data['pass']

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------
# 4. Train Logistic Regression
# ----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------------
# 5. Save model & scaler
# ----------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("Model and scaler saved to model.pkl")
