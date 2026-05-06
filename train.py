import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('data/heart.csv')

# Split
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline model
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
print("Baseline Accuracy:", baseline.score(X_test, y_test))

# Final model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Random Forest Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, 'model/model.pkl')