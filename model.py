import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Updated dataset (more realistic patterns)
data = {
    "amount": [100, 500, 2000, 15000, 30000, 120000, 800, 60000],
    "location": [0, 0, 0, 1, 1, 1, 0, 1],
    "time": [0, 1, 0, 1, 1, 1, 0, 1],
    "type": [1, 1, 0, 2, 2, 1, 3, 4],  # Food, Shopping, Travel, etc
    "fraud": [0, 0, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["amount", "location", "time", "type"]]
y = df["fraud"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save
pickle.dump(model, open("model.pkl", "wb"))