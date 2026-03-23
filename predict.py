from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# Same dataset for accuracy display
data = {
    "amount": [100, 500, 2000, 15000, 30000, 120000, 800, 60000],
    "location": [0, 0, 0, 1, 1, 1, 0, 1],
    "time": [0, 1, 0, 1, 1, 1, 0, 1],
    "type": [1, 1, 0, 2, 2, 1, 3, 4],
    "fraud": [0, 0, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[["amount", "location", "time", "type"]]
y = df["fraud"]

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    amount = float(data["amount"])
    location = int(data["location"])
    time = int(data["time"])
    type_val = int(data["type"])

    prediction = model.predict([[amount, location, time, type_val]])[0]

    if prediction == 1:
        result = "Fraud Detected 🚨"
    else:
        result = "No Fraud"

    return jsonify({
        "result": result,
        "accuracy": f"{accuracy * 100:.2f}%"
    })

if __name__ == "__main__":
    app.run(port=6000)