import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import IsolationForest

# FRAUD DETECTION DASHBOARD.
# trains an Isolation Forest (an anomaly-detection model) on normal credit-card
# transactions. anything that looks weird compared to normal spending gets
# flagged as possible fraud. there's a live dashboard + a form to score a new
# transaction. the model trains on synthetic data at startup so it just runs.
# pip install flask scikit-learn numpy

app = Flask(__name__)
rng = np.random.default_rng(42)

# features per transaction: [amount, hour_of_day, distance_from_home_km]
# normal spending: small-ish amounts, daytime, close to home
normal = np.column_stack([
    rng.normal(60, 25, 1000).clip(1, None),
    rng.normal(14, 4, 1000).clip(0, 23),
    rng.normal(5, 3, 1000).clip(0, None),
])

# train on normal data only - the model learns what "normal" looks like
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(normal)


def score(amount, hour, distance):
    # returns ("fraud"/"ok", anomaly_score)
    x = np.array([[amount, hour, distance]])
    pred = model.predict(x)[0]          # -1 = anomaly, 1 = normal
    raw = model.score_samples(x)[0]     # lower = more anomalous
    return ("fraud" if pred == -1 else "ok"), float(raw)


@app.route("/")
def home():
    # show a few example transactions already scored
    examples = [
        (45, 13, 3), (80, 19, 8), (1500, 3, 900),
        (60, 12, 2), (2200, 4, 1200), (30, 15, 1),
    ]
    rows = []
    for amt, hr, dist in examples:
        label, s = score(amt, hr, dist)
        rows.append({"amount": amt, "hour": hr, "distance": dist,
                     "label": label, "score": round(s, 3)})
    return render_template("index.html", rows=rows)


@app.route("/check", methods=["POST"])
def check():
    d = request.json
    label, s = score(float(d["amount"]), float(d["hour"]), float(d["distance"]))
    return jsonify({"label": label, "score": round(s, 3)})


if __name__ == "__main__":
    app.run(debug=True)
