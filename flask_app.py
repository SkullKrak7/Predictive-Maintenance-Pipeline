from flask import Flask, jsonify, request, render_template, request as flask_request
import time, pickle, numpy as np
from typing import List, Any
from pydantic import ValidationError
from models import InputModel, OutputModel

app = Flask(__name__)

class Metrics:
    def __init__(self) -> None:
        self.start = time.time()
        self.count = 0
        self.lat_ms: List[float] = []
    def observe(self, ms: float) -> None:
        self.count += 1
        self.lat_ms.append(ms)
        if len(self.lat_ms) > 1000:
            self.lat_ms = self.lat_ms[-1000:]
    def p95(self) -> float:
        if not self.lat_ms: return 0.0
        xs = sorted(self.lat_ms)
        k = max(0, int(0.95*(len(xs)-1)))
        return xs[k]

METRICS = Metrics()

with open("model.pkl", "rb") as f:
    ARTS = pickle.load(f)
MODEL = ARTS.get("model")
SCALER = ARTS.get("scaler")
FEATURES = ARTS.get("feature_names") or ARTS.get("features") or [
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]

def vectorize(payload: InputModel) -> np.ndarray:
    mapping = {
        "rotational_speed": "Rotational speed [rpm]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool wear [min]",
    }
    row = {feat: 0.0 for feat in FEATURES}
    data = payload.model_dump()
    for api_key, feat in mapping.items():
        row[feat] = float(data[api_key])
    X = np.array([[row[f] for f in FEATURES]], dtype=float)
    if SCALER is not None:
        X = SCALER.transform(X)
    return X

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.get("/metrics")
def metrics():
    return jsonify({
        "uptime_s": round(time.time() - METRICS.start, 2),
        "requests_total": METRICS.count,
        "latency_p95_ms": round(METRICS.p95(), 2),
    }), 200

@app.post("/predict")
def predict():
    t0 = time.time()
    try:
        payload = InputModel(**request.get_json(force=True))
    except ValidationError as ve:
        return jsonify({"error":"validation_error","details":ve.errors()}), 422
    threshold = float(request.args.get("threshold","0.5"))
    X = vectorize(payload)
    proba = float(MODEL.predict_proba(X)[0,1]) if hasattr(MODEL,"predict_proba") else float(MODEL.predict(X)[0])
    label = int(proba >= threshold)
    METRICS.observe((time.time()-t0)*1000.0)
    return jsonify({"failure_probability": proba, "predicted_label": label, "threshold": threshold}), 200

@app.get("/")
def home():
    return render_template("index.html", result=None, metrics=None, error=None)

@app.post("/ui/predict")
def ui_predict():
    try:
        rs = float(flask_request.form["rotational_speed"])
        tq = float(flask_request.form["torque"])
        tw = float(flask_request.form["tool_wear"])
        threshold = float(flask_request.form.get("threshold","0.5"))
        threshold = min(1.0, max(0.0, threshold))
        form = {"rotational_speed": rs, "torque": tq, "tool_wear": tw}
    except Exception as e:
        return render_template("index.html", result=None, metrics=None, error=str(e))
    X = vectorize(InputModel(**form))
    proba = float(MODEL.predict_proba(X)[0,1]) if hasattr(MODEL,"predict_proba") else float(MODEL.predict(X)[0])
    label = int(proba >= threshold)
    METRICS.count += 1
    m = {"uptime_s": round(time.time()-METRICS.start,2), "requests_total": METRICS.count, "latency_p95_ms": round(METRICS.p95(),2)}
    res = {"failure_probability": round(proba,6), "predicted_label": label, "threshold": threshold}
    return render_template("index.html", result=res, metrics=m)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

