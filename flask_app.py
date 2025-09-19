import time
import pickle
import numpy as np
from typing import List, Any
from flask import Flask, jsonify, request
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
        k = max(0, int(0.95 * (len(xs) - 1)))
        return xs[k]

METRICS = Metrics()

# Load artefacts saved by train_model.py
with open("model.pkl", "rb") as f:
    ARTS = pickle.load(f)

MODEL = ARTS.get("model")
SCALER = ARTS.get("scaler")
FEATURES = (
    ARTS.get("feature_names")
    or ARTS.get("features")
    or ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
)

def vectorize(payload: InputModel) -> np.ndarray:
    mapping = {
        "rotational_speed": "Rotational speed [rpm]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool wear [min]",
    }
    row = {feat: 0.0 for feat in FEATURES}
    data = payload.dict()
    for api_key, feat in mapping.items():
        row[feat] = float(data[api_key])
    X = np.array([[row[f] for f in FEATURES]], dtype=float)
    if SCALER is not None:
        X = SCALER.transform(X)
    return X

@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"}), 200

@app.get("/metrics")
def metrics() -> Any:
    return jsonify({
        "uptime_s": round(time.time() - METRICS.start, 2),
        "requests_total": METRICS.count,
        "latency_p95_ms": round(METRICS.p95(), 2),
    }), 200

@app.post("/predict")
def predict() -> Any:
    t0 = time.time()
    try:
        payload = InputModel(**request.get_json(force=True))
    except ValidationError as ve:
        return jsonify({"error": "validation_error", "details": ve.errors()}), 422
    threshold = float(request.args.get("threshold", "0.5"))
    X = vectorize(payload)
    proba = float(MODEL.predict_proba(X)[0, 1]) if hasattr(MODEL, "predict_proba") else float(MODEL.predict(X)[0])
    label = int(proba >= threshold)
    METRICS.observe((time.time() - t0) * 1000.0)
    out = OutputModel(failure_probability=proba, predicted_label=label, threshold=threshold)
    return jsonify(out.dict()), 200
@app.get("/predict")
def predict_usage():
    # Preserve POST-only semantics while guiding users/tools
    return jsonify({
        "message": "Use POST /predict with JSON body {rotational_speed, torque, tool_wear} and optional ?threshold",
        "example": {
            "method": "POST",
            "url": "/predict?threshold=0.5",
            "body": {"rotational_speed": 1500, "torque": 30, "tool_wear": 5}
        }
    }), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
