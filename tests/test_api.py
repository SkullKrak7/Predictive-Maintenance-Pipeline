import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
)

import json
from flask_app import app

def test_health_ok():
    app.testing = True
    c = app.test_client()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

def test_predict_valid():
    app.testing = True
    c = app.test_client()
    payload = {"rotational_speed": 1200.0, "torque": 35.0, "tool_wear": 10.0}
    r = c.post("/predict", data=json.dumps(payload), content_type="application/json")
    body = r.get_json()
    assert r.status_code == 200
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["predicted_label"] in [0, 1]
    assert set(body.keys()) == {"failure_probability", "predicted_label", "threshold"}
    assert isinstance(body["failure_probability"], float)
    assert isinstance(body["predicted_label"], int)

def test_predict_invalid_422():
    app.testing = True
    c = app.test_client()
    r = c.post("/predict", data=json.dumps({"rotational_speed": "fast"}), content_type="application/json")
    assert r.status_code == 422

def test_predict_bounds():
    app.testing = True
    c = app.test_client()
    bad = {"rotational_speed": 5000, "torque": 30, "tool_wear": 5}
    r = c.post("/predict", data=json.dumps(bad), content_type="application/json")
    assert r.status_code == 422
