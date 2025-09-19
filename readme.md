# Predictive Maintenance Pipeline

A compact, reproducible predictive maintenance project: train an XGBoost model on four numeric sensor features, save insight artefacts, and serve live predictions via a Flask web interface and JSON API. The pipeline is aligned across training, serving, tests, Docker, and CI.

## Why this project
- Forecast likely machine failures to cut downtime and cost, using a lean but realistic ML workflow that’s easy to clone, run, and extend.

## Key features
- Four input features only: Process_temperature_K, Rotational_speed_rpm, Torque_Nm, Tool_wear_min, fixed in order at train and serve time.
- Class imbalance handled with XGBoost scale_pos_weight; no SMOTE and no threshold tuning are implemented.
- Artefacts saved to outputs/: model.pkl (model, scaler, feature_names), corr_heatmap.png, feature_importance_rf.png, metrics.json.
- Flask app exposes a web form and JSON endpoints; a /health route confirms liveness.

## Quick start

### Local
- Requires Python 3.11.
- Steps:
\`\`\`bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Train to produce model.pkl and insight artefacts
python train_model.py

# Serve the app (web + API)
python flask_app.py
\`\`\`
- Open http://127.0.0.1:5000 for the form; GET /health should return {"status":"ok"}.

### Docker
Self‑contained image that trains during build:
\`\`\`dockerfile
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip `
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/outputs && python train_model.py
CMD ["python", "flask_app.py"]
\`\`\`
Build and run:
\`\`\`bash
docker build -t predictive-maintenance .
docker run -p 5000:5000 predictive-maintenance
\`\`\`

## Project structure
\`\`\`
.
├─ templates/
│  └─ index.html
├─ outputs/
├─ tests/
│  └─ test_api.py
├─ flask_app.py
├─ train_model.py
├─ predictive_maintenance.csv
├─ requirements.txt
├─ Dockerfile
└─ .github/workflows/ci.yml
\`\`\`

## Data and features
- Dropped IDs: UDI, Product ID.
- Encoded and then removed: Type one‑hots (Type_L, Type_M), Failure Type (to avoid leakage).
- Standardised numeric sensors; removed Air temperature [K] due to high correlation with Process temperature [K].

Final feature order for train and serve:
- Process_temperature_K
- Rotational_speed_rpm
- Torque_Nm
- Tool_wear_min

## Training pipeline
- Preprocessing: drop IDs/NA, encode + drop Type and Failure Type; scale numerics; drop Air temperature [K].
- Model: XGBoost with scale_pos_weight; stratified split; metrics saved.
- Artefacts: model.pkl with {model, scaler, feature_names}; outputs/corr_heatmap.png; outputs/feature_importance_rf.png; outputs/metrics.json.

## Serving and API
- Health: GET /health → {"status":"ok"}, 200.
- Web UI: GET / renders a form to submit the four inputs and view probability + label.
- Predict: POST /predict JSON:
\`\`\`json
{
  "process_temperature": 320.0,
  "rotational_speed": 1200.0,
  "torque": 35.0,
  "tool_wear": 10.0,
  "threshold": 0.5
}
\`\`\`
Returns probability and label using the provided threshold (default 0.5).

## Testing and CI
- Local: python -m pytest -q. Tests include health and predict; a targeted filter silences the sklearn feature-names warning.
- CI: install deps → train_model.py → ruff → black → pytest → docker build; PYTHONPATH set to repo root for reliable imports.

## Performance snapshot
- The latest run writes metrics to outputs/metrics.json for verification and review.
- Open that file to see test accuracy and the class‑weighted F1 score, alongside the full classification report.
- Correlation and feature importance plots are saved to outputs/corr_heatmap.png and outputs/feature_importance_rf.png for quick sanity checks.

## Quality gates
- Run lint, format‑check, and tests in one go:

\`\`\`bash
ruff check . && black --check . && pytest -q
\`\`\`

## Licence
MIT Licence. See LICENSE.