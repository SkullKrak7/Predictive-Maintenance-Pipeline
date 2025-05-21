# Predictive Maintenance Pipeline with Flask Dashboard

This repository contains a full machine learning pipeline for predictive maintenance.
It includes data processing, model training with XGBoost, and a simple Flask web application
to serve real-time predictions.

## Project Overview

Predictive maintenance involves forecasting equipment failures before they occur, helping reduce downtime and cost. This project follows a complete ML workflow:

* Data loading and exploration
* Cleaning and preprocessing
* Feature engineering and selection
* Model training and evaluation
* Saving the trained model
* Deploying a Flask dashboard for live predictions

## Technologies Used

* Python 3
* pandas, numpy
* scikit-learn
* xgboost
* Flask (with HTML/Jinja2)

## Setup Instructions

```bash
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

pip install -r requirements.txt

python flask_app.py
```

Visit `http://127.0.0.1:5000` in your browser to use the dashboard.

## Docker Usage

To build and run the app using Docker:

```bash
# Build the image
docker build -t predictive-maintenance .

# Run the container
docker run -p 5000:5000 predictive-maintenance
```

Then open `http://localhost:5000` in your browser to use the dashboard.

## Folder Structure

```
predictive-maintenance/
├── templates/                  # HTML templates
│   └── index.html
├── flask_app.py                # Flask app code
├── train_model.py              # Model training and saving
├── predictive_maintenance.csv  # Dataset
├── saved_model.pkl             # Trained model (optional)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container config
└── README.md
```

## GitHub Topics

machine-learning, predictive-maintenance, flask, xgboost, dashboard
