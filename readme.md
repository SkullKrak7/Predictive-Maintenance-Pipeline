# Predictive Maintenance Pipeline with Flask Dashboard

This repository contains a full machine learning pipeline for predictive maintenance. It includes data processing, model training with XGBoost, and a simple Flask web application to serve real-time predictions.

## Project Overview

Predictive maintenance involves forecasting equipment failures before they occur, helping reduce downtime and cost. This project follows a complete ML workflow:

* Data loading and exploration
* Cleaning and preprocessing
* Feature engineering and selection
* Model training and evaluation
* Handling class imbalance using scale\_pos\_weight
* Saving the trained model and scaler
* Deploying a Flask dashboard for live predictions

Note: Future work includes applying SMOTE for rare class oversampling and threshold tuning to optimize recall on the failure class.

## Technologies Used

* Python 3
* pandas, numpy
* scikit-learn
* xgboost
* Flask (with HTML/Jinja2)
* Docker

## Setup Instructions

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python flask_app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to use the dashboard.

## Docker Usage

To build and run the app using Docker:

```bash
# Build the image
docker build -t predictive-maintenance .

# Run the container
docker run -p 5000:5000 predictive-maintenance
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Folder Structure

```
predictive-maintenance/
├── templates/                  # HTML templates
│   └── index.html
├── flask_app.py                # Flask app code
├── train_model.py              # Model training and saving
├── predictive_maintenance.csv  # Dataset
├── model.pkl                   # Trained model (with scaler)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container config
└── README.md
```

## Notes

* Class imbalance (approximately 3 percent failures) handled with scale\_pos\_weight in XGBoost.
* The model and scaler are saved together to ensure consistent prediction during inference.
* Clean column naming and consistent feature preprocessing ensures reproducibility.

## License

This project is licensed under the MIT License. See LICENSE for details.
