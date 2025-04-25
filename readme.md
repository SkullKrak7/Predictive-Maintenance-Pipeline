# 🛠️ Predictive Maintenance Pipeline with Flask Dashboard

This project builds a complete machine learning pipeline for predictive maintenance using a dataset of industrial machine failures.  
A lightweight Flask web application is included to serve real-time predictions based on the trained model.

## 🚀 Project Overview

Predictive maintenance anticipates equipment failures before they occur, helping reduce downtime and maintenance costs.  
This project follows a structured ML workflow including data preprocessing, feature selection, model training, evaluation, and deployment.

## 🗂️ Project Workflow

- **Data Loading & Exploration**  
  Import and inspect the predictive maintenance dataset.

- **Data Cleaning & Preprocessing**  
  - Drop irrelevant columns  
  - Handle missing values  
  - Encode categorical features  
  - Normalize numerical features

- **Feature Engineering & Selection**  
  - Correlation analysis to remove redundant features  
  - Feature importance analysis using Random Forest

- **Model Training & Evaluation**  
  - Train an XGBoost classifier  
  - Evaluate performance using accuracy, precision, recall, and F1-score

- **Model Saving**  
  Save the final model (`.pkl` format) for later inference.

- **Flask Dashboard Deployment**  
  Deploy a web app where users can input sensor data and receive live fault predictions.

## 💻 Technologies Used

- Python 3
- pandas
- numpy
- scikit-learn
- xgboost
- flask (for web dashboard)
- HTML / Jinja2 templates

## 🛠️ Setup Instructions

```bash
# Create a virtual environment
python -m venv env
source env/bin/activate  # or 'env\Scripts\activate' on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python flask_app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to use the prediction dashboard.

## 📁 Folder Structure

```text
predictive-maintenance/
├── templates/                  # HTML files for the dashboard
│   └── index.html               # Main web page
├── flask_app.py                 # Flask app code
├── main.py                      # Model training and saving
├── predictive_maintenance.csv   # Dataset
├── saved_model.pkl              # Trained ML model
├── requirements.txt             # Required libraries
└── README.md
```

## 📣 Contributions

Open to suggestions and improvements! Feel free to fork or raise an issue.
