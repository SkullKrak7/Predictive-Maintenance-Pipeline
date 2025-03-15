# Predictive Maintenance Pipeline
This repository contains a structured machine learning pipeline for predictive maintenance using a dataset of industrial machine failures. The model is built using XGBoost and Random Forest for feature importance analysis.
# Project Overview
Predictive maintenance aims to anticipate equipment failures before they happen, reducing downtime and maintenance costs. This project follows an incremental and structured approach to preprocessing, feature engineering, model training, and evaluation.

# Project Workflow
The project is structured into the following steps:

### Load & Explore Data 

Import the dataset and inspect its structure.
### Data Cleaning & Preprocessing 🛠

Drop unnecessary columns. \
Handle missing values.\
Encode categorical variables.\
Normalize numerical features.
### Feature Engineering & Selection 

Perform correlation analysis.\
Identify and drop highly correlated features.\
Use Random Forest to determine feature importance.
### Model Training & Evaluation 

Split data into training and testing sets.\
Train an XGBoost model.\
Evaluate the model using accuracy and classification metrics.
### Model Saving 

Save the trained model for future inference.