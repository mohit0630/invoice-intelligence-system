# Vendor Invoice Intelligence System

An end-to-end Machine Learning system that predicts freight costs and identifies risky vendor invoices.

This project demonstrates a complete ML workflow including data preprocessing, model training, evaluation, and deployment using a Streamlit dashboard.

---

## Features

* Freight Cost Prediction using Machine Learning
* Vendor Invoice Risk Detection
* Multiple model comparison:

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
* Best model selection based on MAE
* Streamlit dashboard for real-time predictions

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Streamlit
* SQLite
* Joblib

---

## Project Structure

.invoice-intelligence-system
│
├── data
│   └── inventory.db
│
├── freight_cost_prediction
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── inference
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── invoice-flagging
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── models
│   ├── predict_flag_invoice.pkl
│   ├── predict_freight_model.pkl
│   └── scaler.pkl
│
├── notebooks
│   ├── invoice-flagging.ipynb
│   └── Predicting-Freight.ipynb
│
├── app.py
└── README.md

---

## How to Run


Run the Streamlit application:

streamlit run app.py

---

## Business Impact

This system helps finance teams:

* Forecast freight costs
* Detect abnormal vendor invoices
* Reduce manual approval workload
* Improve financial decision-making
