# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer will churn using classification models and a deployed Streamlit web application.

---

## 🚀 Live Demo

[Click here to view the app](https://your-streamlit-app-link)

---

## 📌 Problem Statement

Telecom companies lose revenue when customers leave (churn).
The goal is to predict churn in advance so businesses can take proactive actions.

---

## 📂 Dataset

- Telco Customer Churn Dataset
- Features include:
  - Customer demographics
  - Subscription details
  - Service usage
  - Billing information

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing

- Missing value handling
- Encoding categorical variables
- Feature scaling

### 2. Feature Engineering

- Tenure groups
- Spend categories
- Engagement features
- Risk indicators

### 3. Handling Imbalance

- SMOTE (Synthetic Minority Oversampling Technique)

### 4. Models Trained

- Logistic Regression ✅ (Final Model)
- Decision Tree
- Random Forest
- XGBoost

---

## 🏆 Best Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~0.77 |
| Precision | ~0.54 |
| Recall    | ~0.79 |
| F1 Score  | ~0.64 |
| ROC-AUC   | ~0.86 |

---

## 📊 Key Insights

- Month-to-month contracts have highest churn
- Low tenure customers are high risk
- Electronic check payments increase churn probability
- Customers with more services are less likely to churn

---

## 🌐 Deployment

- Built using **Streamlit**
- Real-time prediction interface
- User-friendly input form

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Matplotlib / Seaborn

---

## 📁 Project Structure


customer-churn-prediction/

│

├── app/

│   └── app.py

│

├── models/

│   ├── logistic_model.pkl

│   └── preprocessor.pkl

│

├── notebooks/

│

├── data/

│

├── requirements.txt

├── README.md

├── .gitignore
