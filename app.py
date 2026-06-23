import streamlit as st
import pandas as pd
import joblib
from utils import feature_engineering

# Load model
model = joblib.load("models/logistic_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details below:")

# ---------------- INPUTS ---------------- #

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("TenureMonths", 0, 100)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

monthly = st.number_input("Monthly Charges", 0.0)
total = st.number_input("Total Charges", 0.0)

# default values for other services
default_yes_no = "No"

# -------------- CREATE DATAFRAME -------------- #

input_data = pd.DataFrame([{
    "Gender": gender,
    "Senior Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Contract": contract,
    "Paperless Billing": paperless,
    "Payment Method": payment,
    "Internet Service": internet,
    "Monthly Charges": monthly,
    "Total Charges": total,
    "Phone Service": "Yes",
    "Multiple Lines": default_yes_no,
    "Online Security": default_yes_no,
    "Online Backup": default_yes_no,
    "Device Protection": default_yes_no,
    "Tech Support": default_yes_no,
    "Streaming TV": default_yes_no,
    "Streaming Movies": default_yes_no,
    "Latitude": 0,
    "Longitude": 0
}])

# -------------- FEATURE ENGINEERING -------------- #

input_data = feature_engineering(input_data)


# -------------- PREPROCESS + PREDICT -------------- #

processed = preprocessor.transform(input_data)
prediction = model.predict(processed)[0]
prob = model.predict_proba(processed)[0][1]

# -------------- OUTPUT -------------- #

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"⚠️ Customer WILL CHURN (Probability: {prob:.2f})")
else:
    st.success(f"✅ Customer will NOT churn (Probability: {prob:.2f})")