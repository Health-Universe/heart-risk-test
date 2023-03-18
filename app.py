import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title("Heart Attack Risk Calculator")
st.write("Predict the risk of heart attack in the next 10 years")

# Inputs
age = st.number_input("Age (years)", 30, 80, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
smoker = st.selectbox("Smoker?", ["Yes", "No"])
bp = st.number_input("Systolic Blood Pressure (mmHg)", 90, 200, step=1)
cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, step=1)
hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, step=1)

# Preprocess inputs
input_data = np.array([[age, int(gender == "Male"), int(smoker == "Yes"), bp, cholesterol, hdl]])
input_data = (input_data - np.array([54, 0.5, 0.5, 131, 246, 50])) / np.array([8.5, 0.5, 0.5, 17, 44, 15])

# Load model
model = LogisticRegression()
model.coef_ = np.array([[-0.33728, 0.93963, 0.94759, 0.75245, 0.40556, -0.62099]])
model.intercept_ = np.array([-2.88537])

# Predict risk
risk = model.predict_proba(input_data)[0, 1]

# Display risk
st.write("Risk of heart attack in the next 10 years: {:.1f}%".format(risk * 100))

