# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os 

st.set_page_config(page_title="Titanic Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival:")

# Safe load
if not os.path.exists("logistic_model.pkl"):
    st.error("🚫 Model file not found. Please upload 'logistic_model.pkl' to the app directory.")
    st.stop()

# Load model
model = joblib.load("logistic_model.pkl")

# Sidebar user input
def get_user_input():
    pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.sidebar.radio("Sex", ['male', 'female'])
    age = st.sidebar.slider("Age", 0, 80, 25)
    sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
    fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 32.2)

    # Convert input to model format
    sex_encoded = 1 if sex == 'male' else 0

    data = {
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare
    }

    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# Display input
st.subheader("Passenger Input Data")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Result
st.subheader("Prediction")
result = "🟢 Survived" if prediction[0] == 1 else "🔴 Did Not Survive"
st.write(f"The model predicts: **{result}**")

st.subheader("Prediction Probability")
st.write(f"Chance of survival: **{prediction_proba[0][1]*100:.2f}%**")
