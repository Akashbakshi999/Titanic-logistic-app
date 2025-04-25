import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival:")

# Check if model file exists
model_path = "titanic_model.pkl"
if not os.path.exists(model_path):
    st.error("🚫 Model file not found. Please upload 'titanic_model.pkl' to your app directory.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Input form
def get_user_input():
    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    sex = st.sidebar.radio("Sex", ['male', 'female'])
    age = st.sidebar.slider("Age", 0, 80, 30)
    sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
    fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 32.2)

    # Encode sex: male = 1, female = 0
    sex_encoded = 1 if sex == 'male' else 0

    # Construct input with correct column names and order
    input_dict = {
        'Pclass': pclass,
        'Sex': sex_encoded,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare
    }

    expected_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    input_df = pd.DataFrame([input_dict], columns=expected_cols)

    return input_df

# Get input data
input_df = get_user_input()

# Display input
st.subheader("Passenger Input Data")
st.write(input_df)

# Predict
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader("Prediction")
result_text = "🟢 Survived" if prediction[0] == 1 else "🔴 Did Not Survive"
st.write(f"The model predicts: **{result_text}**")

st.subheader("Prediction Probability")
st.write(f"Chance of survival: **{prediction_proba[0][1] * 100:.2f}%**")
