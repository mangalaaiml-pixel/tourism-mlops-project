
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")

st.title("Tourism Package Prediction")

age = st.number_input("Age")

income = st.number_input("Monthly Income")

city = st.selectbox(
    "City Tier",
    [1,2,3]
)

if st.button("Predict"):

    sample = pd.DataFrame({
        "Age":[age],
        "MonthlyIncome":[income],
        "CityTier":[city]
    })

    pred = model.predict(sample)

    if pred[0] == 1:
        st.success(
            "Customer likely to purchase package"
        )
    else:
        st.error(
            "Customer unlikely to purchase package"
        )
        
