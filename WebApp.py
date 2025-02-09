import streamlit as st
import pandas as pd
import pickle

# Loading the trained model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("📊 Advertising Sales Prediction App")
st.write("Enter advertisement budgets for TV, Radio, and Newspaper to predict Sales.")

# User input fields
tv_budget = st.number_input("📺 TV Advertising Budget ($)", min_value=0.0, step=1.0)
radio_budget = st.number_input("📻 Radio Advertising Budget ($)", min_value=0.0, step=1.0)
newspaper_budget = st.number_input("📰 Newspaper Advertising Budget ($)", min_value=0.0, step=1.0)

# Predict button
if st.button("📈 Predict Sales"):
    input_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], columns=["TV", "Radio", "Newspaper"])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Sales: {prediction:.2f} units")