import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# Load the trained model
# -----------------------------

model_path = Path(__file__).parent / "iris_model.pkl"

model = joblib.load(model_path)


# -----------------------------
# Page title
# -----------------------------

st.title("Iris Flower Prediction App")

st.header("Enter the measurements of the Iris flower:")


# -----------------------------
# Input values
# -----------------------------

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.5,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_data = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    prediction = model.predict(input_data)

    st.success(f"Predicted Iris Flower: {prediction[0]}")
