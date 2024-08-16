# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and label encoder
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Streamlit UI
st.title("Iris Flower Classification")

# Input fields for flower measurements
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.3)

# Predict button
if st.button('Classify'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    species = le.inverse_transform(prediction)
    st.write(f"The predicted species is: **{species[0]}**")

