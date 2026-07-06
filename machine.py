import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")

species = ["Setosa", "Versicolor", "Virginica"]

st.title("🌸 Iris Flower Classification")

st.write("Enter the flower measurements below.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(data)
    st.success(f"Predicted Species: {species[prediction[0]]}")
    