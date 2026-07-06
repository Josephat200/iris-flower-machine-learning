from pathlib import Path

import joblib
import numpy as np
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "Iris_Classification" / "models" / "iris_model.pkl"
SCALER_PATH = PROJECT_DIR / "Iris_Classification" / "models" / "scaler.pkl"

SPECIES = ["Setosa", "Versicolor", "Virginica"]

st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python Iris_Classification/iris_classification.py` to generate it."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


st.title("🌸 Iris Flower Classification")
st.write("Enter the flower measurements below to predict the species.")

try:
    model, scaler = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.8, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=3.8, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

if st.button("Predict", type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    st.success(f"Predicted Species: **{SPECIES[prediction]}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        st.write("Prediction confidence:")
        st.bar_chart({species: [prob] for species, prob in zip(SPECIES, probabilities)})
