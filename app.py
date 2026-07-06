import streamlit as st
import pandas as pd
import numpy as np

st.title("My First Streamlit App")
st.write("Welcome to your ML app!")

name = st.text_input("Enter your name:")

if name:
    st.write(f"Hello, {name}!")

data = pd.DataFrame({
    "x": np.arange(50),
    "y": np.random.randn(50)
})

st.line_chart(data.set_index("x"))