# Iris Flower Classification (Streamlit)

A Streamlit web app that predicts iris flower species (Setosa, Versicolor, Virginica) from sepal/petal measurements using a pre-trained scikit-learn model.

## Project Structure

```
app.py                          # Streamlit entry point (deploy this file)
requirements.txt                # Pinned, minimal runtime dependencies
runtime.txt                     # Python version for cloud deployment
.streamlit/config.toml          # Streamlit server configuration
Iris_Classification/
  iris_classification.py        # Training script (regenerates the model)
  models/iris_model.pkl         # Trained LogisticRegression model
  models/scaler.pkl             # Fitted StandardScaler
  notebooks/                    # Exploratory notebook
  images/                       # Saved chart(s)
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo.
3. Set the main file path to `app.py`.
4. Streamlit Cloud will install `requirements.txt` (Python version from `runtime.txt`) and start the app automatically — no extra configuration needed.

## Retrain the Model

```bash
python Iris_Classification/iris_classification.py
```

This regenerates `Iris_Classification/models/iris_model.pkl` and `scaler.pkl`.
