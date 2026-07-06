# 🌸 Iris Flower Classification — Streamlit App

A machine learning web application that predicts the species of an iris flower (**Setosa**, **Versicolor**, or **Virginica**) from four measurements: sepal length, sepal width, petal length, and petal width. Built with **Streamlit** for the interface and **scikit-learn** for the underlying model.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Installation & Local Setup](#installation--local-setup)
7. [Using the App](#using-the-app)
8. [The Machine Learning Model](#the-machine-learning-model)
9. [Configuration Files](#configuration-files)
10. [Deployment](#deployment)
11. [Retraining the Model](#retraining-the-model)
12. [Troubleshooting](#troubleshooting)
13. [Credits](#credits)

---

## Overview

This project is an end-to-end machine learning demo: it trains a classifier on the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and serves live predictions through an interactive web UI. A user enters flower measurements, clicks **Predict**, and instantly sees the predicted species along with the model's confidence for each class.

## Features

- **Interactive prediction form** — four numeric inputs (sepal length/width, petal length/width) arranged in a clean two-column layout.
- **Instant prediction** — a single click runs the trained model and displays the predicted species.
- **Confidence visualization** — a bar chart shows the model's predicted probability for all three species, not just the winning one.
- **Cached model loading** — the model and scaler are loaded once per session (`st.cache_resource`) so predictions are fast after the first load.
- **Graceful error handling** — if the model file is missing, the app shows a clear error message instead of crashing.
- **Reproducible training pipeline** — a standalone script regenerates the dataset exploration, chart, model, and scaler from scratch at any time.

## How It Works

1. **Input** — the user enters four flower measurements in centimeters.
2. **Scaling** — the raw measurements are transformed using a `StandardScaler` that was fit on the original training data (this matches the preprocessing used during training, which is required for the model to make accurate predictions).
3. **Prediction** — the scaled features are passed to a trained `LogisticRegression` model, which outputs a predicted class (0, 1, or 2) and a probability for each of the three species.
4. **Output** — the app maps the numeric class back to a species name (Setosa / Versicolor / Virginica) and renders the result plus a probability bar chart.

## Tech Stack

| Layer            | Technology                          |
|-------------------|--------------------------------------|
| Web framework     | [Streamlit](https://streamlit.io) 1.58 |
| ML library        | [scikit-learn](https://scikit-learn.org) 1.9 (Logistic Regression + StandardScaler) |
| Data handling     | pandas, numpy                        |
| Model persistence | joblib (`.pkl` files)                |
| Charting (training)| matplotlib                          |
| Language          | Python 3.12                          |

## Project Structure

```
.
├── app.py                              # Streamlit entry point — the app users interact with
├── requirements.txt                    # Pinned Python dependencies needed to run the app
├── runtime.txt                         # Python version pin for cloud deployment
├── README.md                           # This file
├── .streamlit/
│   └── config.toml                     # Streamlit server settings (headless mode, CORS, etc.)
└── Iris_Classification/
    ├── iris_classification.py          # End-to-end training script (data → model → artifacts)
    ├── requirements.txt                # Dependencies needed only for training/exploration
    ├── README.md                       # Notes specific to the training pipeline
    ├── models/
    │   ├── iris_model.pkl              # Trained LogisticRegression model (used by app.py)
    │   └── scaler.pkl                  # Fitted StandardScaler (used by app.py)
    ├── notebooks/
    │   └── iris_classification_basics.ipynb   # Step-by-step exploratory notebook
    └── images/
        └── sepal_vs_petal.png          # Saved scatter plot from data exploration
```

### File-by-file breakdown

- **`app.py`** — The entire web application. Loads the model/scaler, renders the input form, runs predictions, and displays results. This is the file you point Streamlit (or Streamlit Cloud) at to run the app.
- **`requirements.txt`** — The exact, pinned package versions needed to run `app.py` in production. Kept minimal on purpose so cloud builds are fast and reliable.
- **`runtime.txt`** — Tells Streamlit Community Cloud which Python version to provision (`python-3.12`).
- **`.streamlit/config.toml`** — Server-level settings: runs Streamlit in headless mode (no auto-opening a local browser), disables CORS/XSRF protections that aren't needed for this use case, and turns off anonymous usage stats collection.
- **`Iris_Classification/iris_classification.py`** — A self-contained script that: loads the Iris dataset, explores it (prints stats, saves a scatter plot), splits it into train/test sets, scales the features, trains a Logistic Regression model, evaluates it, and saves the model + scaler to disk. Run this any time you want to regenerate the model artifacts.
- **`Iris_Classification/models/`** — Contains the two serialized (pickled) objects the app depends on: the trained classifier and the fitted scaler.
- **`Iris_Classification/notebooks/`** — A Jupyter notebook version of the same workflow, useful for interactive exploration and learning.
- **`Iris_Classification/images/`** — Static chart(s) generated during data exploration.

## Installation & Local Setup

**Prerequisites:** Python 3.12.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will start on `http://localhost:8501` by default (or whichever port Streamlit/your environment assigns).

## Using the App

1. Open the app in your browser.
2. Enter the four measurements:
   - **Sepal Length (cm)**
   - **Sepal Width (cm)**
   - **Petal Length (cm)**
   - **Petal Width (cm)**
3. Click **Predict**.
4. The app displays:
   - The predicted species (e.g., *"Predicted Species: **Versicolor**"*).
   - A bar chart showing the model's confidence (probability) for all three species.

Default values are pre-filled with realistic sample measurements so you can try a prediction immediately without typing anything.

## The Machine Learning Model

- **Dataset:** The classic Iris dataset (150 samples, 3 balanced classes, 4 numeric features), loaded via `sklearn.datasets.load_iris`.
- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`, `max_iter=1000`), a simple and highly interpretable classifier well-suited to this dataset.
- **Preprocessing:** Features are standardized (zero mean, unit variance) with `StandardScaler` before training and before every prediction, since Logistic Regression is sensitive to feature scale.
- **Train/test split:** 80% train / 20% test, stratified by species, with a fixed random seed (`random_state=42`) for reproducibility.
- **Evaluation:** The training script prints accuracy, a confusion matrix, and a full classification report (precision/recall/F1 per species) to the console.
- **Artifacts:** The trained model and scaler are serialized with `joblib` into `Iris_Classification/models/iris_model.pkl` and `scaler.pkl`. The app loads these two files directly — it does not retrain anything at runtime.

## Configuration Files

| File | Purpose |
|------|---------|
| `.streamlit/config.toml` | Streamlit server behavior (headless mode, CORS/XSRF, usage stats). Deliberately does **not** hardcode a host/port so it works identically on Replit, Streamlit Cloud, or any other host. |
| `requirements.txt` | Runtime dependencies for `app.py`, pinned to tested versions. |
| `runtime.txt` | Python version for platforms (like Streamlit Cloud) that read it to provision the interpreter. |

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app from your repo.
3. Set the **main file path** to `app.py`.
4. Deploy — Streamlit Cloud automatically installs `requirements.txt` and provisions Python using `runtime.txt`. No further configuration is required.

### Replit

This project already includes a configured Replit workflow and deployment target:
- **Workflow** ("Start application") runs `streamlit run app.py --server.port 5000`, which powers the live preview.
- **Deployment** is configured for Replit's Autoscale target, running the same command bound to `0.0.0.0`.

To publish on Replit, use the **Deploy** button in the workspace.

## Retraining the Model

If you want to regenerate the model (e.g., after changing the training logic):

```bash
pip install -r Iris_Classification/requirements.txt
python Iris_Classification/iris_classification.py
```

This will:
1. Recreate the `dataset/`, `models/`, and `images/` folders if missing.
2. Reload and explore the Iris dataset (prints stats to the console, saves a scatter plot).
3. Retrain the Logistic Regression model and StandardScaler.
4. Print evaluation metrics (accuracy, confusion matrix, classification report).
5. Overwrite `Iris_Classification/models/iris_model.pkl` and `scaler.pkl` with the new artifacts.

Restart the Streamlit app afterward so it picks up the new model (the loader is cached per session).

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| "Model file not found" error in the app | `Iris_Classification/models/iris_model.pkl` is missing | Run the retraining script (see above) or ensure the `models/` folder was committed/deployed. |
| Predictions look wrong/inconsistent | Model and scaler versions mismatched, or scaler skipped | Make sure both `iris_model.pkl` and `scaler.pkl` come from the same training run. |
| App won't start locally | Missing dependencies | Run `pip install -r requirements.txt`. |
| Blank page on first load | Streamlit still compiling/starting | Wait a few seconds and refresh — this is normal on first boot. |

## Credits

- Dataset: [UCI Machine Learning Repository — Iris Data Set](https://archive.ics.uci.edu/dataset/53/iris), via `scikit-learn`.
- Built with [Streamlit](https://streamlit.io) and [scikit-learn](https://scikit-learn.org).
