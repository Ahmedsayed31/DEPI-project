from predict import predict
import streamlit as st
import pandas as pd
import joblib
import yaml
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.graph_objects as go

# ========= Load config ==========
def read_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


# ========= Streamlit App ==========
st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📊 Sales Forecast Dashboard")

uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with st.spinner("Processing and predicting..."):
        rmse, r2, fig = predict(df)

    # Metrics Cards
    col1, col2 = st.columns(2)
    col1.metric("📉 RMSE", f"{rmse:.2f}")
    col2.metric("🎯 R² Score", f"{r2:.3f}")

    # Forecast Plot
    st.plotly_chart(fig, use_container_width=True)
