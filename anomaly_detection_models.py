import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Cybersecurity Anomaly Detector - Isolation Forest")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # Preprocess
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data_numeric = data[numeric_cols]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_numeric)

    # Fit Isolation Forest
    model = IsolationForest(contamination=0.2, random_state=42)
    preds = model.fit_predict(scaled_data)

    # Add anomaly labels
    data["Anomaly"] = np.where(preds == -1, "Yes", "No")

    st.subheader("Anomaly Detection Results")
    st.write(data)

    st.subheader("Anomaly Summary")
    st.write(data["Anomaly"].value_counts())

    st.download_button("Download Results as CSV", data.to_csv(index=False), "anomaly_results.csv")
