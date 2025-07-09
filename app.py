import streamlit as st
import pandas as pd
from anomaly_detection_models import load_data, detect_with_isolation_forest, detect_with_autoencoder

st.title("🚨 Network Traffic Anomaly Detector")
st.markdown("Using **Isolation Forest** and **Autoencoder** for unsupervised anomaly detection on network traffic data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Preview of Dataset", data.head())

    method = st.selectbox("Choose Detection Method", ["Isolation Forest", "Autoencoder"])
    if st.button("Run Detection"):
        if method == "Isolation Forest":
            result = detect_with_isolation_forest(data.copy())
        else:
            result = detect_with_autoencoder(data.copy())

        st.success("Anomaly Detection Completed.")
        st.write(result.head())

        st.download_button("Download Result CSV", result.to_csv(index=False), "anomaly_result.csv", "text/csv")
