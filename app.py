import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Cybersecurity Anomaly Detection")

# Input mode selector
mode = st.radio("Select input method", ["Upload CSV", "Manual Entry"])

# Function to run model
def detect_anomalies(dataframe):
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    data = dataframe[numeric_cols].dropna()

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(scaled_data)
    dataframe["Anomaly"] = np.where(preds == -1, "Yes", "No")

    return dataframe

# Upload CSV
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your network data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview", df.head())
        result = detect_anomalies(df)
        st.write("🚨 Anomaly Detection Results", result)
        st.download_button("Download Results", result.to_csv(index=False), "anomaly_output.csv")

# Manual Entry
else:
    st.write("🔧 Enter feature values manually")

    # Define feature names manually or dynamically
    feature_names = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent"]

    # Manual input fields
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Detect Anomaly"):
        df_manual = pd.DataFrame([input_data])
        result = detect_anomalies(df_manual)
        st.write("🚨 Anomaly Detection Result", result)
