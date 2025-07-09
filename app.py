import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Page config
st.set_page_config(page_title="Cybersecurity Anomaly Detector", layout="wide")
st.title("🔒 Cybersecurity Anomaly Detection App")

# Function to highlight anomalies
def highlight_anomalies(row):
    return [
        'background-color: red' if row.get('Anomaly', '') == 'Anomaly' and col == 'Anomaly' else ''
        for col in row.index
    ]

# Function to detect anomalies using Isolation Forest
def detect_anomalies(df):
    df_numeric = df.select_dtypes(include=[np.number])
    clf = IsolationForest(contamination=0.1, random_state=42)
    preds = clf.fit_predict(df_numeric)
    df['Anomaly'] = np.where(preds == -1, 'Anomaly', 'Normal')
    return df

# Upload CSV or manual entry
option = st.radio("Select input method:", ("Upload CSV", "Manual Entry"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Data Preview")
        st.write(data)

        if st.button("🚀 Detect Anomalies"):
            result_df = detect_anomalies(data)
            st.subheader("🚨 Anomaly Detection Result")
            st.dataframe(result_df.style.apply(highlight_anomalies, axis=1))

            # Optional visualization
            if 'Anomaly' in result_df.columns and result_df.select_dtypes(include=[np.number]).shape[1] >= 2:
                numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                fig = px.scatter(
                    result_df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color='Anomaly',
                    title="Anomaly Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)

elif option == "Manual Entry":
    st.info("Enter numerical data row below. Column names must match.")
    example_cols = ["duration", "src_bytes", "dst_bytes"]
    values = {}
    for col in example_cols:
        values[col] = st.number_input(f"{col}", value=0)

    if st.button("🚀 Detect Anomaly"):
        df_manual = pd.DataFrame([values])
        result_df = detect_anomalies(df_manual)
        st.subheader("🚨 Anomaly Detection Result")
        st.dataframe(result_df.style.apply(highlight_anomalies, axis=1))
