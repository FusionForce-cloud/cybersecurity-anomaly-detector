import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from io import StringIO

# -------------------- Page Config & Title --------------------
st.set_page_config(page_title="Cybersecurity Anomaly Detector", layout="wide")
st.title("🔐 Cybersecurity Anomaly Detector")
st.markdown("Detect anomalies in network traffic using Isolation Forest.")

# -------------------- Sidebar Model Choice --------------------
model_choice = st.sidebar.selectbox("Choose Anomaly Detection Model", ["Isolation Forest"])

# -------------------- Function Definitions --------------------
def detect_with_isolation_forest(df):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = model.fit_predict(df)
    df['Anomaly'] = np.where(preds == -1, 'Anomaly', 'Normal')
    return df

def highlight_anomalies(df):
    return ['background-color: red' if val == 'Anomaly' else '' for val in df['Anomaly']]

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 CSV Upload"])

# -------------------- Manual Entry --------------------
with tab1:
    st.subheader("Enter Network Data Manually")
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Duration", min_value=0.0, value=10.0)
        src_bytes = st.number_input("Source Bytes", min_value=0.0, value=100.0)
    with col2:
        dst_bytes = st.number_input("Destination Bytes", min_value=0.0, value=50.0)
        count = st.number_input("Count", min_value=0.0, value=5.0)

    if st.button("Detect Anomaly", key="manual"):
        input_df = pd.DataFrame([[duration, src_bytes, dst_bytes, count]], 
                                columns=['duration', 'src_bytes', 'dst_bytes', 'count'])
        result_df = detect_with_isolation_forest(input_df.copy())
        st.dataframe(result_df.style.apply(highlight_anomalies, axis=1))
        fig = px.histogram(result_df, x='Anomaly', title="Anomaly Count")
        st.plotly_chart(fig)

# -------------------- CSV Upload --------------------
with tab2:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Detect Anomaly", key="csv"):
            result_df = detect_with_isolation_forest(df.copy())
            st.subheader("Detection Results")
            st.dataframe(result_df.style.apply(highlight_anomalies, axis=1))

            fig = px.histogram(result_df, x='Anomaly', title="Anomaly Distribution")
            st.plotly_chart(fig)

            # Download button
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv_data, "anomaly_results.csv", "text/csv")
