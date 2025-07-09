import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Network Anomaly Detector", layout="wide")
st.title("🔒 Network Anomaly Detection App")

# Function to highlight anomalies
def highlight_anomalies(row):
    return [
        'background-color: red' if row.get('Anomaly', '') == 'Anomaly' and col == 'Anomaly' else ''
        for col in row.index
    ]

# Function to detect anomalies using Isolation Forest
def detect_anomalies(df):
    df_numeric = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    clf = IsolationForest(contamination=0.2, random_state=42)
    preds = clf.fit_predict(df_scaled)
    df['Anomaly'] = np.where(preds == -1, 'Anomaly', 'Normal')
    return df

# Input options
option = st.radio("📌 Select input method:", ("Upload CSV", "Manual Entry"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Data Preview")
        st.write(data)

        if st.button("🚀 Detect Anomalies"):
            result_df = detect_anomalies(data)
            st.subheader("🚨 Anomaly Detection Result")
            st.dataframe(result_df.style.apply(highlight_anomalies, axis=1))

            # Optional scatter plot
            if 'Anomaly' in result_df.columns and result_df.select_dtypes(include=[np.number]).shape[1] >= 2:
                numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                fig = px.scatter(
                    result_df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color='Anomaly',
                    title="🧯 Anomaly Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.download_button("⬇️ Download Result CSV", result_df.to_csv(index=False), file_name="anomaly_results.csv")

elif option == "Manual Entry":
    st.info("Enter numerical data row below. Columns are: duration, src_bytes, dst_bytes")

    example_cols = ["duration", "src_bytes", "dst_bytes"]
    values = {}
    for col in example_cols:
        values[col] = st.number_input(f"{col}", value=0)

    if st.button("🚀 Detect Anomaly"):
        # Example baseline dataset to compare against
        base_df = pd.DataFrame([
            {"duration": 1, "src_bytes": 100, "dst_bytes": 200},
            {"duration": 2, "src_bytes": 120, "dst_bytes": 220},
            {"duration": 1, "src_bytes": 110, "dst_bytes": 210},
            {"duration": 3, "src_bytes": 130, "dst_bytes": 230},
            {"duration": 2, "src_bytes": 90,  "dst_bytes": 190},
            {"duration": 4, "src_bytes": 115, "dst_bytes": 250}
        ])

        manual_row = pd.DataFrame([values])
        combined_df = pd.concat([base_df, manual_row], ignore_index=True)

        result_df = detect_anomalies(combined_df)

        st.subheader("🚨 Anomaly Detection Result (for your input)")
        st.dataframe(result_df.tail(1).style.apply(highlight_anomalies, axis=1))


        # Optional scatter plot (works if at least 2 numeric features are entered)
        if result_df.select_dtypes(include=[np.number]).shape[1] >= 2:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            fig = px.scatter(
                result_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color='Anomaly',
                title="🧯 Anomaly Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)
