
# anomaly_detection_models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("cybersecurity_attacks.csv")

# Drop irrelevant or high-cardinality text fields
drop_cols = ['Timestamp', 'Source IP Address', 'Destination IP Address',
             'Payload Data', 'Malware Indicators', 'Alerts/Warnings',
             'Attack Signature', 'User Information', 'Device Information',
             'Geo-location Data', 'Proxy Information', 'Firewall Logs',
             'IDS/IPS Alerts', 'Log Source']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Fill missing values
df.fillna(-1, inplace=True)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['Attack Type'], errors='ignore'))

# --- 1. Isolation Forest ---
iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_preds = iso_model.fit_predict(X)
df['IsolationForest_Anomaly'] = np.where(iso_preds == -1, 'Anomaly', 'Normal')

# --- 2. Autoencoder ---
input_dim = X.shape[1]
encoding_dim = 16

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train/test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_data=(X_test, X_test), verbose=1)

# Compute reconstruction error
X_pred = autoencoder.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=1)

threshold = np.percentile(mse, 95)
df['Autoencoder_Anomaly'] = np.where(mse > threshold, 'Anomaly', 'Normal')

# Show result counts
print("Isolation Forest Results:")
print(df['IsolationForest_Anomaly'].value_counts())

print("\nAutoencoder Results:")
print(df['Autoencoder_Anomaly'].value_counts())
