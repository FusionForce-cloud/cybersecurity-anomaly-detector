import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense

def load_data(path):
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number]).dropna()
    return df

def detect_with_isolation_forest(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(scaled_data)
    data['anomaly'] = np.where(preds == -1, 1, 0)
    return data

def detect_with_autoencoder(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    x_train, x_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(x_train.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, x_train, epochs=20, batch_size=32, validation_data=(x_test, x_test), verbose=0)

    preds = model.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - preds, 2), axis=1)
    threshold = np.percentile(mse, 95)
    anomalies = (mse > threshold).astype(int)
    data['anomaly'] = anomalies
    return data
