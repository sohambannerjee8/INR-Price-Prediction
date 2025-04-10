import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="USD/INR Auto Forecast", layout="wide")
st.title("Automated USD/INR Exchange Rate Forecasting")

# Custom styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('HistoricalData.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data = data.sort_values('Date').set_index('Date')[['Close/Last']]
    data = data.rename(columns={'Close/Last': 'Close'})
    return data

def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, epochs):
        self.progress_bar = progress_bar
        self.epochs = epochs
        
    def on_epoch_end(self, epoch, logs=None):
        progress = 10 + int(90 * (epoch+1)/self.epochs)
        self.progress_bar.progress(progress)

def main():
    # Load data automatically
    try:
        data = load_data()
    except FileNotFoundError:
        st.error("HistoricalData.csv file not found in the working directory")
        return
    
    # Show data preview
    st.subheader("Historical Data Overview")
    st.dataframe(data.tail(10), use_container_width=True)
    
    # Fixed parameters
    look_back = 60
    epochs = 50
    train_size = int(len(data) * 0.8)
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare data
    status_text.text("Preparing training data...")
    train_data = data[:train_size]
    
    # Create test sequences using last look_back days of train + test data
    test_sequence_data = pd.concat([train_data[-look_back:], data[train_size:]])
    X_test, y_test, scaler = prepare_lstm_data(test_sequence_data, look_back)
    
    X_train, y_train, _ = prepare_lstm_data(train_data, look_back)
    
    # Build and train model
    status_text.text("Building LSTM model...")
    model = build_lstm_model((look_back, 1))
    
    status_text.text(f"Training model ({epochs} epochs)...")
    progress_bar.progress(10)
    
    # Train with progress updates
    history = model.fit(X_train, y_train,
                       epochs=epochs,
                       batch_size=32,
                       verbose=0,
                       callbacks=[TrainingCallback(progress_bar, epochs)])
    
    # Generate predictions
    status_text.text("Generating predictions...")
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)
    
    # Align test data with predictions
    test_data = data[train_size:][:len(test_predict)]
    
    # Future predictions
    last_sequence = scaler.transform(data[-look_back:][['Close']].values)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), 
                               periods=365, 
                               freq='D')
    future_predict = []
    current_sequence = last_sequence.copy()
    
    for _ in range(len(future_dates)):
        pred = model.predict(current_sequence.reshape(1, look_back, 1), verbose=0)
        future_predict.append(pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred
    
    future_predict = scaler.inverse_transform(np.array(future_predict).reshape(-1, 1))
    
    # Display results
    progress_bar.empty()
    status_text.empty()
    st.success("Analysis complete!")
    
    # Metrics
    mse = mean_squared_error(test_data['Close'], test_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data['Close'], test_predict)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mse:.4f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
    col3.metric("Mean Absolute Error", f"{mae:.4f}")
    
    # Plots
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data.index, data['Close'], label='Historical Data')
    ax.plot(test_data.index, test_predict, label='Model Predictions')
    ax.plot(future_dates, future_predict, label='1-Year Forecast')
    ax.set_title("USD/INR Exchange Rate Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exchange Rate")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Forecast table
    st.subheader("Detailed 1-Year Forecast")
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Rate': future_predict.flatten()
    })
    st.dataframe(forecast_df.set_index('Date'), use_container_width=True)

if __name__ == "__main__":
    main()
