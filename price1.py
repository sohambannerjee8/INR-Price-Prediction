import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import streamlit as st

def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    st.title("USD/INR Exchange Rate Prediction")
    
    # Load data
    try:
        data = pd.read_csv('HistoricalData.csv')  # Assumes CSV is in repo root
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data = data.sort_values('Date').set_index('Date')[['Close/Last']].rename(columns={'Close/Last': 'Close'})
    except FileNotFoundError:
        st.error("HistoricalData.csv not found in repository!")
        return
    
    # Train-test split
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Prepare data
    look_back = 60
    X_train, y_train, scaler = prepare_lstm_data(train_data, look_back)
    X_test, y_test, _ = prepare_lstm_data(pd.concat([train_data[-look_back:], test_data]), look_back)
    
    # Build and train model
    model = build_lstm_model((look_back, 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Test predictions
    test_predict = model.predict(X_test, verbose=0)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Error metrics
    mse = mean_squared_error(y_test_inv, test_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, test_predict)
    st.write(f"Test MSE: {mse:.4f}")
    st.write(f"Test RMSE: {rmse:.4f}")
    st.write(f"Test MAE: {mae:.4f}")
    
    # Forecast until 2024
    last_sequence = scaler.transform(data[-look_back:][['Close']].values)
    future_dates = pd.date_range(start=data.index[-1], end='2024-12-31', freq='D')
    future_predict = []
    current_sequence = last_sequence.copy()
    
    for _ in range(len(future_dates)):
        pred = model.predict(current_sequence.reshape(1, look_back, 1), verbose=0)
        future_predict.append(pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred
    
    future_predict = scaler.inverse_transform(np.array(future_predict).reshape(-1, 1))
    
    # Monthly predictions for 2024
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predict.flatten()})
    monthly_predictions_2024 = future_df[
        (future_df['Date'].dt.year == 2024) & (future_df['Date'].dt.day == 1)
    ]
    st.write("\nUSD/INR Predictions for 2024 (First of Each Month):")
    st.dataframe(monthly_predictions_2024[['Date', 'Predicted_Close']])
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(data.index, data['Close'], label='Historical Data')
    ax.plot(test_data.index, test_predict, label='Test Prediction')
    ax.plot(future_dates, future_predict, label='Future Prediction')
    ax.set_title('USD/INR Exchange Rate Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
