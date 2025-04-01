#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys  
get_ipython().system('{sys.executable} -m pip install --user pandas numpy scikit-learn tensorflow matplotlib')


# In[24]:


import sys  
get_ipython().system('{sys.executable} -m pip install --user prophet')
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    # Use Close/Last price for analysis
    df = df[['Close/Last']].rename(columns={'Close/Last': 'Close'})
    
    return df

# 2. ARIMA Model
def fit_arima_model(data):
    # Fit ARIMA model (p,d,q) = (5,1,0) based on common financial time series parameters
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()
    return model_fit

# 3. Prophet Model for long-term prediction
def fit_prophet_model(data):
    # Prepare data for Prophet
    prophet_df = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(prophet_df)
    return model

# 4. Generate future predictions
def generate_predictions(arima_model, prophet_model, data, end_date='2026-12-31'):
    # Short-term ARIMA forecast (90 days)
    arima_forecast = arima_model.forecast(steps=90)
    
    # Long-term Prophet forecast
    future_dates = prophet_model.make_future_dataframe(periods=365*4, freq='D')  # 4 years ahead
    prophet_forecast = prophet_model.predict(future_dates)
    
    return arima_forecast, prophet_forecast

# 5. Visualize results
def plot_results(data, arima_forecast, prophet_forecast):
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(data.index, data['Close'], label='Historical Data', color='blue')
    
    # Plot ARIMA forecast
    last_date = data.index[-1]
    arima_dates = pd.date_range(start=last_date, periods=90, freq='D')
    plt.plot(arima_dates, arima_forecast, label='ARIMA Forecast (90 days)', color='red')
    
    # Plot Prophet forecast
    plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet Forecast (to 2026)', color='green')
    plt.fill_between(prophet_forecast['ds'], 
                    prophet_forecast['yhat_lower'], 
                    prophet_forecast['yhat_upper'], 
                    color='green', 
                    alpha=0.2)
    
    plt.title('USD/INR Exchange Rate Prediction')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
def main():
    # File path
    file_path = 'HistoricalData.csv'
    
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    
    # Fit models
    arima_model = fit_arima_model(data)
    prophet_model = fit_prophet_model(data)
    
    # Generate predictions
    arima_forecast, prophet_forecast = generate_predictions(arima_model, prophet_model, data)
    
    # Plot results
    plot_results(data, arima_forecast, prophet_forecast)
    
    # Print sample predictions
    print("\nSample ARIMA Predictions (next 5 days):")
    print(arima_forecast[:5])
    
    print("\nSample Prophet Predictions (Yearly intervals until 2026):")
    yearly_predictions = prophet_forecast[prophet_forecast['ds'].dt.month == 1][
        prophet_forecast['ds'].dt.day == 1][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print(yearly_predictions[yearly_predictions['ds'] > '2023-03-01'].head(5))

if __name__ == "__main__":
    # Required libraries
    try:
        main()
    except ImportError as e:
        print(f"Required library missing: {e}")
        print("Please install required libraries using:")
        print("pip install pandas numpy statsmodels prophet matplotlib")


# In[36]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

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
    # Load data
    file_path = 'HistoricalData.csv'
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data = data.sort_values('Date').set_index('Date')[['Close/Last']].rename(columns={'Close/Last': 'Close'})
    
    # Train-test split
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Prepare data for LSTM
    look_back = 60
    X_train, y_train, scaler = prepare_lstm_data(train_data, look_back)
    X_test, y_test, _ = prepare_lstm_data(pd.concat([train_data[-look_back:], test_data]), look_back)
    
    # Build and train model
    model = build_lstm_model((look_back, 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Test predictions
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Error metrics
    mse = mean_squared_error(y_test_inv, test_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, test_predict)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Forecast until end of 2024
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
    
    # Create DataFrame with predictions
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predict.flatten()})
    
    # Filter for monthly predictions (1st of each month in 2024)
    monthly_predictions_2024 = future_df[
        (future_df['Date'].dt.year == 2024) & (future_df['Date'].dt.day == 1)
    ]
    
    # Print monthly predictions for 2024
    print("\nUSD/INR Exchange Rate Predictions for 2024 (First of Each Month):")
    print(monthly_predictions_2024[['Date', 'Predicted_Close']].to_string(index=False))
    
    # Plot results
    plt.figure(figsize=(15, 8))
    plt.plot(data.index, data['Close'], label='Historical Data')
    plt.plot(test_data.index, test_predict, label='Test Prediction')
    plt.plot(future_dates, future_predict, label='Future Prediction')
    plt.title('USD/INR Exchange_todo: Add title here')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


# In[ ]:




