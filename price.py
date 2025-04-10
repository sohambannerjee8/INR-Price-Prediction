import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="USD/INR Forecast", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
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

def plot_results(history, test_data, test_predict, future_dates, future_predict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Training loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Training Progress')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Price predictions plot
    ax2.plot(test_data.index, test_data['Close'], label='Actual Price')
    ax2.plot(test_data.index, test_predict, label='Test Predictions')
    ax2.plot(future_dates, future_predict, label='Future Predictions')
    ax2.set_title('USD/INR Exchange Rate Forecast')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Exchange Rate')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def main():
    st.title("USD/INR Exchange Rate Forecasting using LSTM")
    
    # File upload
    uploaded_file = st.file_uploader("Upload historical data (CSV)", type="csv")
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        # Show raw data
        st.subheader("Historical Data Preview")
        st.dataframe(data.tail(), use_container_width=True)
        
        # Parameters
        look_back = st.sidebar.slider("Lookback Window", 30, 90, 60)
        epochs = st.sidebar.slider("Epochs", 10, 100, 50)
        train_size = int(len(data) * 0.8)
        
        if st.button("Train Model and Generate Forecast"):
            with st.spinner('Training model...'):
                # Prepare data
                train_data = data[:train_size]
                test_data = data[train_size:]
                
                # Prepare sequences
                X_train, y_train, scaler = prepare_lstm_data(train_data, look_back)
                X_test, y_test, _ = prepare_lstm_data(data, look_back)
                X_test = X_test[train_size - look_back:]
                
                # Build and train model
                model = build_lstm_model((look_back, 1))
                history = model.fit(X_train, y_train, 
                                  epochs=epochs, 
                                  batch_size=32,
                                  validation_split=0.1,
                                  verbose=0)
                
                # Test predictions
                test_predict = model.predict(X_test)
                test_predict = scaler.inverse_transform(test_predict)
                test_data = test_data.iloc[look_back:]
                
                # Error metrics
                mse = mean_squared_error(test_data['Close'], test_predict)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_data['Close'], test_predict)
                
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
                
                # Show results
                st.success("Model training completed!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")
                
                # Plots
                fig = plot_results(history, test_data, test_predict, future_dates, future_predict)
                st.pyplot(fig)
                
                # Future predictions table
                st.subheader("1-Year Forecast")
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Rate': future_predict.flatten()
                })
                st.dataframe(future_df.set_index('Date'), use_container_width=True)

if __name__ == "__main__":
    main()
