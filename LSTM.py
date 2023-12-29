import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score
import yfinance as yf

def fetch_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period=period)
        selected_data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        selected_data['Date'] = historical_data.index.date
        return selected_data
    except Exception as e:
        raise ValueError("THERE IS NO SUCH BUSINESS CODE !!") from e

def LSTM_implement(df_Stock, your_data):
    df_Stock['Date'] = pd.to_datetime(df_Stock['Date'])
    df_Stock.set_index('Date', inplace=True)
    
    data = df_Stock[['Open', 'High', 'Low', 'Close']]
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.8795)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, :])
        y_train.append(train_data[i, 3])  # Close price is at index 3

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    x_val = []
    y_val = []
    for i in range(training_data_len, len(scaled_data) - 52):
        x_val.append(scaled_data[i - 60:i, :])
        y_val.append(scaled_data[i, 3])  # Close price is at index 3

    x_val, y_val = np.array(x_val), np.array(y_val)
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=30, epochs=2)

    test_data = scaled_data[training_data_len + 191:, :]
    x_test = []
    y_test = dataset[training_data_len + 251:, 3]  # Close price is at index 3
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, :])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_true = y_test.reshape(-1, 1)
    
    explained_variance = r2_score(y_true, predictions)
    test_explain_variation = explained_variance * 100

    c = your_data['Open'] + your_data['High'] + your_data['Low']
    f = float('inf')
    closest_index = -1

    for i, (m, n, p, _) in enumerate(df_Stock[['Open', 'High', 'Low', 'Close']].values):
        b = float(abs(m + n + p - c))
        if b <= f:
            f = b
            closest_index = i
    df = pd.concat([df_Stock.iloc[:closest_index + 1], your_data, df_Stock.iloc[closest_index + 1:]]).reset_index(drop=True)

    data = df.filter(['Close'])
    dataset = data.values
    room = int(len(dataset) * 0.05)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    if closest_index - room > 60:
        test_data = scaled_data[closest_index - room: closest_index, :]
        x_test = []
        y_test = dataset[closest_index - room + 60: closest_index, 3]  # Close price is at index 3
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, :])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    else:
        test_data = scaled_data[0: closest_index, :]
        x_test = []
        y_test = dataset[int(closest_index / 2): closest_index, 3]  # Close price is at index 3
        for i in range(int(closest_index / 2), len(test_data)):
            x_test.append(test_data[i - int(closest_index / 2):i, :])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    your_data_prediction = predictions[-1][0]

    return test_explain_variation, your_data_prediction

# Example usage:
stock = fetch_stock_data('AAPL', '10y')
data = pd.DataFrame({'Open': [30], 'High': [50], 'Low': [20], 'Volume': [567890]})
result = LSTM_implement(stock, data)
test_explain_variation, your_data_prediction = result
print("Test Explain Variation:", test_explain_variation)
print("Your Data Prediction:", your_data_prediction)
