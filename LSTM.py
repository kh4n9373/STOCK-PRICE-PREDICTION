import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score


def LSTM_implement(df_Stock, your_data):
    data = df_Stock.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.8795)

    scaler= MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_val = []
    y_val = []
    for i in range(training_data_len, len(scaled_data) - 52):
        x_val.append(scaled_data[i - 60:i, 0])
        y_val.append(scaled_data[i, 0])
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=30, epochs=50)

    test_data = scaled_data[training_data_len + 191:, :]
    x_test = []
    y_test = dataset[training_data_len + 251:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_true = y_test
    y_pred = predictions
    explained_variance = r2_score(y_true, y_pred)
    test_explain_variation = explained_variance * 100

    your_data_array = your_data.values
    your_data_scaled = scaler.transform(your_data_array)
    your_data_reshaped = np.reshape(your_data_scaled, (1, 60, 1))
    your_data_prediction = model.predict(your_data_reshaped)
    your_data_prediction = scaler.inverse_transform(your_data_prediction)
    return test_explain_variation, your_data_prediction
