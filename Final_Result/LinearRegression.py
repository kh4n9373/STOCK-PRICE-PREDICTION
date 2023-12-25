# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams #customizing plot display options
import numpy as np
import seaborn as sns
import os
plt.style.use('fivethirtyeight')

# Loading the dataset
stock = pd.read_csv('/home/khangpt/STOCK-PRICE-PREDICTION/Data/MSFT_historical_data_yfinance.csv',index_col=0)
df_Stock = stock
df_Stock['Date'] = pd.to_datetime(df_Stock['Date'])
df_Stock = df_Stock.reset_index()
df_Stock.set_index('Date',inplace=True)

# Creating training, validation, and test sets
# 88% for training, 10% for validation, 2% for testing
def create_train_test_test(df_Stock):
    features = df_Stock.drop(['Close'],axis=1)
    target = df_Stock['Close']
    
    data_len = df_Stock.shape[0]
    # print('Historical Stock Data length is - ', str(data_len))
    
    #create a chronological split for train and testing
    train_split = int(data_len * 0.88)
    # print('Training Set length - ', str(train_split))

    val_split = train_split + int(data_len * 0.1)
    # print('Validation Set length - ', str(int(data_len * 0.1)))
    
    # print('Test Set length - ', str(int(data_len * 0.02)))
    
    #Splitting features and target into train, validation and test examples
    X_train, X_val, X_test = features[:train_split], features[train_split:val_split], features[val_split:]
    Y_train, Y_val, Y_test = target[:train_split], target[train_split:val_split], target[val_split:]
    
    #Print shape of samples
    # print(X_train.shape, X_val.shape, X_test.shape)
    # print(Y_train.shape, Y_val.shape, Y_test.shape)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


X_train, X_val, X_test, Y_train, Y_val, Y_test = create_train_test_test(df_Stock)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

Y_train_pred = lr.predict(X_train)
Y_val_pred = lr.predict(X_val)
Y_test_pred = lr.predict(X_test)


df_pred = pd.DataFrame(Y_test.values, columns=['Actual'], index=Y_test.index)
df_pred['Predicted'] = Y_test_pred
df_pred = df_pred.reset_index()
df_pred['Date'] = pd.to_datetime(df_pred['Date'], format='%Y-%m-%d')
df_pred



