# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams #customizing plot display options
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Loading the dataset
stock = pd.read_csv('/home/khangpt/STOCK-PRICE-PREDICTION/Data/MSFT_historical_data_yfinance.csv',index_col=0)
df_Stock = stock
df_Stock['Date'] = pd.to_datetime(df_Stock['Date'])
df_Stock = df_Stock.reset_index()
df_Stock.set_index('Date',inplace=True)

# Creating training, validation, and test sets
# 88% for training, 10% for validation, 2% for testing
def create_train_test_set(df_Stock):
    features = df_Stock.drop(['Close'],axis=1)
    target = df_Stock['Close']
    
    data_len = df_Stock.shape[0]
    # print('Historical Stock Data length is - ', str(data_len))
    
    #create a chronological split for train and testing
    train_split = int(data_len * 0.8)
    # print('Training Set length - ', str(train_split))
    # print('Test Set length - ', str(int(data_len * 0.2)))
    
    #Splitting features and target into train, validation and test examples
    X_train,X_test = features[:train_split], features[train_split+1:]
    Y_train,Y_test = target[:train_split], target[train_split+1:]
    
    #Print shape of samples
    # print(X_train.shape, X_val.shape, X_test.shape)
    # print(Y_train.shape, Y_val.shape, Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test

def implement(df_Stock,your_data):
    X_train, X_test, Y_train, y_test = create_train_test_set(df_Stock)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    y_pred = lr.predict(X_test)
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    accuracy = np.mean(np.abs((y_pred - y_pred) / y_test)) * 100

    print(f'The accuracy score: {accuracy:.2%}')

    return lr.predict(your_data)
    



