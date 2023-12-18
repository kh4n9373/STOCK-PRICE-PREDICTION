#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing yfinance api
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

api_key = "143c9a19a5a05044020a48c6de25ad81"

# Set the data source as "fred" 
pdr.fred.FredReader.api_key = api_key


# In[8]:


#crawling data from yFinance

# def fetch_stock_data(symbol, period):
#     stock = yf.Ticker(symbol)
#     historical_data = stock.history(period=period)
#     selected_data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']]
#     selected_data['Date'] = historical_data.index.date
#     return selected_data

def fetch_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    historical_data = stock.history(period=period)
    selected_data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    selected_data['Date'] = historical_data.index.date
    return selected_data


# In[4]:


#saving the data into csv
def save_to_csv(data, symbol):
    file_name = f"{symbol}_historical_data_yfinance.csv"
    data.to_csv(file_name, index=False)


# In[9]:


#changing symbol (stock name) and period to get the csv file with the data
symbol = "MSFT"
period = "10y"
stock_data = fetch_stock_data(symbol,period)
save_to_csv(stock_data, symbol)


# In[5]:


#crawling data from FRED
def fetch_and_save_data(series_name, start_date, end_date):
    data = pdr.get_data_fred(series_name, start_date, end_date)
    file_name = f"{series_name}_data.csv"
    data.to_csv(file_name)



# In[6]:


#changing series name, start and end date to get the desired data in a csv file
series_name = "GDP"
start_date = datetime(2013, 11, 1)
end_date = datetime(2023, 11, 1)

fetch_and_save_data(series_name, start_date, end_date)

