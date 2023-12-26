import pandas as pd
def read_the_data(df_Stock):
    df_Stock['Date'] = pd.to_datetime(df_Stock['Date'])
    df_Stock.set_index('Date',inplace=True)
