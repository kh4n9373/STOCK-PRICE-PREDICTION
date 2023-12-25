from crawling_data import fetch_stock_data, save_to_csv, fetch_and_save_data
import datetime
# Use the functions
symbol = "AMZN"
period = "10y"
stock_data = fetch_stock_data(symbol, period)
save_to_csv(stock_data, symbol)

series_name = "UNRATE"
start_date = datetime(2010, 1, 1)
end_date = datetime(2023, 1, 1)
fetch_and_save_data(series_name, start_date, end_date)
