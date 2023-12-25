import pandas as pd
from read_the_data import read_the_data
from crawling_data import fetch_stock_data
from LinearRegression import implement
import datetime
import sys
import time

def loading_animation():
    chars = ['.', '..', '...', '....']
    for _ in range(10):  # Adjust the number of iterations based on your preference
        for char in chars:
            sys.stdout.write(f'\rFetching{char}')
            sys.stdout.flush()
            time.sleep(0.3)  # Adjust the sleep duration for the desired speed

print('**Initializing Stock Price Analysis Program**')
print('------------------------------------------')

while True:
    name = input("Enter the business code (e.g., AAPL, GOOG, TSLA): ")
    if name.isalpha():  # Ensure code consists of letters only
        loading_animation()
        stock_data = fetch_stock_data(name, '10y')
        print('\n')
        print('Finished !')
        break
    print("Invalid business code. Please enter letters only.")

print('\n')

while True:
    try:
        opun = float(input("Enter today's open price: "))
        high = float(input("Enter today's high price: "))
        low = float(input("Enter today's low price: "))
        volume = float(input("Enter today's trading volume: "))
        break
    except ValueError:
        print("Invalid input. Please enter numbers only.")

if high < low:
    print("Error: High price cannot be lower than low price. Please try again.")

data = pd.DataFrame()
data['Open'] = [opun]
data['High'] = [high]
data['Low'] = [low]
data['Volume'] = [volume]

read_the_data(stock_data)
time.sleep(3)
final = implement(stock_data, data)

print('\n')
print('Predicted close price :', final[0])
print('\n----------------------Good luck :)------------------------')

sys.exit()