# You can run our entire program using terminal on this file.
import pandas as pd
from read_the_data import read_the_data
from crawling_data import fetch_stock_data
from LinReg import Lr_implement
from KNN import KNN_implement
from Tree import Tree_implement
import sys
import time

def loading_animation():
    chars = ['.', '..', '...', '....']
    for _ in range(10):  
        for char in chars:
            sys.stdout.write(f'\rFetching{char}')
            sys.stdout.flush()
            time.sleep(0.3) 

print('**Initializing Stock Price Analysis Program**')
print('------------------------------------------')

while True:
    name = input("Enter the business code (e.g., AAPL, GOOG, TSLA): ")
    if name.isalpha(): 
        loading_animation()
        stock_data = fetch_stock_data(name, '10y')
        print('\n')
        print('Finished !')
        break
    else:
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

model = input('\nChoose Your Models (LR), (TR), (KN): ').upper()
time.sleep(3)
print('\n')
if model == 'LR':
    accu, final = Lr_implement(stock_data, data)
elif model == 'TR':
    accu, final = Tree_implement(stock_data, data)
elif model == 'KN':
    accu, final = KNN_implement(stock_data, data)
else:
    print("Invalid input. Please choose LR, TR, or KN.")

print('\n')
print(f'Performance of {model}:', accu)
print('Predicted close price :', final[0])
print('\n----------------------------------------------')

sys.exit()