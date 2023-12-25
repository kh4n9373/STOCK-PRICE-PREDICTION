import pandas as pd
from LinearRegression import lr
print('**Initializing Stock Price Analysis Program**')
print('------------------------------------------')

while True:
    name = input("Enter the business code (e.g., AAPL, GOOG, TSLA): ")
    if name.isalpha():  # Ensure code consists of letters only
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
model = lr
final = model.predict(data)
print('\n')
print('Predicted close price :',final[0],' ,Good luck :)')