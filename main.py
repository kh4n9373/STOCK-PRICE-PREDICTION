from flask import Flask, render_template, request
import pandas as pd
from read_the_data import read_the_data
from crawling_data import fetch_stock_data
from LinReg import Lr_implement
from KNN import KNN_implement
from Tree import Tree_implement

import sys
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    opun = float(request.form['opun'])
    high = float(request.form['high'])
    low = float(request.form['low'])
    volume = float(request.form['volume'])

    stock_data = fetch_stock_data(name, '10y')

    data = pd.DataFrame({'Open': [opun], 'High': [high], 'Low': [low], 'Volume': [volume]})

    read_the_data(stock_data)

    model = request.form['model'].upper()
    time.sleep(3)

    if model == 'LR':
        accu,final = Lr_implement(stock_data, data)
    elif model == 'TR':
        accu,final = Tree_implement(stock_data, data)
    elif model == 'KN':
        accu,final = KNN_implement(stock_data, data)
    # elif model =='LSTM':
    #     accu,final = LSTM_implement(stock_data, data)
    else:
        return "Invalid input. Please choose LR, TR, or KN."

    if final[0] < opun:
        return render_template('result.html', model=model,accu=round(accu*100,3), predict='DOWN', final=round(final[0],4))
    elif final[0] > opun:
        return render_template('result.html', model=model,accu=round(accu*100,3), predict='UP', final=round(final[0],4))

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(port=port,debug=True)