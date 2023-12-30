## Hanoi University of Science and Technology 
### Vien cong nghe thong tin va truyen thong


![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/44d8d8e1-bbb2-42b3-93c4-96d80a86dc52)

# Group 1
# Precision Forecasting : A Comprehensive Analysis and Prediction for Stock Closing Price (AI PROJECT)
Guided by Prof. Khoat Than Quang

Contributors:
- Anh Quach Tuan
- Khang Pham Tran Tuan
- Torben Adrian Smid
- Ha Vu Ngoc
- Kien Dinh Van

This repository was created from November 25 2023 to serve the capstone project of the course Introduction to Artificial Intelligent in HUST. 

# How to install and use the Stock Price Predict app ?

0. You should have installed python language on your PC.

1. First you have to download GIT to your PC. If you are using Window, checkout for the GIT official website. If you are using MACOS, open the terminal and install git using command line 'brew install git', else please use 'sudo apt-get update && sudo apt-get install git' for ubuntu system.
   
2. Open up your terminal/cmd, git init and clone this repository to your PC,
~~~
git init
git clone https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION.git
~~~
3. Change the directory to the Stock-Price-Prediction you've just cloned
~~~
cd Stock-Price-Prediction/
~~~
4. Install the requirements.txt so that the app can run properly (on prefered libraries):
~~~
pip install -q -r requirements.txt
~~~
5. Then you are going to run the program. There are 2 options to running our AI agent for you to choose from.
## Option 1 :Run the program using terminal 
Run the python file main_terminal.py
~~~
python main_terminal.py
~~~

Our program should be shown up on your terminal :

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/ec1ceed9-0454-42d8-8a76-e68374ef1fa7)

After you filled the business code, the program start fetching the historical data of that business. (If you typed the name wrong, the program will raise the error).

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/829998b3-533b-47f0-b753-3682641b0588)


Then, there will be the sequence of requests, just the basic things of the stock price today and you need to fill it out. 

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/33e2b91e-0b68-4747-b6dd-a3d811eface1)

After that, choose the Machine Learning model you want, LR, TR, KN stand for linear regression, decision tree, k nearest neighboor respectively.

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/ce6828ba-818d-4614-a651-1e712aef490d)

Wait for second, the program will print out the performance of the model on the historical dataset, and it's predicted stock price

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/3505eff9-5b8b-485d-97fe-d3d08647fdad)


## Option 2: Run the program with webui

First you need to run the python file main_webui.py
~~~
python main_webui.py #num
~~~
#num is the value of the port number you prefer (0000 to 9999 is better), you should based on this value to run the app on the website. If you dont fill in the 'num',the executor will set the num=5000 by default.

The example of running the app successfully:

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/699744f5-7792-4dc8-b245-bccb0d40a61e)


Go up to your search engine (chrome, safari, bing,...) type this to the search bar :

~~~
localhost: #num
~~~

The value num is the value you chose for the port. If you havent filled it yet, so it should be 5000, in the above example, that value is 1234, which means we have to type:

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/4c1908c5-c362-4b72-b9f4-53704e5538f8)


Then, the application should be shown up :

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/53539fd7-1e97-407c-a7ca-4936b95aed55)


Just fill in the necessary attribute of the stock price, click 'Predict' and you will get the answer, with the performance of the algorithm on train set

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/b1defc02-5e8c-45a4-bab4-80d97e2a169b)



## ABOUT THE MARKDOWN FILES:

**Proposed_Approach.md** : This is the old markdown file we created when starting the project. This file documents what we want to do, should do, and will do after discussing with each other, in order to complete the project in the best possible way, based on the criteria set by the teacher for a good project.

**Report.md** : Including what we want to write in the report. In consists of :

- Introduction
- Data Preparation
- Machine Learning Models:
  - LSTM
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine
- Code Components
- Results
- Difficulties
- Libraries

  Like a miniature of this repository :)
  
## ABOUT THE FOLDER:

**DATA** : Where we store our collected data, as well as the place we implemented the data crawling system.
- Files for data collecting :
  - crawling_data.ipynb
    or crawling_data.py
- Acquired stock price historical data :
  - AAPL
  - GOOG
  - MSFT
  - GSPC
  - VFS
    
**MACHINE LEARNING** : Every Machine Learning - Based Algorithm we implemented to find the answer for the close price. Including:
- Decision Tree
- Linear Regression
- K Nearest Neighboor
- Random Forest
- Long Short Term Memory
- Support Vector Machine
In each notebook, we explain clearly the idea of every code snippet to implement Machine Learning models properly.

6 algorithms in total, but we only found 4 of them was appropriate to predicting numerical stock price (You should pay your attetion to Decision Tree, KNN, Linear Regression and LSTM. And because LSTM takes too long time to respond, and the input of the data is very different from Decision Tree, KNN, and Linear Regression, we only take 3 remaining ones for deploying.


**SCRIPT FOR MAKING SLIDES** : 
  We (each member individually) have incorporated our key points into the slides. During the presentation, we will rely on those script lines to articulate our thoughts. Script synthesis is aimed at making the slide creation (handled by one member) more cohesive, instead of having multiple people working on slides simultaneously.


## The architecture of stock price prediction program
![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/7afc522f-1ef6-4c33-87b9-9198baf0d656)

   Following an in-depth exploration and implementation of diverse algorithms for the stock price prediction task, our next step involves conducting experiments with a sample dataset. This dataset encompasses the various attributes associated with a company's stock on a specific day

   The primary goal is to evaluate the performance of each model in predicting the closing price of the company's stock at the conclusion of the same day. This empirical approach allows us to assess the efficacy of the developed models in a real-world context, providing valuable insights into their predictive capabilities for short-term stock price movements.

   **Data Acquisition**: The program starts by fetching historical stock price data from Yahoo Finance (YFINANCE) using the provided business code and date range. This data likely includes attributes like open price, high price, low price, volume, and date.

   **Data Processing**: The retrieved data is then stored in a DataFrame, which is a tabular structure often used in data analysis. This organizes the data into rows and columns, making it easier to manipulate and analyze.Specific attributes relevant to price prediction are chosen from the DataFrame. These might include open price, high price, low price, and volume.

   **Model Selection**:You can choose between three different machine learning models for prediction: Linear Regression (LR), Decision Tree (TR), and K-Nearest Neighbors (KN). Each model has its own strengths and weaknesses, and the choice may depend on the specific characteristics of the chosen stock and historical data. Based on your selection, the app uses the chosen model to analyze the selected data attributes and build a predictive model.

   **Prediction**:The model then uses the learned patterns from the historical data to predict the closing price of the stock for the current day.

   **Output**: The predicted closing price is displayed on the interface.

   **Overall Workflow:** The app essentially follows a common machine learning workflow for price prediction:

   - Gather historical data relevant to the	prediction task.
   - Prepare and clean the data for analysis.
   - Choose and train a machine learning model	based on the selected data.
   - Use the trained model to predict future	outcomes (stock price in this case).
   - Display the prediction.

## Limitations and Considerations
   It's important to keep in mind that stock price prediction is a complex task, and no model can guarantee perfect accuracy. Various factors beyond the scope of the provided data, like news events, economic conditions, and investor sentiment, can also influence stock prices. Therefore, it's crucial to treat predictions as estimations and use them alongside other analysis and considerations before making investment decisions.

## Anyways,
   Thank you for reviewing !! 
   
   Star the project if you appreciate our very first project :)
   
   Although our program is not too good or even it may has a bunch of mistake or mis-understanding (we're just naive students, step-by-step learning and implementing it), this project was built through our collective efforts in research and wholehearted contributions. 
   
   Your encouragement will be a tremendous motivation for us to undertake future projects !
