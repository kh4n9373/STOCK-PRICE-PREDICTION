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

This repository was created from November 25 2023 to serve the capstone project or the cource Introduction to Artificial Intelligent in HUST. 

# How to install and use the Stock Price Predict app ?

1. First you have to download GIT to your PC. If you are using Window, checkout for the GIT official website. If you are using MACOS, open the terminal and install git using command line 'brew install git', else please use 'sudo apt-get update && sudo apt-get install git' for ubuntu system.
2. Open up your terminal/cmd, git init and clone this repository to your PC,
~~~
git init
git clone https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION.git
~~~
3. Install the requirements.txt so that the app can run properly (on desired libraries):
~~~
pip install -q -r requirements.txt
~~~
4. Change the directory to the Stock-Price-Prediction you've just cloned
~~~
cd Stock-Price-Prediction/
~~~
5. Then run the app by using this command line :
~~~
python main.py #num
~~~
#num is the value of the port number you prefer ranging from 0000 to 9999, you should based on this value to run the app on the website. If you dont fill in the 'num',the executor will set the num=5000 by default.

The example of running the app successfully:

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/67f3acbe-ec89-4988-aca7-b4271321ac35)


6. Go up to your search engine (chrome, safari, bing,...) type this to the search bar :

~~~
localhost: #num
~~~

The value num is the value you chose for the port else it would be 5000, in the above example, that value is 1234, which means we have to type:

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/4c1908c5-c362-4b72-b9f4-53704e5538f8)


Then, the application should be shown up :

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/734a22ee-a3d0-475c-b7bb-6ce257b9e2d9)

7. Just fill the necessary attribute of the stock price, click 'Predict' and you will get the answer, with the performance of the algorithm on train set

![image](https://github.com/ktuanPT373/STOCK-PRICE-PREDICTION/assets/112315454/c2923d5b-bf4e-4a88-9dad-fee0afed1e27)


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
  
6 algorithms in total, but we only found 4 of them was appropriate to predicting numerical stock price (You should pay your attetion to Decision Tree, KNN, Linear Regression and LSTM

**SCRIPT FOR MAKING SLIDES** : 
  We (each member individually) have incorporated our key points into the slides. During the presentation, we will rely on those script lines to articulate our thoughts. Script synthesis is aimed at making the slide creation (handled by one member) more cohesive, instead of having multiple people working on slides simultaneously.


  

