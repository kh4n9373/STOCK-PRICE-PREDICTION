# STOCK-PRICE-PREDICTION: Our team
AI PROJECT : STOCK PRICE PREDICTION

Anh Quach Tuan

Torben Smid

Khang Pham Tran Tuan

Kien Dinh Van

Ha Vu Ngoc

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#paragraph1)
3. [Machine Learning Models](#paragraph2)
    1. [LSTM](#subparagraph1)
    2. [Linear Regression](#subparagraph2)
    3. [Decision Tree](#subparagraph3)
    4. [Random Forest](#subparagraph4)
4. [Code Components](#paragraph3)
5. [Results](#paragraph4)
6. [Difficultie](#paragraph5)
7. [Libraries](#paragraph6)

## Introduction

Our interest in machine learning, data analytics, and the stock market has led us to choose financial machine learning as the topic for our final project in the course Introduction to AI. Despite our limited prior experience with machine learning, we are fully committed to putting forth our best effort and meticulously documenting the entire process of building the models.

As a collaborative group, we recognize the dynamic nature of the financial industry, which is rapidly evolving and actively seeking innovative ways to leverage machine learning for the effective management of risk and financial losses. Several research efforts have also been carried out to predict the market in order to make profit using different techniques. Our project aims to contribute to this evolution by exploring the intersection of financial analytics and artificial intelligence. 

The adoption of Artificial Neural Network (ANN) techniques and decision tree algorithms has gained significant traction in the business landscape, thanks to its adeptness in modeling relationships among non-linear variables. These sophisticated data mining methodologies allow deeper analysis of large set of data, ecpecially those characterized by rapid fluctuations within short time spans. Consequently, ANN stands out as a promising tool for forecasting stock market dynamics.



## Data Preparation <a name="paragraph1"></a>

We fetch data from Yahoo Finance, an excellent source for reliable stock market movements and prices. Specifically, we used the stock market data from Microsoft, Vinfast, Apple and Google. For each trading day of the stock, a record of each of the following is posted:
-Open = the price when the market opened in the morning.
-Close = the price when the market closed in the afternoon.
-High = the highest price during that trading day.
-Low = the lowest price during that trading day.
-Volume = number of shares of the stock traded that day.
-Adj Close (Adjusted Close) = a price adjusted to account various corporate actions, such as dividends, stock splits, and other events that might affect the stock's price. 

Using yFinance ensures that we have up-to-date and accurate data on stock prices and trading volumes. We then leverage the pandas library for efficient data manipulation and preprocessing, allowing us to handle missing values, normalize data, and create meaningful features. The processed dataset is exported to CSV format, maintaining compatibility with various machine learning frameworks. 


## Machine Learning Models <a name="paragraph2"></a>

Within the field of machine learning (ML), our report on Stock Price Prediction explores the utilization of advanced models to anticipate trends in the stock market. Machine learning, a subset of artificial intelligence, encompasses a range of methodologies, with supervised and unsupervised learning being particularly significant approaches. In supervised learning, a model is trained using a dataset that is labeled, allowing the algorithm to understand the relationship between input features and corresponding output labels. In the context of predicting stock market behavior, supervised learning is applicable as historical stock data can be used as a labeled dataset, enabling the model to learn and generate predictions based on known outcomes.

Within the supervised learning framework, two fundamental types of tasks are regression and classification. Regression involves predicting a continuous outcome, such as stock prices, making it directly applicable to stock market prediction. On the other hand, classification deals with categorizing data into discrete classes such as "Buy", "Sell" and "Hold", a methodology that we applied within the Random Forest Algorithm. This classification approach faciliates the decision-making process by providing actionable insights into potential investment strategies based on the historical behavior of the stock.

The learning process in machine learning has three main elements that are essential; these are the target function, training set and testing set. 
-The training set is used to construct model learning by revealing the patterns and relationship it may contain. 
-The target function explains how the inputs correlate to the outputs. 
-The testing evaluates the model and consists of unseen data for gauging its generalization ability.


### LSTM <a name="subparagraph1"></a>

#### How it Works

#### Why We Use It

### Linear Regression <a name="subparagraph2"></a>

#### How it Works

#### Why We Use It

### Decision Tree <a name="subparagraph3"></a>

#### How it Works

The Decision Tree algorithm employed in our stock market prediction project operates as a sophisticated decision-making process. It effectively analyzes historical stock data by posing a series of inquiries regarding market conditions, such as the comparison between the closing price and the previous day's closing price, or the presence of an upward trend in the stock. By answering these inquiries, the algorithm ultimately generates a prediction for the movement of tomorrow's stock price.

#### Why We Use It

Decision Trees offer an intuitive and straightforward approach to decision-making, providing a clear and logical path. Their simplicity allows for easy interpretation and understanding of the decision-making process. Although Decision Trees can occasionally become overly detailed, leading to predictions influenced by irrelevant data, they still hold immense value. In fact, Decision Trees serve as a fundamental element in our strategy, acting as the building blocks for more sophisticated ensemble methods such as Random Forests. 

### Random Forest <a name="subparagraph4"></a>

#### How it Works
The Random Forest algorithm enhances decision-making by constructing a "forest" of Decision Trees. Each tree is trained on a random subset of historical stock data, introducing diversity in the learning process. When it is time to predict the stock price for the next day, each tree in the forest provides its prediction, and the final decision is made through a voting process. This collective approach helps mitigate individual tree biases and typically yields more accurate and reliable predictions.

#### Why We Use Buy, Sell, or Hold Predictions:
The reason we utilize the "Buy," "Sell," or "Hold" prediction approach in our Random Forest implementation is to determine whether it is advisable to buy, sell, or hold the stock based on the ensemble decision of the forest. This three-class prediction approach adds practicality to our strategy, enabling us to anticipate not only upward or downward movements but also suggest holding the stock when there is no clear indication of a significant change. This aligns with real-world trading decisions and provides actionable insights for users.

## Code Components <a name="paragraph3"></a>

## Results <a name="paragraph4"></a>

## Difficulties <a name="paragraph5"></a>

## Libraries <a name="paragraph6"></a>

1. Pandas (The Pandas development team, publisher Zendoo, version latest)
2. Numpy (Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke & Travis E. Oliphant, published 09.x.2020)
3. yfinance (yfinance is a community effort, data belongs to Yahoo)
4. Sklearn (Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.,published 2011)
5. Matplotlib (Hunter , J. D., published 2007)

