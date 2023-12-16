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
    5. [Support Vector Machine](#subparagraph5)
4. [Code Components](#paragraph3)
5. [Results](#paragraph4)
6. [Difficultie](#paragraph5)
7. [Libraries](#paragraph6)

## Introduction

Our interest in machine learning, data analytics, and the stock market has led us to choose financial machine learning as the topic for our final project in the course Introduction to AI. Despite our limited prior experience with machine learning, we are fully committed to putting forth our best effort and meticulously documenting the entire process of building the models.

As a collaborative group, we recognize the dynamic nature of the financial industry, which is rapidly evolving and actively seeking innovative ways to leverage machine learning for the effective management of risk and financial losses. Our project aims to contribute to this evolution by exploring the intersection of financial analytics and artificial intelligence.

## Data Preparation <a name="paragraph1"></a>

We fetch data from Yahoo Finance, an excellent source for reliable stock market movements and prices. This ensures we have up-to-date and accurate data on stock prices and trading volumes. We then leverage the pandas library for efficient data manipulation and preprocessing, allowing us to handle missing values, normalize data, and create meaningful features. The processed dataset is exported to CSV format, maintaining compatibility with various machine learning frameworks. 

## Machine Learning Models <a name="paragraph2"></a>

### LSTM <a name="subparagraph1"></a>

#### How it Works

Long Short Term Memory (LSTM) - is a model that increases the memory of Recurrent Neural Networks (RNN). The core idea behind LSTM is the use of memory cells. These cells can store information for long durations, and their state can be selectively updated or cleared. To control the flow of information into and out of the memory cell, the algorithm uses three distinct gates: Input gate (manages the amount of new information that enters the memory cell), Forget gate (controls the removal of information from the memory cell) and Output gate (determines the information to be output based on the current cell state). About cell state, it is the internal memory of the LSTM which runs along the entire sequence and is modified by the gates at each time step. Ultimately, LSTM is trained using backpropagation through time (BPTT), similar to other RNNs with the aim to minimize the difference between the predicted output and the actual output.

#### Why We Use It

In comparison with traditional RNNS, LSTM is a modified version to cope with the vanishing gradient problem, which is caused by the repeated use of the same parameters in RNN blocks, at each step. To do so, LSTM leverages gating mechanisms to control the flow of information and gradients. This helps prevent the vanishing gradient problem and allows the network to learn and retain information over longer sequences. As a result, LSTM is more effective at predicting time-series patterns, especially stock prices.

### Linear Regression <a name="subparagraph2"></a>

#### How it Works

Linear regression works by modeling the relationship between a dependent variable (e.g., stock price) and one or more independent variables through a linear equation. The model aims to find the best-fitting line that minimizes the difference between predicted and actual values in historical data. The equation takes the form Y = b0 + b1∗X + ϵ, where Y is the dependent variable, X is the independent variable, b0 is the y-intercept, b1 is the slope, and ϵ represents the error term. Training the model involves determining the coefficients b0 and B1 to create a predictive formula, enabling the estimation of future stock prices based on new input values. 

#### Why We Use It

Linear regression is employed in stock price prediction for several reasons. First, it offers interpretability, allowing analysts to understand and quantify the impact of independent variables on stock prices through the coefficients in the linear equation. Second, it provides a quick and straightforward implementation, making it a practical choice for initial analyses and as a baseline model for comparison. While linear regression has its limitations, such as assuming a linear relationship between variables, it remains valuable for its simplicity, ease of interpretation, and as a starting point for more complex modeling approaches in the domain of stock price prediction. 
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

### Support Vector Machine <a name="subparagraph5"></a>

#### How it Works

The goal of Support Vector Machine Regression(SVR) is to find a function that predicts the continuous output variable (target) based on input features while minimizing the prediction error. SVR does this by identifying a hyperplane that best represents the distribution of data points, considering a certain margin of error (epsilon). Unlike classification SVM, which aims to maximize the margin between classes, SVR aims to fit as many instances as possible within a specified margin while limiting violations (data points outside the margin) to within a certain tolerance.

#### Why We Use It

The reasons why SVR can be considered for stock price prediction are stock price movements often exhibit nonlinear patterns, and SVR is capable of capturing nonlinear relationships between input features and target variables. SVR is flexible and can handle various types of data distributions. It can adapt to different market conditions and capture trends or patterns that may not be captured by linear models. This model is also less sensitive to outliers compared to some other regression techniques. In the stock market, where unexpected events can lead to outliers in the data, this robustness can be beneficial. In addition, SVR includes a regularization term in its cost function, helping prevent overfitting. 

## Code Components <a name="paragraph3"></a>

## Results <a name="paragraph4"></a>

## Difficulties <a name="paragraph5"></a>

## Libraries <a name="paragraph6"></a>

1. Pandas (The Pandas development team, publisher Zendoo, version latest)
2. Numpy (Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke & Travis E. Oliphant, published 09.x.2020)
3. yfinance (yfinance is a community effort, data belongs to Yahoo)
4. Sklearn (Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.,published 2011)
5. Matplotlib (Hunter , J. D., published 2007)

