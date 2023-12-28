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
6. [Difficulties](#paragraph5)
7. [Libraries](#paragraph6)

## Introduction

Our interest in machine learning, data analytics, and the stock market has led us to choose financial machine learning as the topic for our final project in the course Introduction to AI. Despite our limited prior experience with machine learning, we are fully committed to putting forth our best effort and meticulously documenting the entire process of building the models.

As a collaborative group, we recognize the dynamic nature of the financial industry, which is rapidly evolving and actively seeking innovative ways to leverage machine learning for the effective management of risk and financial losses. Several research efforts have also been carried out to predict the market in order to make profit using different techniques. Our project aims to contribute to this evolution by exploring the intersection of financial analytics and artificial intelligence. 

The adoption of Artificial Neural Network (ANN) techniques and decision tree algorithms has gained significant traction in the business landscape, thanks to its adeptness in modeling relationships among non-linear variables. These sophisticated data mining methodologies allow deeper analysis of large set of data, ecpecially those characterized by rapid fluctuations within short time spans. Consequently, ANN stands out as a promising tool for forecasting stock market dynamics.



## Data Preparation <a name="paragraph1"></a>

We fetch data from Yahoo Finance, an excellent source for reliable stock market movements and prices. Specifically, we used the stock market data from Microsoft, Vinfast, Apple and Google. For each trading day of the stock, a record of each of the following is posted:
- Open = the price when the market opened in the morning.
- Close = the price when the market closed in the afternoon.
- High = the highest price during that trading day.
- Low = the lowest price during that trading day.
- Volume = number of shares of the stock traded that day.
- Adj Close (Adjusted Close) = a price adjusted to account various corporate actions, such as dividends, stock splits, and other events that might affect the stock's price. 

Using yFinance ensures that we have up-to-date and accurate data on stock prices and trading volumes. We then leverage the pandas library for efficient data manipulation and preprocessing, allowing us to handle missing values, normalize data, and create meaningful features. The processed dataset is exported to CSV format, maintaining compatibility with various machine learning frameworks. 


## Machine Learning Models <a name="paragraph2"></a>

Within the field of machine learning (ML), our report on Stock Price Prediction explores the utilization of advanced models to anticipate trends in the stock market. Machine learning, a subset of artificial intelligence, encompasses a range of methodologies, with supervised and unsupervised learning being particularly significant approaches. In supervised learning, a model is trained using a dataset that is labeled, allowing the algorithm to understand the relationship between input features and corresponding output labels. In the context of predicting stock market behavior, supervised learning is applicable as historical stock data can be used as a labeled dataset, enabling the model to learn and generate predictions based on known outcomes.

Within the supervised learning framework, two fundamental types of tasks are regression and classification. Regression involves predicting a continuous outcome, such as stock prices, making it directly applicable to stock market prediction. On the other hand, classification deals with categorizing data into discrete classes such as "Buy", "Sell" and "Hold", a methodology that we applied within the Random Forest Algorithm. This classification approach faciliates the decision-making process by providing actionable insights into potential investment strategies based on the historical behavior of the stock.

The learning process in machine learning has three main elements that are essential; these are the target function, training set and testing set. 
- The training set is used to construct model learning by revealing the patterns and relationship it may contain. 
- The target function explains how the inputs correlate to the outputs. 
- The testing evaluates the model and consists of unseen data for gauging its generalization ability.


### LSTM <a name="subparagraph1"></a>

#### How it Works

Long Short Term Memory (LSTM) - is a model that increases the memory of Recurrent Neural Networks (RNN). The core idea behind LSTM is the use of memory cells. These cells can store information for long durations, and their state can be selectively updated or cleared. To control the flow of information into and out of the memory cell, the algorithm uses three distinct gates: Input gate (manages the amount of new information that enters the memory cell), Forget gate (controls the removal of information from the memory cell) and Output gate (determines the information to be output based on the current cell state). About cell state, it is the internal memory of the LSTM which runs along the entire sequence and is modified by the gates at each time step. Ultimately, LSTM is trained using backpropagation through time (BPTT), similar to other RNNs with the aim to minimize the difference between the predicted output and the actual output.

#### Specifically



#### Why We Use It

In comparison with traditional RNNS, LSTM is a modified version to cope with the vanishing gradient problem, which is caused by the repeated use of the same parameters in RNN blocks, at each step. To do so, LSTM leverages gating mechanisms to control the flow of information and gradients. This helps prevent the vanishing gradient problem and allows the network to learn and retain information over longer sequences. As a result, LSTM is more effective at predicting time-series patterns, especially stock prices.

### Linear Regression <a name="subparagraph2"></a>

#### How it Works

Linear regression works by modeling the relationship between a dependent variable (e.g., stock price) and one or more independent variables through a linear equation. The model aims to find the best-fitting line that minimizes the difference between predicted and actual values in historical data. The equation takes the form Y = b0 + b1∗X + ϵ, where Y is the dependent variable, X is the independent variable, b0 is the y-intercept, b1 is the slope, and ϵ represents the error term. Training the model involves determining the coefficients b0 and B1 to create a predictive formula, enabling the estimation of future stock prices based on new input values. 

#### Why We Use It

Linear regression is employed in stock price prediction for several reasons. First, it offers interpretability, allowing analysts to understand and quantify the impact of independent variables on stock prices through the coefficients in the linear equation. Second, it provides a quick and straightforward implementation, making it a practical choice for initial analyses and as a baseline model for comparison. While linear regression has its limitations, such as assuming a linear relationship between variables, it remains valuable for its simplicity, ease of interpretation, and as a starting point for more complex modeling approaches in the domain of stock price prediction. 
### Decision Tree <a name="subparagraph3"></a>

#### How it Works

The Decision Tree algorithm employed in our stock market prediction project operates as a decision-making process. It effectively analyzes historical stock data by posing a series of inquiries regarding market conditions, such as the comparison between the closing price and the previous day's closing price, or the presence of an upward trend in the stock. By answering these inquiries, the algorithm ultimately generates a prediction for the movement of tomorrow's stock price.

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

Predicting stock prices through machine learning poses several challenges, hindering the seamless development and deployment of accurate models. These hurdles encompass the scarcity of detailed data, difficulties in assessing model performance metrics, and the inherent limitation of models relying solely on historical stock prices.

### Insufficient Data Granularity

The success of any stock price prediction model hinges on the availability and richness of data. However, we grapple with a lack of detailed information, obtaining only the fundamental stock parameters such as Open Price, High, Low, Close Price, and Volume. To enhance the predictive capabilities of our model, we require additional contextual data, including market conditions, comprehensive financial reports of companies, and insights into governmental policies that may influence the stock market. Moreover, vital numeric indicators such as Return on Investment (ROI), Price-to-Earnings (P/E) ratios, and liquidity metrics are conspicuously absent, impeding a holistic understanding of the stock's potential.

In addressing this challenge, an extensive effort should be directed towards acquiring diverse datasets, encompassing a broader spectrum of financial indicators and external factors. Integration of macroeconomic indicators and sentiment analysis from financial news could significantly augment the predictive power of the model.

### Precision and Recall Evaluation Complexities

The evaluation of machine learning models, encounters intricacies in assessing precision and recall. While the model may accurately predict the general direction of stock prices, deviations in actual stock prices present challenges in calculating precision and recall. This discrepancy arises from the model's potential accuracy in predicting trends but with significant variations in the actual stock price values.

To overcome this challenge, a nuanced approach to evaluation metrics is imperative. In addition to precision and recall, considering metrics that account for the magnitude of price variations, such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE), can offer a more comprehensive assessment of the model's performance.

### Model Performance Discrepancies

The disparities in performance among the algorithms, including Linear Regression, Decision Tree, KNN, and LSTM, present notable challenges in predicting stock prices. Linear Regression struggles with non-linear relationships, Decision Tree is susceptible to noise and overfitting, and KNN may fail to accurately capture global trends. LSTM, while adept at handling temporal dependencies, is not immune to sudden fluctuations. 

These challenges stem from difficulties in modeling complex relationships, sensitivity to noise, and the inherent unpredictability of stock market data. The choice of features and the dynamic nature of stock prices further compound the complexity, requiring a nuanced approach to address the intricacies of the financial markets effectively.

### Historical Price Pattern Learning

The limitation of our models to historical stock prices implies that the learning process is confined to discerning patterns in these historical data. While this has proven effective in capturing trends, it might fall short in predicting stock behavior in response to novel events or abrupt market shifts.

Expanding the model's learning horizon by incorporating real-time data, news sentiment analysis, and market events can potentially bridge this gap. A dynamic model that adapts to changing market conditions and incorporates timely information could offer a more accurate representation of the stock's future trajectory.

## Libraries <a name="paragraph6"></a>

1. Pandas (The Pandas development team, publisher Zendoo, version latest)
2. Numpy (Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke & Travis E. Oliphant, published 09.x.2020)
3. yfinance (yfinance is a community effort, data belongs to Yahoo)
4. Sklearn (Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.,published 2011)
5. Matplotlib (Hunter , J. D., published 2007)

