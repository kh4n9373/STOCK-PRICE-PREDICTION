# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

# Set random seed
np.random.seed(42)

# Load data
df = pd.read_csv("../Data/VFS_historical_data_yfinance.py")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.head()

# Plot data
plt.figure(figsize=(10,6))
plt.plot(df['Close'], label='MSFT')
plt.title('MSFT Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Split data into train and test sets
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]
X_train = train.drop('Close', axis=1)
y_train = train['Close']
X_test = test.drop('Close', axis=1)
y_test = test['Close']

# Define a function to calculate accuracy
def accuracy(y_true, y_pred):
  # Define a threshold
  threshold = 0.01
  # Calculate the percentage difference
  diff = np.abs(y_true - y_pred) / y_true
  # Return the accuracy score
  return np.mean(diff < threshold)

# Define a function to evaluate a model
def evaluate(model, X_train, y_train, X_test, y_test):
  # Fit the model on the train set
  model.fit(X_train, y_train)
  # Predict on the test set
  y_pred = model.predict(X_test)
  # Calculate the metrics
  mse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  acc = accuracy(y_test, y_pred)
  # Print the results
  print(f"MSE: {mse:.2f}")
  print(f"MAE: {mae:.2f}")
  print(f"Accuracy: {acc:.2f}")
  # Plot the predictions
  plt.figure(figsize=(10,6))
  plt.plot(y_test, label='Actual')
  plt.plot(y_pred, label='Predicted')
  plt.title(f"{model.__class__.__name__} Predictions")
  plt.xlabel('Date')
  plt.ylabel('Price ($)')
  plt.legend()
  plt.show()

# Define a function to perform cross validation
def cross_validate(model, X, y, cv=5):
  # Perform cross validation
  scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
  # Print the results
  print(f"Cross Validation MSE: {np.mean(-scores):.2f} (+/- {np.std(-scores):.2f})")

# Linear Regression
lr = LinearRegression