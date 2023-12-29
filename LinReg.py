import numpy as np
from sklearn.linear_model import LinearRegression
from split_train_test import create_train_test_test

def Lr_implement(df_Stock,your_data):
    X_train, X_test, Y_train, Y_test = create_train_test_test(df_Stock) # Split train/test set
    lr = LinearRegression() # Implement model
    lr.fit(X_train, Y_train)
    y_pred = lr.predict(X_test) # Predict test set
    accuracy = 1 - np.mean(np.abs((y_pred - Y_test) / Y_test)) # Test for accuracy
    result = lr.predict(your_data) # Predict base on user data
    return accuracy, result



