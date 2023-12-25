
import pandas as pd
import numpy as np
from split_train_test import create_train_test_test
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def Tree_implement(df_Stock, your_data):
    X_train, X_test, Y_train, Y_test = create_train_test_test(df_Stock)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    tree = DecisionTreeRegressor()
    tree.fit(X_train, Y_train)
    
    y_pred = tree.predict(X_test)
    accuracy = mean_squared_error(Y_test, y_pred)
    print('\nPerformance:', accuracy)

    your_data = scaler.transform(your_data)
    result = tree.predict(your_data)
    return result


