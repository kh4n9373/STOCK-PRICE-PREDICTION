
import pandas as pd
import numpy as np
from sklearn import model_selection
from split_train_test import create_train_test_test
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import model_selection


def KNN_implement(df_Stock, your_data):
    X_train, X_test, Y_train, Y_test = create_train_test_test(df_Stock)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_model = knn_regressor.fit(X_train_scaled, Y_train)

    knn_kfold = model_selection.KFold(n_splits=10, random_state=100, shuffle=True)
    results_kfold = model_selection.cross_val_score(knn_model, X_test_scaled, Y_test.astype(int), cv=knn_kfold)
    
    your_data_scaled = scaler.transform(your_data)
    res = knn_model.predict(your_data_scaled)
    return results_kfold.mean() * 100,res


