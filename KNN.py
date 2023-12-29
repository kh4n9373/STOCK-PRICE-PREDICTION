import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import model_selection
from sklearn.metrics import r2_score
def KNN_implement(df_Stock, your_data):
    X = df_Stock.drop(columns=['Close'])
    Y = df_Stock['Close']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_model = knn_regressor.fit(X_train_scaled, Y_train)

    knn_kfold = model_selection.KFold(n_splits=10, random_state=100, shuffle=True)
    results_kfold = model_selection.cross_val_score(knn_model, X_test_scaled, Y_test.astype(int), cv=knn_kfold)
    
    your_data_scaled = scaler.transform(your_data)
    res = knn_model.predict(your_data_scaled)
    y_pred = knn_model.predict(X_test_scaled)
    accuracy = 1 - np.mean(np.abs((y_pred- Y_test) / Y_test))
    return accuracy, res
