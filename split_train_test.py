import pandas as pd
import numpy as np
def create_train_test_test(df_Stock):
    features = df_Stock.drop(['Close'],axis=1)
    target = df_Stock['Close']
    
    data_len = df_Stock.shape[0]
    
    train_split = int(data_len * 0.8)
    X_train, X_test = features[:train_split], features[train_split+1:]
    Y_train, Y_test = target[:train_split], target[train_split+1:]
    
    return X_train, X_test, Y_train, Y_test
