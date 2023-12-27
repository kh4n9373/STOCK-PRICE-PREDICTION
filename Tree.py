from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
# from split_train_test import create_train_test_test

def Tree_implement(df_Stock, your_data):
    X = df_Stock.drop(columns=['Close'])
    Y = df_Stock['Close']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=42)
    tree = DecisionTreeRegressor()
    tree.fit(X_train, Y_train)

    y_pred = tree.predict(X_test)
    accuracy = 100 - mean_absolute_percentage_error(Y_test, y_pred) * 100

    result_tree = tree.predict(your_data)

    return accuracy/100, result_tree