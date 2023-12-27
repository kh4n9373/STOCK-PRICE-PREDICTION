from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from split_train_test import create_train_test_test

def Tree_implement(df_Stock, your_data):
    X_train, X_test, Y_train, Y_test = create_train_test_test(df_Stock)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    your_data = scaler.transform(your_data)

    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=3)
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    tree = DecisionTreeRegressor(**best_params)
    tree.fit(X_train, Y_train)

    y_pred = tree.predict(X_test)
    accuracy = 100 - mean_absolute_percentage_error(Y_test, y_pred) * 100

    result_tree = tree.predict(your_data)

    return accuracy/100, result_tree