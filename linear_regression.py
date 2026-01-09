import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression_model(X_train, y_train, X_test, y_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    linear_train_predictions = linear_model.predict(X_train)
    linear_train_mse = mean_squared_error(y_train, linear_train_predictions)
    linear_test_predictions = linear_model.predict(X_test)
    test_mse = mean_squared_error(y_test, linear_test_predictions)
    
    print("Linear regression model evaluation:")
    print("Train MSE:", linear_train_mse)
    print("Test MSE:", test_mse)
    print("RMSE:", np.sqrt(test_mse))
    print("R^2:", r2_score(y_test, linear_test_predictions))
    print("The predicted prices are off by approximately $", np.sqrt(test_mse))

    lr_preds = linear_model.predict(X_test)
    return lr_preds

