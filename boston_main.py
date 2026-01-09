from src.exploratory_data_analysis import get_final_dataset
from src.linear_regression import linear_regression_model
from src.neural_network import neural_network_model, prepare_torch_data, train_model, evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    X, y = get_final_dataset()
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # linear regression
    lr_results = linear_regression_model(X_train, y_train, X_test, y_test)
    # neural network
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler = prepare_torch_data(
        X_train, y_train, X_test, y_test)
    nn_model = neural_network_model(X_train_tensor.shape[1])
    nn_model = train_model(nn_model, X_train_tensor, y_train_tensor)
    nn_preds = evaluate_model(nn_model, X_test_tensor, y_test_tensor)
    
    # compare the predicted prices by linear regression and neural network
    predicted_prices = pd.DataFrame({
        "Actual Price": y_test.values,
        "Linear Regression Prediction": lr_results,
        "Neural Network Prediction": nn_preds
    })
    print("Predicted prices:\n", predicted_prices.head(10))


if __name__ == "__main__":
    main()
