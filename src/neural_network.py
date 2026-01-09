
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim


def prepare_torch_data(X_train, y_train, X_test, y_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(
        y_train.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(
        y_test.values, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler


def neural_network_model(input_size):
    model_sequential = nn.Sequential(
        nn.Linear(input_size, 64),  # linear layer
        nn.ReLU(),  # activation function
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model_sequential


def train_model(model_sequential, X_train_tensor, y_train_tensor, num_epochs=1500, lr=0.001):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model_sequential.parameters(), lr=lr)

    for epoch in range(num_epochs):
        predictions = model_sequential(X_train_tensor)  # forward pass
        MSE = loss_fn(predictions, y_train_tensor)  # compute the loss
        optimizer.zero_grad()  # reset the gradients
        MSE.backward()  # backward pass
        optimizer.step()  # update the weights

        # if (epoch + 1) % 100 == 0:
        #     print(f"Epoch {epoch+1}/{num_epochs}, MSE: {MSE.item():.4f}")

    return model_sequential


def evaluate_model(model_sequential, X_test_tensor, y_test_tensor):
    model_sequential.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():  # we don't track the gradients, we only want to evaluate the model
        test_predictions = model_sequential(X_test_tensor)  # forward pass
        # compute the loss
        test_loss = loss_fn(test_predictions, y_test_tensor)

        print("\nNeural network model evaluation:")
        print("MSE:", test_loss.item())
        print("RMSE:", np.sqrt(test_loss.item()))
        print("R²:", r2_score(y_test_tensor.numpy(), test_predictions.numpy()))
        print("The predicted prices are off by approximately $", np.sqrt(test_loss.item()))

        # turn predictions into a simple list
        nn_preds = test_predictions.numpy().flatten()
        return nn_preds
