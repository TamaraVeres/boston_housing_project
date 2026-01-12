# Boston House Pricing Project

This project explores the Boston housing dataset using exploratory data analysis and compares the performance of linear regression and a neural network for house price prediction.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis:

```bash
python boston_main.py
```

## Project Structure

```
house_pricing_project/
├── boston_main.py           # Main entry point
├── requirements.txt         # Dependencies
├── README.md
└── src/
    ├── exploratory_data_analysis.py
    ├── linear_regression.py
    └── neural_network.py
```

## Dataset Description

The initial columns of the dataset were renamed for simpler, more understandable names.

| Feature | Original Name | Description |
|---------|---------------|-------------|
| crime | CRIM | How much crime there is in the area |
| large_lots | ZN | How much land is reserved for large houses |
| industry | INDUS | How industrial (factories/businesses) the area is |
| river | CHAS | Whether the area is next to the Charles River (1 = yes, 0 = no) |
| pollution | NOX | How polluted the air is |
| rooms | RM | Average house size (number of rooms) |
| house_age | AGE | How old the houses are |
| job_distance | DIS | How far the area is from job centers |
| highway_access | RAD | How easy it is to reach highways (1→24) |
| tax_rate | TAX | How high property taxes are |
| school_crowding | PTRATIO | How crowded schools are |
| poverty | LSTAT | How poor the population is |

## Exploratory Data Analysis

The script first loads the dataset, renames the feature columns, and removes the additional "B" column for ethical reasons.

It then performs a basic exploratory analysis, including a statistical summary, skewness analysis, and inspection of feature distributions. The features crime and large_lots are highly skewed, while job_distance and highway_access also present moderate skewness.

Next, a correlation analysis is conducted to examine the linear relationships between the features and the target variable (house price). The two features most strongly correlated with house prices are number of rooms and poverty. The number of rooms shows a positive correlation with price, whereas poverty is negatively correlated. The feature with the weakest correlation to house price is job_distance.

## Feature Engineering

The data analysis script also includes a feature engineering part where new variables (features) are created to improve the model's ability to learn patterns from the data.

First the highway_access feature is transformed into three categorical groups (low, medium, and high access) and encoded using one-hot encoding.

In addition, polynomial and interaction features are created to capture nonlinear relationships that cannot be modeled directly by linear regression. These include the squared the number of rooms, as well as interaction terms between key variables such as rooms and poverty, pollution and industry, crime and poverty, and school crowding and poverty.

Finally, interaction terms between job distance and different levels of highway access are added to reflect how accessibility may influence housing prices differently depending on location.

## Linear Regression

Linear regression is a basic machine learning model that predicts house prices by learning a linear relationship between the input features (such as number of rooms, crime rate, or poverty level) and the target variable (house price). The model fits a straight line that best represents the data by minimizing the difference between the predicted and actual prices.

In this project, the linear regression model is trained on the training dataset and then used to predict prices for unseen test data. The model's performance is evaluated using the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R² score, which indicate how close the predictions are to the true house prices.

Feature scaling is not applied for linear regression because the model is not sensitive to the scale of the input features. Scaling does not change the predictions of linear regression, as the model coefficients automatically adjust to different feature magnitudes. In other words, linear regression "balances" the feature sizes by adjusting the weights, so scaling does not change the predicted house prices. For this reason, scaling is optional and mainly used for models such as neural networks.

## Neural Network

A neural network is a machine learning model that can learn complex and nonlinear relationships between the input features and the target variable. Unlike linear regression, it uses multiple layers and activation functions to gradually transform the input data into better predictions.

Before training the neural network, the input features are scaled using standardization or normalization, in this dataset standardization was used. This means that each feature is transformed to have a mean of 0 and a standard deviation of 1. Feature scaling is necessary for neural networks because they are sensitive to large differences in feature values.

During training, the neural network makes predictions on the training data, compares them to the true prices using Mean Squared Error (MSE), and updates its weights using gradient descent (Adam optimizer). This process is repeated over multiple epochs and with each epoch prediction error is reduced.

Finally, the trained model is evaluated on the test dataset using MSE, RMSE, and R² to measure how accurately it predicts the house prices.

## Results

| Model | Average Prediction Error | Performance |
|-------|--------------------------|-------------|
| Linear Regression | ~$3,700 | Performs well and explains most of the variation in house prices |
| Neural Network | ~$3,000 | Achieves slightly better results with lower prediction error and higher explained variance |

## Conclusion

The linear regression model performs well and explains most of the variation in house prices, with predictions being off by approximately $3,700. The neural network achieves slightly better results, with a lower prediction error and a higher explained variance, with prices being off by approximately $3,000.
