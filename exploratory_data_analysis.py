import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# loading the boston housing prices dataset


def load_data():
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    print("Data loaded successfully")
    X = boston.data
    y = boston.target
    y.name = "price"
    return X, y

# cleaning and renaming the features


def clean_data(X):
    X = X.drop(columns="B")
    X = X.rename(columns={
        "CRIM": "crime",
        "ZN": "large_lots",
        "INDUS": "industry",
        "CHAS": "river",
        "NOX": "pollution",
        "RM": "rooms",
        "AGE": "house_age",
        "DIS": "job_distance",
        "RAD": "highway_access",
        "TAX": "tax_rate",
        "PTRATIO": "school_crowding",
        "LSTAT": "poverty"
    })
    print("Data cleaned successfully")
    return X

# basic inspection


def basic_inspection(X, y):
    print("Feature description:\n", X.info())
    print("Feature statistics:\n", X.describe())
    print("Feature head:\n", X.head())
    print("Target description:\n", y.info())
    print("Target statistics:\n", y.describe())
    print("Target head:\n", y.head())
    print("\nMissing feature values:\n", X.isnull().sum())
    print("\nFeature duplicates:", X.duplicated().sum())
    print("\nMissigng target values:", y.isnull().sum())
    print("Basic inspection completed")
    return X, y


# checking skewness and distribution
def skewness_and_distribution(X, y):
    X["river"] = X["river"].astype("int8")
    X["highway_access"] = X["highway_access"].astype("int8")
    print("\nSkewness of the features:\n", X.skew())
    X.hist(bins=30, figsize=(15, 10))
    plt.title("Distribution of the features")
    plt.show()
    y.hist(bins=30, figsize=(15, 10))
    plt.title("House price distribution")
    plt.show()
    return X, y

# checking the correlation between the features and the target


def correlation_matrix(X, y):
    df = X.assign(price=y)
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    return X, y

# encoding


def feature_engineering_and_transformation(X):
    # converting the numerical columns to float16
    numerical_cols = ["crime", "large_lots", "industry", "pollution", "rooms",
                      "house_age", "job_distance", "tax_rate", "school_crowding", "poverty"]
    X[numerical_cols] = X[numerical_cols].astype("float32")
    # encoding the highway_access column into 3 categories: Low RAD, Medium RAD, High RAD
    X["highway_access"] = pd.cut(X["highway_access"], bins=[
                                 0, 4, 8, 24], labels=[1, 2, 3])
    X = pd.concat(
        [X, pd.get_dummies(X["highway_access"], prefix="highway_access")], axis=1)
    X = X.drop(columns=["highway_access"])
    # polynomial and interaction features
    X["rooms^2"] = X["rooms"]**2
    X["size_poverty_interaction"] = X["rooms"] * X["poverty"]
    X["pollution_industry_interaction"] = X["pollution"] * X["industry"]
    X["crime_poverty_interaction"] = X["crime"] * X["poverty"]
    X["ptratio_poverty_interaction"] = X["school_crowding"] * X["poverty"]
    X["distance_vs_highway_access"] = X["job_distance"] * X["highway_access_1"]
    X["distance_vs_highway_access_2"] = X["job_distance"] * X["highway_access_2"]
    X["distance_vs_highway_access_3"] = X["job_distance"] * X["highway_access_3"]
    print("Feature engineering and transformation completed", X.head())
    return X


def get_final_dataset():
    X, y = load_data()
    X = clean_data(X)
    basic_inspection(X, y)
    X, y = skewness_and_distribution(X, y)
    X, y = correlation_matrix(X, y)
    X = feature_engineering_and_transformation(X)
    return X, y
