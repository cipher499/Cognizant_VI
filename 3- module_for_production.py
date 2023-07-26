"""
Model training & Evaluation: Cognizant VI
Created: Wed, July 26, 2023
@author: cipher499
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# GLOBAL CONSTANTS

# number of folds for cross-validation
K = 10

# split of the train-test data
SPLIT = 0.80

# load the data into a dataframe
def load_data(path: str = "path/to/csv"):
    """
    This function takes a path string to a CSV file and loads it into a DataFrame
    : param  path(optional): str, relative path of the CSV file
    : return df: pd.DataFrame
    """

    df = pd.read_csv(f"{path}")
    
    # drop the unnecessary column
    df.drop("Unnamed: 0", axis=1, inplace=True)
    return df

# Create a target variable and predictor variables
def target_and_predictors(data: pd.DataFrame = None, target: str = "estimated_stock_pct"):
    """
    This function takes the dataframe as input, and returns the target dataframe & the features dataframe
    : param     data: pd.DataFrame -> dataframe containing data for the model
    : param     target: str(optional) -> target value that you want to predict
    : return    X: pd.DataFrame -> features
                y: pd.Series -> target
    """
    #Raise an exception if the target is not present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns={target})
    y = data[target]
    return X,y

# Train the algorithm
def train_algorithm(X: pd.DataFrame = None, y: pd.DataFrame = None):
    """
    This function takes the features and the target as input, trains a RandomForestAggressor on them across
    K folds & using cross-validation, outputs the performance metric for each fold
    """

    # Create a list for storing accuracy of each fold
    accuracy = []

    for fold in range(0, K):
        
        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale the features
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")

# Execute training pipeline
def run():
    """
    This function executes the training pipeline of loading the prepared
    dataset from a CSV file and training the machine learning model

    :param

    :return
    """

    # Load the data
    df = load_data()

    # Split the data into predictors and target variables
    X, y = target_and_predictors()(data=df)

    # Train the machine learning model
    train_algorithm()(X=X, y=y)
