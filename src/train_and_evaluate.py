#1 Read params
#2 Load Train and Test file
#3 Train the model
#4 Save the Metrics and Params

import os
import pandas as pd
import warnings 
import sys 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config['split_data']['train_path']
    test_data_path = config['split_data']['test_path']
    model_dir = config['model_dir']
    alpha = config['estimators']['ElasticNet']['params']['alpha']
    l1_ratio = config['estimators']['ElasticNet']['params']['l1_ratio']
    target = config['base']['target_col']
    random_state = config['base']['random_state']
    scores_file = config['reports']['scores']
    params_file = config['reports']['params']

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    X_train = train.drop(target, axis = 1)
    X_test = test.drop(target, axis = 1)
    y_train = train[target]
    y_test = test[target]

    model = ElasticNet(alpha = alpha, l1_ratio=l1_ratio, random_state=random_state)

    model.fit(X_train, y_train)

    predict = model.predict(X_test)

    rmse, mae, r2 = eval_metrics(y_test, predict)

    print(f"Elastic Net Model(alpha: {alpha}, l1_ratio: {l1_ratio})")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    os.makedirs("report", exist_ok=True)

    with open(scores_file, "w") as f:
        scores = {
            "RMSE" : rmse,
            "MAE": mae,
            "R2Score": r2
        }
        json.dump(scores, f)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio
        }
        json.dump(params, f)


    os.makedirs(model_dir, exist_ok =True)
    model_path = os.path.join(model_dir,"model.joblib")
    joblib.dump(model, model_path)


def eval_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)