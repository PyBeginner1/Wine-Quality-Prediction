import pandas
import yaml
import pandas as pd
import argparse


def get_data(config_path):
    config = read_params(config_path)
    data_path = config['data_source']['s3_source']
    df = pd.read_csv(data_path)
    return df

def read_params(config_path):
    with open(config_path, "r") as f:
        config_file = yaml.safe_load(f)
    return config_file


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path = parsed_args.config)