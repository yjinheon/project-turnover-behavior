# Import important packages
import pandas as pd
import numpy as np
import joblib
import os
import yaml
# folder to load config file
CONFIG_PATH = "../"
# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("_config_ex1.yaml")

# load data
df = pd.read_csv(os.path.join(config["data_directory"], config["data_name"]))
df.columns = df.columns.str.lower()


if __name__ == "__main__":
    # load data
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.isnull().sum().sum())
    print(df.dtypes)
    print(df.columns)
    print(df.columns[df.isnull().any()])
    print(df.columns[df.isnull().any()].shape)
