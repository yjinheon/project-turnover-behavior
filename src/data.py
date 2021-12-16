# Import important packages
import pandas as pd
import numpy as np
import joblib   
import os
import yaml
from tools import *
from tools.glimpse import *

# folder to load config file
CONFIG_PATH = "../"
# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("_config_ex1.yaml")

# dir
def search(dirname):
    for dirname, _, filenames in os.walk(dirname):
            for filename in filenames:
                full_filename = os.path.join(dirname, filename)
                print (full_filename)

obj = MLclass(X_train, y_train)

class MLclass:
    def __init__(self, config_path):
        self.df = pd.read_csv(os.path.join(config["data_directory"], config["data_name"]).replace("\\", "/"),encoding='cp949')
    
    def subset_df(self):
    


if __name__ == "__main__":
    # load data
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.info())
    print(df.describe())
    print(glimpse(df))
