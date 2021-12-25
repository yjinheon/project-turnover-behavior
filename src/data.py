# Import important packages
import pandas as pd
import numpy as np
import joblib   
import os
import yaml
from pathlib import Path # 상위 폴더 경로 관련
from tools import *


# folder to load config file
BASE_PATH = Path(__file__).parent.parent

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(BASE_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


# dir
def search(dirname):
    for dirname, _, filenames in os.walk(dirname):
            for filename in filenames:
                full_filename = os.path.join(dirname, filename)
                print (full_filename)

config = load_config('_config_ex1.yaml')                
    
    
df = pd.read_csv(config['data_directory']+config['data_name'],encoding='cp949')
    


if __name__ == "__main__":
    # load data
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.info())
    print(df.describe())
