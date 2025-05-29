import pandas as pd
import numpy as np
import yaml

# Read config file  
def read_config():
    with open('configs/paths.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load data from csv file
def load_data():
    config = read_config()
    df = pd.read_csv(config['paths']['raw_data_path'], encoding='ISO-8859-1')
    return df

def load_daily_data():
    config = read_config()
    df = pd.read_csv(config['paths']['processed_data_path'])
    return df

config = read_config()



#--------------------------------------------------------------------------------