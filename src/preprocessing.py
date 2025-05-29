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

def load_test_data():
    config = read_config()
    df = pd.read_csv(config['paths']['test_data_path'])
    return df

config = read_config()


def DataTransformer(df):

     # Convert order date and ship date to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'],format='%Y-%m-%d')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'],format='%Y-%m-%d')

    # Drop columns
    columns_to_drop = ['Row ID','Postal Code','Order ID','Country','Customer ID','Product ID','Customer Name','Product Name']
    df.drop(columns=[x for x in df.columns if x in columns_to_drop],inplace=True)

    # Extract time taken for shipping
    df['Time_taken'] = (df['Ship Date'] - df['Order Date']).dt.days
    df.drop(columns='Ship Date',inplace=True)

    # Extract price per unit
    df['price_per_unit'] = df['Sales'] / df['Quantity']

    # Extract month and year from order date
    df['month'] = df['Order Date'].dt.month
    df['year'] = df['Order Date'].dt.year
        
    # Extract day of week and week of year
    df['day_of_week']= df['Order Date'].dt.dayofweek
    df['week_of_year'] = df['Order Date'].dt.isocalendar().week

    # Extract lag features
    df['sales_lag_1'] = df['Sales'].shift(1)
    df['sales_lag_2'] = df['Sales'].shift(2)
    df['sales_lag_3'] = df['Sales'].shift(3)
    df['sales_lag_4'] = df['Sales'].shift(4)
    df['sales_lag_5'] = df['Sales'].shift(5)

    df.dropna(inplace=True) # Drop the rows with missing values

    return df


#---------------------------------------------------------------------------------
# Transform user data
def transform_user_data(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'],format='%d-%m-%Y',errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'],format='%d-%m-%Y',errors='coerce')

        # Drop columns
    columns_to_drop = ['Row ID','Postal Code','Order ID','Country','Customer ID','Product ID','Customer Name','Product Name']
    df.drop(columns=[x for x in df.columns if x in columns_to_drop],inplace=True)

    df['Time_taken'] = (df['Ship Date'] - df['Order Date']).dt.days
    

    df['price_per_unit'] = df['Sales'] / df['Quantity']
  
    # Extract month, year, day of week and week of year from order date
    df['month'] = df['Order Date'].dt.month
    df['year'] = df['Order Date'].dt.year
    df['day_of_week']= df['Order Date'].dt.dayofweek
    df['week_of_year'] = df['Order Date'].dt.isocalendar().week
    
    df.drop(columns=['Ship Date','Order Date'],inplace=True)


    return df



'''
Feature need for prediction:
1. Order Date (year,month,day,week_of_year,day_of_week)
2. Ship Mode
3. Segment
4. City
5. State
6. Category
7. Sub-Category
8. sales_lag1
9. sales_lag2
10. sales_lag3
11. sales_lag4
12. sales_lag5
13. Quantity
14. Discount
15. Profit
16. price_per_unit
17. Time_taken
'''