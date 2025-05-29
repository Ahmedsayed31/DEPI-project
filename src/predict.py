from pipeline import plot_weekly_forecast,prepare_data,read_config
from preprocessing import load_daily_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from datetime import timedelta
import xgboost as xgb
import warnings 
warnings.filterwarnings('ignore')

config = read_config()
config_path = config['paths']
config_dir = config['dir']



def predict_next_week(date_str: str):
    # Prepare base data
    target_df, actual_sales = prepare_data(date_str)

    # Convert date string to datetime
    current_date = pd.to_datetime(date_str)

    # List to store predictions
    predictions = []

    # Keep sales history to calculate lags & rolling features
    df_daily = load_daily_data()
    df_daily = df_daily.drop(['Unnamed: 0'], axis=1)
    df_daily['Order Date'] = pd.to_datetime(df_daily['Order Date'], format="%Y-%m-%d")
    df_daily = df_daily.set_index('Order Date')
    sales_history = df_daily['Sales'].copy()

    # Loop for each day in the next 7 days
    for _ in range(7):
        # Get lag values
        target_df['sales_lag_1'] = sales_history.shift(1).loc[current_date]
        target_df['sales_lag_2'] = sales_history.shift(2).loc[current_date]
        target_df['sales_lag_3'] = sales_history.shift(3).loc[current_date]
        target_df['sales_lag_7'] = sales_history.shift(7).loc[current_date]
        target_df['sales_lag_14'] = sales_history.shift(14).loc[current_date]
        target_df['sales_lag_30'] = sales_history.shift(30).loc[current_date]
        target_df['sales_lag_365'] = sales_history.shift(365).loc[current_date]

        # Get rolling averages
        target_df['rolling_mean_sales_3d'] = sales_history.rolling(3).mean().loc[current_date]
        target_df['rolling_mean_sales_7d'] = sales_history.rolling(7).mean().loc[current_date]
        target_df['rolling_mean_sales_14d'] = sales_history.rolling(14).mean().loc[current_date]
        target_df['rolling_mean_sales_30d'] = sales_history.rolling(30).mean().loc[current_date]
        target_df['rolling_mean_sales_1q'] = sales_history.rolling(120).mean().loc[current_date]

        # Predict the sales
        xgb_model = xgb.Booster()
        xgb_model.load_model(config_path['xgb_model_path'])

        dmatrix = xgb.DMatrix(target_df)
        pred = xgb_model.predict(dmatrix)[0]
        predictions.append(pred)

        # Update sales history with the predicted value
        sales_history.loc[current_date] = pred

        # Update date
        current_date += timedelta(days=1)

        # Copy static features from previous day
        if current_date in df_daily.index:
            static_features = df_daily.loc[current_date].drop('Sales')
        else:
            static_features = target_df.iloc[0].copy()

        target_df = pd.DataFrame([static_features])

    results = pd.DataFrame(actual_sales.values,columns=['actual'],index=actual_sales.index).reset_index()
    results['predictions'] = predictions

    actual = results['actual'].values
    preds = results['predictions'].values

    fig = plot_weekly_forecast(results)

    mse = mean_squared_error(actual,preds)
    rmse = np.sqrt(mse)

    r2 = r2_score(actual,preds)


    return fig,rmse,r2,results

fig,rmse ,r2,result_df = predict_next_week(date_str='2014-12-21')
print(rmse,r2,result_df)
