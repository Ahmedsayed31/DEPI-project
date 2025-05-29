from logging import config
from unittest import result
from pipeline import transformation,plot_forecast,prepare_user_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_data,read_config
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import joblib
from datetime import datetime, timedelta


def get_data_around_date(date_column, target_date, days_after=30):
    """
    Returns data within a specified time window around a target date
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the data
        date_column (str): Name of the column containing dates
        target_date (str/datetime): Target date (can be string or datetime)
        days_before (int): Days to include before target date (default 5)
        days_after (int): Days to include after target date (default 30)
    
    Returns:
        pd.DataFrame: Filtered data within the specified date range
    """

    df = load_data()


    # Ensure target_date is pandas Timestamp
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    elif isinstance(target_date, datetime.date):
        target_date = pd.Timestamp(target_date)
    
    # Ensure date column is datetime64
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column],format='%d-%m-%Y')

    end_date = target_date + timedelta(days=days_after)

    mask = (df[date_column]>=target_date) & (df[date_column]<=end_date)

    filtered_data = df.loc[mask].sort_values(date_column)

    lags = filtered_data['Sales'].iloc[:5].values.tolist()
    x = filtered_data.iloc[5:6]
    y_actual = filtered_data['Sales'].iloc[5:].values

    #prepare the data
    dates = filtered_data['Order Date']

    x = prepare_user_data(x)
    y=x['Sales']
    x = x.drop('Sales',axis=1)

    return x , y_actual, dates ,lags

# x , y_actual, dates ,lags= get_data_around_date('Order Date','2014-07-01')


# def predict_next_month(date_column, target_date):
#     """
#     Predict sales for each row in y_actual using the model, updating lags each time.

#     Args:
#         model: Trained model.
#         x (pd.DataFrame): First input row with lag values and features.
#         y_actual (np.array): Actual sales values to compare.
#         dates (pd.Series): Corresponding dates for each actual sale.

#     Returns:
#         pd.DataFrame: DataFrame with predictions, actuals, errors, and dates.
#     """
    
#     x, y_actual, dates,lags = get_data_around_date(date_column, target_date)

#     config = read_config()
#         # Load the model
#     model_uri = config['paths']['rf_model_path']
#     model = joblib.load(model_uri)


#     predictions = []
#     x_base = x.copy()

#     for i in range(len(y_actual)):
#         x_input = x_base.iloc[i:i+1].copy()

#         for j in range(1, 6):
#             x_input[f'sales_lag_{j}'] = lags[j-1]

#         y_pred = model.predict(x_input)[0]
#         predictions.append(y_pred)

#         # update lags with prediction
#         lags.append(y_pred)

#     y_pred_array = np.array(predictions)

#     mae = mean_absolute_error(y_actual, y_pred_array)
#     rmse = np.sqrt(mean_squared_error(y_actual, y_pred_array))
#     r2 = r2_score(y_actual, y_pred_array)

#     result_df = pd.DataFrame({
#         "Date": dates[:len(y_actual)].reset_index(drop=True),
#         "Predicted Sales": y_pred_array,
#         "Actual Sales": y_actual[:len(y_pred_array)],
#         "Absolute Error": np.abs(y_pred_array - y_actual[:len(y_pred_array)])
#     })

#     print(f"MAE: {mae:.2f}")
#     print(f"RMSE: {rmse:.2f}")
#     print(f"R² Score: {r2:.2f}")

#     return result_df

def predict_with_updating_lags(target_date, date_column='Order Date', model_path=None, forecast_days=30):
    """
    Predicts sales with recursive lag updates with FIXED prediction variation
    
    Args:
        target_date (str/datetime): Starting date for predictions
        date_column (str): Name of date column
        model_path (str): Path to trained model
        forecast_days (int): Number of days to forecast
        
    Returns:
        pd.DataFrame: Forecast results with dates and VARIED predictions
    """
    # Load initial data and model
    x, y_actual, dates, initial_lags = get_data_around_date(date_column, target_date)
    
    if model_path is None:
        config = read_config()
        model_path = config['paths']['rf_model_path']
    model = joblib.load(model_path)
    
    # Initialize with first 5 lags (ensure we have exactly 5)
    current_lags = initial_lags[-5:].copy() if len(initial_lags) >= 5 else [0]*5
    predictions = []
    first_row_features = x.iloc[0:1].copy()
    
    # Forecasting loop with ACTUAL lag updates
    for day in range(forecast_days):
        # Prepare input with UPDATING lags
        x_input = first_row_features.copy()
        
        # Update ONLY the lag features (1-5)
        for lag_num in range(1, 6):
            x_input[f'sales_lag_{lag_num}'] = current_lags[-lag_num]
        
        # Make prediction
        y_pred = model.predict(x_input)[0]
        predictions.append(y_pred)
        
        # Update lags (FIFO queue)
        current_lags.pop(0)
        current_lags.append(y_pred)
    
    # Create results DataFrame with VARIED predictions
    result_df = pd.DataFrame({
        'Date': pd.date_range(start=pd.to_datetime(target_date), periods=forecast_days),
        'Predicted_Sales': predictions,
        'Day_of_Week': pd.date_range(start=pd.to_datetime(target_date), periods=forecast_days).day_name(),
        'Week_of_Year': pd.date_range(start=pd.to_datetime(target_date), periods=forecast_days).isocalendar().week
    })
    
    return result_df




# التنبؤ لـ 30 يوم بدءًا من تاريخ معين
forecast_results = predict_with_updating_lags('2014-01-01')

# عرض النتائج
print(forecast_results.head(10))

# تصور النتائج
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(forecast_results['Date'], forecast_results['Predicted_Sales'], marker='o')
plt.title('30-Day Sales Forecast with Lag Updates')
plt.xlabel('Date')
plt.ylabel('Predicted Sales')
plt.grid(True)
plt.show()

# config = read_config()
#         # Load the model
# model_uri = config['paths']['rf_model_path']
# model = joblib.load(model_uri)

# y_pred = model.predict(x)
# print(y_pred,y)
