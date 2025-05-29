from preprocessing import read_config ,load_daily_data
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


config = read_config()
config_dir = config['dir']
config_path = config['paths']


# Plot the forecast
def plot_forecast(dates,y_actual,y_pred):

    # Convert y_test and y_pred to pandas Series
    dates = pd.to_datetime(dates)

    y_actual_series = pd.Series(y_actual.values, index=dates)
    y_pred_series = pd.Series(y_pred, index=dates)

    y_actual_resampled = y_actual_series.resample('W').sum()
    y_pred_resampled = y_pred_series.resample('W').sum()


    fig = go.Figure()

    # Actual Sales
    fig.add_trace(go.Scatter(
        x=y_actual_resampled.index,
        y=y_actual_resampled.values,
        mode='lines+markers',
        name='Actual Weekly Sales',
        line=dict(color='royalblue', width=2),
        marker=dict(size=6)
    ))

    # Predicted Sales
    fig.add_trace(go.Scatter(
        x=y_pred_resampled.index,
        y=y_pred_resampled.values,
        mode='lines+markers',
        name='Predicted Weekly Sales',
        line=dict(color='firebrick', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    # Format the plot
    fig.update_layout(
        title='📈 Weekly Actual vs Predicted Sales',
        title_font=dict(size=22, color='darkblue'),
        xaxis_title='Date',
        yaxis_title='Sales Amount',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white',
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='gray'
        ),
        hovermode='x unified',
        height=550,
        width=1100
    )

    fig.write_image(config['dir']['artifacts_dir'] +'/forecast.png')

    return fig


#---------------------------------------------------------------------------------
def prepare_data(date:str):

    df_daily = load_daily_data()
    df_daily = df_daily.drop(['Unnamed: 0'],axis=1)
    df_daily['Order Date'] = pd.to_datetime(df_daily['Order Date'],format="%Y-%m-%d")
    date = pd.to_datetime(date)

    end_date = date + timedelta(days=6)
    daily_sales = df_daily.set_index(['Order Date'])['Sales']
    actual_sales = daily_sales.loc[date:end_date]

    target_date = date
    target_df = df_daily[df_daily['Order Date']==target_date].drop(['Sales','Order Date'],axis=1)

    return target_df ,actual_sales 


