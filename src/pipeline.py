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
def plot_weekly_forecast(df):
    """
    Plot actual vs predicted sales using Plotly.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['dates', 'actual', 'preds']
    """
    fig = go.Figure()

    # Actual sales trace
    fig.add_trace(go.Scatter(
        x=df['Order Date'],
        y=df['actual'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='blue')
    ))

    # Predicted sales trace
    fig.add_trace(go.Scatter(
        x=df['Order Date'],
        y=df['predictions'],
        mode='lines+markers',
        name='Predicted Sales',
        line=dict(color='orange', dash='dash')
    ))

    # Layout settings
    fig.update_layout(
        title='Weekly Sales Forecast vs Actual',
        xaxis_title='Date',
        yaxis_title='Sales',
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
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


