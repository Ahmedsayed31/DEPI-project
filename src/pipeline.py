from preprocessing import DataTransformer,read_config ,transform_user_data
from sklearn.preprocessing import QuantileTransformer,StandardScaler,LabelEncoder
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


config = read_config()
config_dir = config['dir']
config_path = config['paths']
# Transformation
def transformation(data):

    # Transform the data
    data = DataTransformer(data)

    # Split the data
    # Extract the dates
    # data['Order Date'] = pd.to_datetime(data['Order Date'])
    dates = data['Order Date']

    # Extract the features and target
    x = data.drop(['Sales','Order Date'],axis=1)
    y = data['Sales']

    # Encoding and Scaling      
    def encoding_and_scaling(data):

        quantile_columns = ['Profit', 'price_per_unit']
        quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        data[quantile_columns] = quantile_transformer.fit_transform(data[quantile_columns])

        # Standard Scaling for normally distributed numerical columns
        num_scaling = ['Quantity', 'Discount', 'Time_taken']    
        scaler = StandardScaler()
        data[num_scaling] = scaler.fit_transform(data[num_scaling])

        # Label Encoding for categorical columns
        col_label = data.select_dtypes('object').columns.tolist()
        label_encoder = LabelEncoder()
        for col in col_label:
            data[col] = label_encoder.fit_transform(data[col])

        return data
    
    # Apply the encoding and scaling
    x = encoding_and_scaling(x)

    # Return the transformed data
    return x,y,dates





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
# Transform user data
def prepare_user_data(data):
    data = transform_user_data(data)

    def encoding_and_scaling(data):

        quantile_columns = ['Profit', 'price_per_unit']
        qe = joblib.load(config_path['quantile_encoder_path'])
        data[quantile_columns] = qe.transform(data[quantile_columns])

        # Standard Scaling for normally distributed numerical columns
        num_scaling = ['Quantity', 'Discount', 'Time_taken']   
        scaler =joblib.load(config_path['scaler_encoder_path'])
        data[num_scaling] = scaler.transform(data[num_scaling])

        # Label Encoding for categorical columns
        label_encoders = joblib.load(config_path['label_encoder_path'])
        col_label = ['Ship Mode','Segment','City','State','Region','Category','Sub-Category']
        for col in col_label:
            if col in label_encoders:
                data[col] = label_encoders[col].transform(data[col])

        return data

    data = encoding_and_scaling(data)

    return data