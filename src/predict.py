from pipeline import transformation,plot_forecast
import numpy as np
from preprocessing import load_test_data,read_config
from sklearn.metrics import mean_squared_error,r2_score
import joblib


def predict(df):

    df = load_test_data()
    config = read_config()

    #Transform the data
    x,y,dates = transformation(df)

    # Load the model
    model_uri = config['paths']['rf_model_path']
    model = joblib.load(model_uri)

    # Predict the data
    y_pred = model.predict(x)

    # Calculate the metrics
    mse = mean_squared_error(y,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y,y_pred)

    # Plot the forecast
    fig = plot_forecast(dates,y,y_pred)

    return rmse, r2, fig

