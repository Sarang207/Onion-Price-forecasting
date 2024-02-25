from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your data
df = pd.read_excel("projectdatset.xlsx", parse_dates=["Reported Date"])

# Assuming 'Reported Date' is the name of your date column
df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce')

# Drop rows with NaT values if needed
df = df.dropna(subset=['Reported Date'])

# Create 'Date' and 'Time' columns
df['Date'] = df['Reported Date'].dt.date
df['Time'] = df['Reported Date'].dt.time

# Create a new DataFrame using .loc to avoid SettingWithCopyWarning
new_df = df.loc[:, ["Reported Date", "Modal Price (Rs./Quintal)"]].copy()

# Assuming 'Modal Price (Rs./Quintal)' is the column of interest
new_df['Transformed'] = np.log1p(new_df['Modal Price (Rs./Quintal)'])

# Filling missing values with a specific value (e.g., 0)
new_df['Transformed'].fillna(0, inplace=True)

# Checking for missing values again
print(new_df[['Transformed', 'Reported Date']].isnull().sum())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Your time series forecasting code here
    # Assuming 'Transformed' is the name of your transformed column
    transformed_data = new_df['Transformed']

    # Fit ARIMA model
    order = (1, 3, 1)
    model = ARIMA(transformed_data, order=order)
    fit_model = model.fit()

    # Residual analysis
    # Get residuals
    residuals = fit_model.resid

    # Plot residuals and save as PNG
    plt.figure(figsize=(10, 6))
    plt.plot(new_df['Reported Date'], residuals)
    plt.title('Residuals Plot')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.savefig('static/residuals_plot.png')  # Save the plot as an image
    plt.close()  # Close the plot to avoid displaying in Flask application




    # Forecast the next value
    next_forecast = fit_model.get_forecast(steps=1)
    forecasted_value = next_forecast.predicted_mean.iloc[0]

    # Convert log-transformed value to the original scale
    forecasted_value_original = np.exp(forecasted_value)

    # Assuming the actual value for the next time step is available
    actual_value = new_df['Transformed'].iloc[-1]

    # Calculate MAE (Mean Absolute Error)
    mae = mean_absolute_error([actual_value], [forecasted_value])

    print(f'Forecasted log value for the next time step: {forecasted_value}')
    print(f'Forecasted original value for the next time step: {forecasted_value_original}')
    print(f'Mean Absolute Error (MAE): {mae}')
    # In the forecast route, pass the image file path to the template
    return render_template('forecast.html', forecasted_value=forecasted_value_original,
                           plot_image='static/residuals_plot.png')

if __name__ == '__main__':
    app.run(debug=True)
