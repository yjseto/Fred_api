import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# FRED API configuration
base_url = 'https://api.stlouisfed.org/fred/'
api_key = ' '  # insert your FRED API key here

def fetch_fred_data(series_id, start_date, end_date):
    endpoint = 'series/observations'
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
        'frequency': 'q'  # Quarterly data
    }
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['value'] = df['value'].astype(float)
        return df['value']
    else:
        print(f'Failed to retrieve data for {series_id}. Status code:', response.status_code)
        return None

def fetch_and_preprocess_data(start_date, end_date):
    series_ids = {
        'unemployment': 'UNRATE',
        'population': 'POP',
        'employment': 'PAYEMS',
        'labor_force': 'CLF16OV',
        'money_supply': 'M2SL',
        'interest_rate': 'FEDFUNDS'
    }

    data = {}
    for name, series_id in series_ids.items():
        data[name] = fetch_fred_data(series_id, start_date, end_date)

    df = pd.DataFrame(data)
    df = df.interpolate()

    for col in df.columns:
        if col != 'unemployment':
            df[f'{col}_lag4'] = df[col].shift(4)

    df = df.dropna()
    return df

def prepare_data(df, target_col='unemployment', scaler=None):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y, scaler

def train_and_predict(X_train, y_train, X_test):
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    # Fetch and preprocess data
    data = fetch_and_preprocess_data('1970-01-01', '2023-12-31')

    # Split data into training (1970-2000) and testing (2001 onwards) sets
    train_data = data[data.index < '2001-01-01']
    test_data = data[data.index >= '2001-01-01']

    # Prepare training data
    X_train, y_train, scaler = prepare_data(train_data)

    # Initialize lists to store actual and predicted values
    actual_values = []
    predicted_values = []
    
    # Iterate through test data, updating the model each quarter
    for i in range(len(test_data)):
        # Prepare test data
        X_test, y_test, _ = prepare_data(test_data.iloc[:i+1], scaler=scaler)
        
        # Train model and make prediction
        prediction = train_and_predict(X_train, y_train, X_test[-1].reshape(1, -1))
        
        # Store actual and predicted values
        actual_values.append(y_test.iloc[-1])
        predicted_values.append(prediction[0])
        
        # Update training data
        X_train = np.vstack([X_train, X_test[-1]])
        y_train = np.append(y_train, y_test.iloc[-1])

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    print(f"Root Mean Squared Error: {rmse}")

    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, actual_values, label='Actual')
    plt.plot(test_data.index, predicted_values, label='Predicted')
    plt.title('Unemployment Rate Forecast (4 Quarters Ahead)')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('unemployment_forecast.png')
    plt.close()

    print("Forecast visualization saved as 'unemployment_forecast.png'")