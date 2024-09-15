import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
import json

def fetch_stock_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    data = data[['Close']]
    data.dropna(inplace=True)
    return data

def preprocess_data(data, scaler_type):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler type")

    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def create_lstm_model(units=50, optimizer='adam', time_step=60):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def create_gru_model(units=50, optimizer='adam', time_step=60):
    model = Sequential()
    model.add(GRU(units, return_sequences=True, input_shape=(time_step, 1)))
    model.add(GRU(units, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def evaluate_with_validation_methods(model, X, y, params, time_step=60, model_type='lstm'):
    results = {}
    batch_size = params.pop('batch_size', 32)
    epochs = params.pop('epochs', 50)

    if model_type in ['lstm', 'gru']:
        model_instance = model(**params, time_step=time_step)
    else:
        model_instance = model(**params)

    # 80-20 Holdout Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_type in ['lstm', 'gru']:
        model_instance.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model_instance.predict(X_test)
    else:
        model_instance.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred = model_instance.predict(X_test.reshape(X_test.shape[0], -1))

    mse_holdout = mean_squared_error(y_test, y_pred)
    results['Holdout'] = mse_holdout

    # 90-10 Holdout Split
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_type in ['lstm', 'gru']:
        model_instance.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model_instance.predict(X_test)
    else:
        model_instance.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred = model_instance.predict(X_test.reshape(X_test.shape[0], -1))

    mse_9010 = mean_squared_error(y_test, y_pred)
    results['90-10 Holdout'] = mse_9010

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5)
    mse_kfold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if model_type in ['lstm', 'gru']:
            model_instance.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            y_pred = model_instance.predict(X_test)
        else:
            model_instance.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            y_pred = model_instance.predict(X_test.reshape(X_test.shape[0], -1))

        mse_kfold += mean_squared_error(y_test, y_pred)
    mse_kfold /= 5
    results['KFold'] = mse_kfold

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_tscv = 0
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if model_type in ['lstm', 'gru']:
            model_instance.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            y_pred = model_instance.predict(X_test)
        else:
            model_instance.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            y_pred = model_instance.predict(X_test.reshape(X_test.shape[0], -1))

        mse_tscv += mean_squared_error(y_test, y_pred)
    mse_tscv /= 5
    results['TimeSeries'] = mse_tscv

    return results

def convert_numpy_ints(data):
    if isinstance(data, dict):
        return {k: convert_numpy_ints(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_ints(v) for v in data]
    elif isinstance(data, np.int32):
        return int(data)
    else:
        return data

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    time_step = 60

    best_params = {
        "LSTM": {
            "params": {
                "units": 150,
                "optimizer": "adam",
                "batch_size": 32,
                "epochs": 150
            },
            "mse": 0.000813587130610585
        },
        "GRU": {
            "params": {
                "units": 100,
                "optimizer": "adam",
                "batch_size": 16,
                "epochs": 150
            },
            "mse": 0.0007052955045525027
        },
        "SVM": {
            "params": {
                "kernel": "rbf",
                "gamma": "auto",
                "degree": 5,
                "C": 10
            },
            "mse": 0.00225925668970501
        },
        "AdaBoost": {
            "params": {
                "random_state": 30,
                "n_estimators": 40,
                "learning_rate": 1.5
            },
            "mse": 0.005306782829605009
        },
        "RandomForest": {
            "params": {
                "random_state": 40,
                "n_estimators": 80,
                "min_samples_leaf": 4,
                "criterion": "squared_error"
            },
            "mse": 0.0076670908528374175
        }
    }

    models = {
        'LSTM': create_lstm_model,
        'GRU': create_gru_model,
        'SVM': SVR,
        'AdaBoost': AdaBoostRegressor,
        'RandomForest': RandomForestRegressor
    }

    validation_results = {}

    for ticker in tickers:
        print(f"Evaluating for ticker: {ticker}")
        data = fetch_stock_data(ticker)
        scaled_data, scaler = preprocess_data(data, 'minmax')
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        for model_name, create_model in models.items():
            print(f"Testing {model_name} model")
            if model_name in ['LSTM', 'GRU']:
                results = evaluate_with_validation_methods(create_model, X, y, best_params[model_name]['params'], time_step, model_type=model_name.lower())
            else:
                results = evaluate_with_validation_methods(create_model, X, y, best_params[model_name]['params'], model_type=model_name.lower())

            if model_name not in validation_results:
                validation_results[model_name] = {}

            for method, mse in results.items():
                if method not in validation_results[model_name]:
                    validation_results[model_name][method] = []
                validation_results[model_name][method].append(mse)

    # Compute average MSEs for each validation method and each model
    avg_validation_results = {}
    for model_name, methods in validation_results.items():
        avg_validation_results[model_name] = {}
        for method, mses in methods.items():
            avg_validation_results[model_name][method] = np.mean(mses)

    # Plot results
    methods = list(avg_validation_results['LSTM'].keys())
    x = np.arange(len(methods))
    width = 0.15

    plt.figure(figsize=(12, 6))

    for i, (model_name, results) in enumerate(avg_validation_results.items()):
        plt.bar(x + i * width, [results[method] for method in methods], width, label=model_name)

    plt.xlabel('Validation Method')
    plt.ylabel('Mean Squared Error')
    plt.title('Comparison of Validation Methods Across Models')
    plt.xticks(x + width * len(models) / 2, methods)
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_methods_comparison.png')
    plt.show()

    # Save validation results to a file
    validation_results = convert_numpy_ints(validation_results)
    with open('validation_results_comparison.txt', 'w') as file:
        json.dump(validation_results, file, indent=4)

    # Print validation results
    print(json.dumps(validation_results, indent=4))

