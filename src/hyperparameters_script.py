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
import random

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
    model.add(LSTM(int(units), return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(int(units), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_evaluate_lstm(X_train, y_train, X_test, y_test, params, time_step=60):
    model = create_lstm_model(units=params['units'], optimizer=params['optimizer'], time_step=time_step)
    model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def create_gru_model(units=50, optimizer='adam', time_step=60):
    model = Sequential()
    model.add(GRU(int(units), return_sequences=True, input_shape=(time_step, 1)))
    model.add(GRU(int(units), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_evaluate_gru(X_train, y_train, X_test, y_test, params, time_step=60):
    model = create_gru_model(units=params['units'], optimizer=params['optimizer'], time_step=time_step)
    model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, params):
    model = SVR(**params)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    mse = mean_squared_error(y_test, y_pred)
    return mse

def train_and_evaluate_adaboost(X_train, y_train, X_test, y_test, params):
    model = AdaBoostRegressor(**params)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    mse = mean_squared_error(y_test, y_pred)
    return mse

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Define a function to evaluate a model with different validation techniques
def evaluate_with_validation_methods(model, X, y, best_params, time_step=60):
    results = {}

    # 80-20 Holdout Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model.set_params(**best_params)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    mse_holdout = mean_squared_error(y_test, y_pred)
    results['Holdout'] = mse_holdout

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5)
    mse_kfold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        mse_kfold += mean_squared_error(y_test, y_pred)
    mse_kfold /= 5
    results['KFold'] = mse_kfold

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_tscv = 0
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        mse_tscv += mean_squared_error(y_test, y_pred)
    mse_tscv /= 5
    results['TimeSeries'] = mse_tscv

    return results

# Function to convert numpy.int32 to int
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
    ticker = 'GOOGL'
    data = fetch_stock_data(ticker)

    scaled_data, scaler = preprocess_data(data, 'minmax')

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Define parameter grids for RandomizedSearchCV
    lstm_params = {
        'units': [50, 80, 100, 120, 150],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 150]
    }

    gru_params = {
        'units': [50, 80, 100, 120, 150],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 150]
    }

    svm_params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto']
    }

    adaboost_params = {
        'n_estimators': [40, 60, 80, 100],
        'learning_rate': [0.1, 0.5, 1, 1.5],
        'random_state': [10, 20, 30, 40, 50]
    }

    rf_params = {
        'n_estimators': [80, 100, 120, 150],
        'criterion': ['squared_error', 'absolute_error'],
        'random_state': [10, 20, 30, 40, 50],
        'min_samples_leaf': [1, 2, 4, 6]
    }

    # Hyperparameter optimization using RandomizedSearchCV
    best_params = {}

    tscv = TimeSeriesSplit(n_splits=5)

    # LSTM
    lstm_best_params = None
    lstm_best_mse = float('inf')
    for _ in range(10):  # Random search iterations
        params = {
            'units': np.random.choice(lstm_params['units']),
            'optimizer': np.random.choice(lstm_params['optimizer']),
            'batch_size': np.random.choice(lstm_params['batch_size']),
            'epochs': np.random.choice(lstm_params['epochs'])
        }
        mse = train_and_evaluate_lstm(X_train, y_train, X_test, y_test, params, time_step=time_step)
        if mse < lstm_best_mse:
            lstm_best_mse = mse
            lstm_best_params = params
    best_params['LSTM'] = {'params': lstm_best_params, 'mse': lstm_best_mse}

    # GRU
    gru_best_params = None
    gru_best_mse = float('inf')
    for _ in range(10):  # Random search iterations
        params = {
            'units': np.random.choice(gru_params['units']),
            'optimizer': np.random.choice(gru_params['optimizer']),
            'batch_size': np.random.choice(gru_params['batch_size']),
            'epochs': np.random.choice(gru_params['epochs'])
        }
        mse = train_and_evaluate_gru(X_train, y_train, X_test, y_test, params, time_step=time_step)
        if mse < gru_best_mse:
            gru_best_mse = mse
            gru_best_params = params
    best_params['GRU'] = {'params': gru_best_params, 'mse': gru_best_mse}

    # SVM
    svm = SVR()
    svm_random_search = RandomizedSearchCV(estimator=svm, param_distributions=svm_params, n_iter=50, cv=tscv, verbose=2, n_jobs=-1)
    svm_random_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    best_svm = svm_random_search.best_estimator_
    svm_val_predictions = best_svm.predict(X_test.reshape(X_test.shape[0], -1))
    svm_best_mse = mean_squared_error(y_test, svm_val_predictions)
    best_params['SVM'] = {'params': svm_random_search.best_params_, 'mse': svm_best_mse}

    # AdaBoost
    adaboost = AdaBoostRegressor()
    adaboost_random_search = RandomizedSearchCV(estimator=adaboost, param_distributions=adaboost_params, n_iter=50, cv=tscv, verbose=2, n_jobs=-1)
    adaboost_random_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    best_adaboost = adaboost_random_search.best_estimator_
    adaboost_val_predictions = best_adaboost.predict(X_test.reshape(X_test.shape[0], -1))
    adaboost_best_mse = mean_squared_error(y_test, adaboost_val_predictions)
    best_params['AdaBoost'] = {'params': adaboost_random_search.best_params_, 'mse': adaboost_best_mse}

    # Random Forest
    rf = RandomForestRegressor()
    rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, n_iter=50, cv=tscv, verbose=2, n_jobs=-1)
    rf_random_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    best_rf = rf_random_search.best_estimator_
    rf_val_predictions = best_rf.predict(X_test.reshape(X_test.shape[0], -1))
    rf_best_mse = mean_squared_error(y_test, rf_val_predictions)
    best_params['RandomForest'] = {'params': rf_random_search.best_params_, 'mse': rf_best_mse}

    # Convert numpy.int32 values to int
    best_params = convert_numpy_ints(best_params)

    # Save best parameters and MSEs to a file
    with open('best_params.txt', 'w') as file:
        json.dump(best_params, file, indent=4)

    # Print best parameters and MSEs
    print(json.dumps(best_params, indent=4))

    # Plot results
    algorithms = list(best_params.keys())
    mse_values = [best_params[algo]['mse'] for algo in algorithms]

    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, mse_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Algorithms')
    plt.ylabel('Mean Squared Error')
    plt.title('Comparison of Algorithm Performance')
    plt.savefig('algorithm_performance.png')
    plt.close()

    # Test the best hyperparameters on a random stock from 2015 to present using different validation methods
    ticker = 'AAPL'
    data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(data, 'minmax')
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Get the best hyperparameters found previously
    lstm_best_params = best_params['LSTM']['params']
    gru_best_params = best_params['GRU']['params']
    svm_best_params = best_params['SVM']['params']
    adaboost_best_params = best_params['AdaBoost']['params']
    rf_best_params = best_params['RandomForest']['params']

    # Evaluate each model with different validation methods
    lstm_model = create_lstm_model(**lstm_best_params, time_step=time_step)
    lstm_results = evaluate_with_validation_methods(lstm_model, X, y, lstm_best_params, time_step=time_step)
    print(f"LSTM Validation Results: {lstm_results}")

    gru_model = create_gru_model(**gru_best_params, time_step=time_step)
    gru_results = evaluate_with_validation_methods(gru_model, X, y, gru_best_params, time_step=time_step)
    print(f"GRU Validation Results: {gru_results}")

    svm_model = SVR(**svm_best_params)
    svm_results = evaluate_with_validation_methods(svm_model, X, y, svm_best_params, time_step=time_step)
    print(f"SVM Validation Results: {svm_results}")

    adaboost_model = AdaBoostRegressor(**adaboost_best_params)
    adaboost_results = evaluate_with_validation_methods(adaboost_model, X, y, adaboost_best_params, time_step=time_step)
    print(f"AdaBoost Validation Results: {adaboost_results}")

    rf_model = RandomForestRegressor(**rf_best_params)
    rf_results = evaluate_with_validation_methods(rf_model, X, y, rf_best_params, time_step=time_step)
    print(f"Random Forest Validation Results: {rf_results}")

    # Combine results for comparison
    validation_results = {
        'LSTM': lstm_results,
        'GRU': gru_results,
        'SVM': svm_results,
        'AdaBoost': adaboost_results,
        'RandomForest': rf_results
    }

    # Convert numpy.int32 values to int
    validation_results = convert_numpy_ints(validation_results)

    # Save validation results to a file
    with open('validation_results.txt', 'w') as file:
        json.dump(validation_results, file, indent=4)

    # Print validation results
    print(json.dumps(validation_results, indent=4))
