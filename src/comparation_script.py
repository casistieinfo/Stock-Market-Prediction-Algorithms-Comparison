import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, GRU
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os

class BaseModel:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def calculate_percentage_accuracy(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mean_actual = np.mean(y_true)
        percentage_error = (rmse / mean_actual) * 100
        percentage_accuracy = 100 - percentage_error
        return percentage_accuracy 

class LSTM_Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM'
        self.model = self.build_model()
        self.configure_gpu()

    def configure_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU available, using CPU.")

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=150, return_sequences=True, input_shape=(60, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=150))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def serialize_data(self, historical_data):
        data = historical_data['Close'].values
        return data.reshape(-1, 1)

    def normalize_data(self, data):
        return self.scaler.fit_transform(data)

    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def split_data(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :]
        test_data = data[train_size:, :]
        return train_data, test_data

    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, historical_data):
        data = self.serialize_data(historical_data)
        normalized_data = self.normalize_data(data)

        train_data, test_data = self.split_data(normalized_data)
        X_train, y_train = self.create_dataset(train_data)
        X_test, y_test = self.create_dataset(test_data)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        self.train_model(X_train, y_train)

        test_predictions = self.model.predict(X_test)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        percentage_accuracy = self.calculate_percentage_accuracy(y_test_actual, test_predictions)
        return percentage_accuracy

class RandomForest_Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = 'RF'
        self.model = RandomForestRegressor(n_estimators=80, criterion='absolute_error', min_samples_leaf=1, random_state=40)

    def add_features(self, historical_data):
        historical_data['MA_10'] = historical_data['Close'].rolling(window=10).mean()
        historical_data['MA_50'] = historical_data['Close'].rolling(window=50).mean()
        historical_data = historical_data.dropna()
        return historical_data

    def serialize_data(self, historical_data):
        historical_data = self.add_features(historical_data)
        data = historical_data[['Close', 'MA_10', 'MA_50']].values
        return data

    def normalize_data(self, data):
        return self.scaler.fit_transform(data)

    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def split_data(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :]
        test_data = data[train_size:, :]
        return train_data, test_data

    def train_model(self, X_train, y_train):
        self.model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    def predict(self, historical_data):
        data = self.serialize_data(historical_data)
        normalized_data = self.normalize_data(data)

        train_data, test_data = self.split_data(normalized_data)
        X_train, y_train = self.create_dataset(train_data)
        X_test, y_test = self.create_dataset(test_data)

        self.train_model(X_train, y_train)

        test_predictions = self.model.predict(X_test.reshape(X_test.shape[0], -1))
        test_predictions = self.scaler.inverse_transform(np.c_[test_predictions, np.zeros((len(test_predictions), 2))])[:, 0]
        test_predictions = np.maximum(test_predictions, 0)  # Ensure non-negative predictions
        y_test_actual = self.scaler.inverse_transform(np.c_[y_test, np.zeros((len(y_test), 2))])[:, 0]

        percentage_accuracy = self.calculate_percentage_accuracy(y_test_actual, test_predictions)
        return percentage_accuracy

class GRU_Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = 'GRU'
        self.model = self.build_model()
        self.configure_gpu()

    def configure_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU available, using CPU.")

    def build_model(self):
        model = Sequential()
        model.add(GRU(units=150, return_sequences=True, input_shape=(60, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=150))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def serialize_data(self, historical_data):
        data = historical_data['Close'].values
        return data.reshape(-1, 1)

    def normalize_data(self, data):
        return self.scaler.fit_transform(data)

    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def split_data(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :]
        test_data = data[train_size:, :]
        return train_data, test_data

    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, historical_data):
        data = self.serialize_data(historical_data)
        normalized_data = self.normalize_data(data)

        train_data, test_data = self.split_data(normalized_data)
        X_train, y_train = self.create_dataset(train_data)
        X_test, y_test = self.create_dataset(test_data)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        self.train_model(X_train, y_train)

        test_predictions = self.model.predict(X_test)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        percentage_accuracy = self.calculate_percentage_accuracy(y_test_actual, test_predictions)
        return percentage_accuracy

class SVM_Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = 'SVM'
        self.model = SVR(C=1, kernel='linear', degree=2)

    def serialize_data(self, historical_data):
        data = historical_data['Close'].values
        return data.reshape(-1, 1)

    def normalize_data(self, data):
        return self.scaler.fit_transform(data)

    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def split_data(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :]
        test_data = data[train_size:, :]
        return train_data, test_data

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, historical_data):
        data = self.serialize_data(historical_data)
        normalized_data = self.normalize_data(data)

        train_data, test_data = self.split_data(normalized_data)
        X_train, y_train = self.create_dataset(train_data)
        X_test, y_test = self.create_dataset(test_data)

        self.train_model(X_train.reshape(X_train.shape[0], -1), y_train)

        test_predictions = self.model.predict(X_test.reshape(X_test.shape[0], -1))
        test_predictions = self.scaler.inverse_transform(test_predictions.reshape(-1, 1))
        test_predictions = np.maximum(test_predictions, 0)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        percentage_accuracy = self.calculate_percentage_accuracy(y_test_actual, test_predictions)
        return percentage_accuracy

class ADABoost_Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = 'ADAB'
        self.model = AdaBoostRegressor(n_estimators=40, learning_rate=1.5, random_state=50)

    def serialize_data(self, historical_data):
        data = historical_data['Close'].values
        return data.reshape(-1, 1)

    def normalize_data(self, data):
        return self.scaler.fit_transform(data)

    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def split_data(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :]
        test_data = data[train_size:, :]
        return train_data, test_data

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, historical_data):
        data = self.serialize_data(historical_data)
        normalized_data = self.normalize_data(data)

        train_data, test_data = self.split_data(normalized_data)
        X_train, y_train = self.create_dataset(train_data)
        X_test, y_test = self.create_dataset(test_data)

        self.train_model(X_train.reshape(X_train.shape[0], -1), y_train)

        test_predictions = self.model.predict(X_test.reshape(X_test.shape[0], -1))
        test_predictions = self.scaler.inverse_transform(test_predictions.reshape(-1, 1))
        test_predictions = np.maximum(test_predictions, 0)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        percentage_accuracy = self.calculate_percentage_accuracy(y_test_actual, test_predictions)
        return percentage_accuracy

def fetch_stock_data(ticker, start_date):
    stock_data = yf.download(ticker, start=start_date)
    return stock_data

def process_dataset(dataset_name, tickers, start_date, end_date, models, output_dir):
    results = {}
    total_rows = 0
    stock_count = 0

    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    results_file = os.path.join(dataset_output_dir, 'results.txt')
    
    with open(results_file, 'w') as f:
        for model in models:
            accuracy_list = []
            f.write(f"Evaluating model: {model.name}\n")
            for ticker in tickers:
                print(f"Processing {ticker}...")
                stock_data = fetch_stock_data(ticker, start_date)
                if not stock_data.empty:
                    total_rows += len(stock_data)
                    stock_count += 1
                    accuracy = model.predict(stock_data)
                    accuracy_list.append(accuracy)
                    f.write(f"{ticker} Accuracy: {accuracy:.2f}%\n")
                else:
                    f.write(f"{ticker} data not available.\n")
            results[model.name] = accuracy_list
            avg_accuracy = np.mean(accuracy_list)
            f.write(f"Average Accuracy for {model.name}: {avg_accuracy:.2f}%\n\n")
            print(f"Average Accuracy for {model.name}: {avg_accuracy:.2f}%")

        # Plotting the results
        bar_width = 0.15
        index = np.arange(len(tickers))

        plt.figure(figsize=(15, 8))

        for i, model_name in enumerate(results.keys()):
            plt.bar(index + i * bar_width, results[model_name], bar_width, label=model_name)

        plt.xlabel('Stock Ticker')
        plt.ylabel('Percentage Accuracy')
        plt.title(f'Comparison of Percentage Accuracy Across Different Models for Each Stock in {dataset_name}')
        plt.xticks(index + 2 * bar_width, tickers, rotation=90)
        plt.legend(title='Model')
        plt.grid(True)

        plot_file = os.path.join(dataset_output_dir, 'model_comparison.png')
        plt.savefig(plot_file)
        plt.close()

    return results, total_rows, stock_count


def main():
    # Define tickers for each category
    tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    finance_tickers = ['JPM', 'BAC', 'WFC', 'C', 'GS']
    automotive_tickers = ['TSLA', 'GM', 'F', 'TM', 'HMC']
    medical_tickers = ['JNJ', 'PFE', 'MRK', 'ABT', 'TMO']
    alimentation_tickers = ['KO', 'PEP', 'MDLZ', 'GIS', 'KHC']
    cosmetics_tickers = ['EL', 'PG', 'UL', 'CL', 'RBGLY']
    clothing_tickers = ['NKE', 'ADIDAS', 'LULU', 'UAA', 'RL']

    datasets = {
        "Tech Companies": (tech_tickers, "2021-01-01", None),
        "Finance Companies": (finance_tickers, "2021-01-01", None),
        "Automotive Companies": (automotive_tickers, "2021-01-01", None),
        "Medical Companies": (medical_tickers, "2021-01-01", None),
        "Alimentation Companies": (alimentation_tickers, "2021-01-01", None),
        "Cosmetics Companies": (cosmetics_tickers, "2021-01-01", None),
        "Clothing Companies": (clothing_tickers, "2021-01-01", None)
    }

    models = [LSTM_Model(), RandomForest_Model(), GRU_Model(), SVM_Model(), ADABoost_Model()]
    output_dir = 'model_results'
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_results = {}
    dataset_details = {}

    for dataset_name, (tickers, start_date, end_date) in datasets.items():
        print(f"Processing dataset: {dataset_name}")
        results, total_rows, stock_count = process_dataset(dataset_name, tickers, start_date, end_date, models, output_dir)
        dataset_results[dataset_name] = results
        dataset_details[dataset_name] = {
            'total_rows': total_rows,
            'stock_count': stock_count,
            'columns': list(fetch_stock_data(tickers[0], start_date).columns)
        }

    # Plot average accuracies for each model across all datasets
    avg_accuracies = {model.name: [] for model in models}

    for dataset_name, results in dataset_results.items():
        for model_name, accuracies in results.items():
            avg_accuracy = np.mean(accuracies)
            avg_accuracies[model_name].append(avg_accuracy)

    plt.figure(figsize=(10, 6))
    dataset_names = list(datasets.keys())
    bar_width = 0.15
    index = np.arange(len(dataset_names))

    for i, model_name in enumerate(avg_accuracies.keys()):
        plt.bar(index + i * bar_width, avg_accuracies[model_name], bar_width, label=model_name)

    plt.xlabel('Dataset')
    plt.ylabel('Average Percentage Accuracy')
    plt.title('Average Accuracy of Algorithms Across Different Datasets')
    plt.xticks(index + 2 * bar_width, dataset_names, rotation=45)
    plt.legend(title='Model')
    plt.grid(True)

    plot_file = os.path.join(output_dir, 'average_accuracy_across_datasets.png')
    plt.savefig(plot_file)
    plt.close()

    # Output dataset details
    details_file = os.path.join(output_dir, 'dataset_details.txt')
    with open(details_file, 'w') as f:
        for dataset_name, details in dataset_details.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Number of Rows: {details['total_rows']}\n")
            f.write(f"Number of Stocks: {details['stock_count']}\n")
            f.write(f"Number of Columns: {len(details['columns'])}\n")
            f.write(f"Columns: {', '.join(details['columns'])}\n\n")

if __name__ == "__main__":
    main()
