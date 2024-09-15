import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
from keras.models import load_model, Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

class BaseModel:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def predict(self, historical_data, time_to_predict, update_progress=None, progress_step=0, model_index=0):
        pass

    def calculate_metrics(self, true_values, predicted_values):
        metrics = {
            'MAE': mean_absolute_error(true_values, predicted_values),
            'MSE': mean_squared_error(true_values, predicted_values),
            'RMSE': mean_squared_error(true_values, predicted_values, squared=False),
            'MAPE': self.mean_absolute_percentage_error(true_values, predicted_values),
            'R2': r2_score(true_values, predicted_values)
        }
        return metrics

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def add_noise_to_predictions(self, predictions, historical_data, scale=0.1):
        historical_volatility = np.std(historical_data['Close'].pct_change().dropna())
        noise = np.random.normal(0, historical_volatility * scale, size=predictions.shape)
        return predictions * (1 + noise)

    def aggregate_predictions(self, predictions_dict):
        aggregated_predictions = None
        for stock, predictions in predictions_dict.items():
            if aggregated_predictions is None:
                aggregated_predictions = predictions
            else:
                aggregated_predictions += predictions
        return aggregated_predictions / len(predictions_dict)

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
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping])
        return history

    def predict_future(self, last_sequence, n_steps):
        predicted = []
        current_seq = last_sequence
        for _ in range(n_steps):
            pred = self.model.predict(current_seq.reshape(1, current_seq.shape[0], 1))
            predicted.append(pred[0, 0])
            current_seq = np.append(current_seq[1:], pred)
        return predicted

    def get_prediction_horizon(self, time_to_predict):
        horizons = {'1 day': 1, '1 week': 7, '1 month': 30, '3 months': 90}
        return horizons.get(time_to_predict, 1)

    def predict(self, historical_data, time_to_predict, update_progress=None, progress_step=0, model_index=0):
        predictions_dict = {}
        future_predictions_dict = {}
        all_metrics = {}

        for stock, data in historical_data.items():
            serialized_data = self.serialize_data(data)
            normalized_data = self.normalize_data(serialized_data)

            train_data, test_data = self.split_data(normalized_data)
            X_train, y_train = self.create_dataset(train_data)
            X_test, y_test = self.create_dataset(test_data)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            self.train_model(X_train, y_train)

            test_predictions = self.model.predict(X_test)
            test_predictions = self.scaler.inverse_transform(test_predictions)
            y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

            metrics = self.calculate_metrics(y_test_actual, test_predictions)
            all_metrics[stock] = metrics
            test_dates = data.index[-len(y_test):]
            test_pred_df = pd.DataFrame(test_predictions, index=test_dates, columns=['Close'])
            predictions_dict[stock] = test_pred_df

            prediction_horizon = self.get_prediction_horizon(time_to_predict)
            last_sequence = normalized_data[-60:]
            future_predictions = self.predict_future(last_sequence, prediction_horizon)
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = self.scaler.inverse_transform(future_predictions)

            future_predictions = self.add_noise_to_predictions(future_predictions, data)
            future_predictions_dict[stock] = future_predictions

            if update_progress:
                update_progress(progress_step * (model_index + 0.5))

        aggregated_predictions = self.aggregate_predictions(predictions_dict)
        aggregated_future_predictions = self.aggregate_predictions(future_predictions_dict)

        if update_progress:
            update_progress(progress_step * (model_index + 1))

        return aggregated_predictions, aggregated_future_predictions, all_metrics

class RandomForest_Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = 'RF'
        self.model = RandomForestRegressor(n_estimators=80, criterion='absolute_error', min_samples_leaf=1, random_state=40)
    
    def serialize_data(self, historical_data):
        data = historical_data['Close'].values
        return data.reshape(-1, 1)
    
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

    def predict_future(self, last_sequence, n_steps):
        predicted = []
        current_seq = last_sequence.reshape(1, -1)
        for _ in range(n_steps):
            pred = self.model.predict(current_seq)
            predicted.append(pred[0])
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1] = pred[0]
        return predicted

    def get_prediction_horizon(self, time_to_predict):
        horizons = {'1 day': 1, '1 week': 7, '1 month': 30, '3 months': 90}
        return horizons.get(time_to_predict, 1)

    def predict(self, historical_data, time_to_predict, update_progress=None, progress_step=0, model_index=0):
        predictions_dict = {}
        future_predictions_dict = {}
        all_metrics = {}

        for stock, data in historical_data.items():
            serialized_data = self.serialize_data(data)
            normalized_data = self.normalize_data(serialized_data)

            train_data, test_data = self.split_data(normalized_data)
            X_train, y_train = self.create_dataset(train_data)
            X_test, y_test = self.create_dataset(test_data)

            self.train_model(X_train, y_train)

            test_predictions = self.model.predict(X_test.reshape(X_test.shape[0], -1))
            test_predictions = self.scaler.inverse_transform(np.c_[test_predictions, np.zeros((len(test_predictions), 1))])[:, 0]
            test_predictions = np.maximum(test_predictions, 0)
            y_test_actual = self.scaler.inverse_transform(np.c_[y_test, np.zeros((len(y_test), 1))])[:, 0]

            metrics = self.calculate_metrics(y_test_actual, test_predictions)
            all_metrics[stock] = metrics
            test_dates = data.index[-len(y_test):]
            test_pred_df = pd.DataFrame(test_predictions, index=test_dates, columns=['Close'])
            predictions_dict[stock] = test_pred_df

            prediction_horizon = self.get_prediction_horizon(time_to_predict)
            last_sequence = normalized_data[-60:]
            future_predictions = self.predict_future(last_sequence, prediction_horizon)
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = self.scaler.inverse_transform(np.c_[future_predictions, np.zeros((len(future_predictions), 1))])[:, 0]
            future_predictions = np.maximum(future_predictions, 0)

            future_predictions_dict[stock] = future_predictions

            if update_progress:
                update_progress(progress_step * (model_index + 0.5))

        aggregated_predictions = self.aggregate_predictions(predictions_dict)
        aggregated_future_predictions = self.aggregate_predictions(future_predictions_dict)

        if update_progress:
            update_progress(progress_step * (model_index + 1))

        return aggregated_predictions, aggregated_future_predictions, all_metrics

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
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping])
        return history

    def predict_future(self, last_sequence, n_steps):
        predicted = []
        current_seq = last_sequence
        for _ in range(n_steps):
            pred = self.model.predict(current_seq.reshape(1, current_seq.shape[0], 1))
            predicted.append(pred[0, 0])
            current_seq = np.append(current_seq[1:], pred)
        return predicted

    def get_prediction_horizon(self, time_to_predict):
        horizons = {'1 day': 1, '1 week': 7, '1 month': 30, '3 months': 90}
        return horizons.get(time_to_predict, 1)

    def predict(self, historical_data, time_to_predict, update_progress=None, progress_step=0, model_index=0):
        predictions_dict = {}
        future_predictions_dict = {}
        all_metrics = {}

        for stock, data in historical_data.items():
            serialized_data = self.serialize_data(data)
            normalized_data = self.normalize_data(serialized_data)

            train_data, test_data = self.split_data(normalized_data)
            X_train, y_train = self.create_dataset(train_data)
            X_test, y_test = self.create_dataset(test_data)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            self.train_model(X_train, y_train)

            test_predictions = self.model.predict(X_test)
            test_predictions = self.scaler.inverse_transform(test_predictions)
            y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

            metrics = self.calculate_metrics(y_test_actual, test_predictions)
            all_metrics[stock] = metrics
            test_dates = data.index[-len(y_test):]
            test_pred_df = pd.DataFrame(test_predictions, index=test_dates, columns=['Close'])
            predictions_dict[stock] = test_pred_df

            prediction_horizon = self.get_prediction_horizon(time_to_predict)
            last_sequence = normalized_data[-60:]
            future_predictions = self.predict_future(last_sequence, prediction_horizon)
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = self.scaler.inverse_transform(future_predictions)

            future_predictions = self.add_noise_to_predictions(future_predictions, data)
            future_predictions_dict[stock] = future_predictions

            if update_progress:
                update_progress(progress_step * (model_index + 0.5))

        aggregated_predictions = self.aggregate_predictions(predictions_dict)
        aggregated_future_predictions = self.aggregate_predictions(future_predictions_dict)

        if update_progress:
            update_progress(progress_step * (model_index + 1))

        return aggregated_predictions, aggregated_future_predictions, all_metrics

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
        self.model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    def predict_future(self, last_sequence, n_steps):
        predicted = []
        current_seq = last_sequence
        for _ in range(n_steps):
            pred = self.model.predict(current_seq.reshape(1, -1))
            predicted.append(pred[0])
            current_seq = np.append(current_seq[1:], pred)
        return predicted

    def get_prediction_horizon(self, time_to_predict):
        horizons = {'1 day': 1, '1 week': 7, '1 month': 30, '3 months': 90}
        return horizons.get(time_to_predict, 1)

    def predict(self, historical_data, time_to_predict, update_progress=None, progress_step=0, model_index=0):
        predictions_dict = {}
        future_predictions_dict = {}
        all_metrics = {}

        for stock, data in historical_data.items():
            serialized_data = self.serialize_data(data)
            normalized_data = self.normalize_data(serialized_data)

            train_data, test_data = self.split_data(normalized_data)
            X_train, y_train = self.create_dataset(train_data)
            X_test, y_test = self.create_dataset(test_data)

            self.train_model(X_train.reshape(X_train.shape[0], -1), y_train)

            test_predictions = self.model.predict(X_test.reshape(X_test.shape[0], -1))
            test_predictions = self.scaler.inverse_transform(test_predictions.reshape(-1, 1))
            test_predictions = np.maximum(test_predictions, 0)  # Ensure non-negative predictions
            y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

            metrics = self.calculate_metrics(y_test_actual, test_predictions)
            all_metrics[stock] = metrics
            test_dates = data.index[-len(y_test):]
            test_pred_df = pd.DataFrame(test_predictions, index=test_dates, columns=['Close'])
            predictions_dict[stock] = test_pred_df

            prediction_horizon = self.get_prediction_horizon(time_to_predict)
            last_sequence = normalized_data[-60:]
            future_predictions = self.predict_future(last_sequence, prediction_horizon)
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = self.scaler.inverse_transform(future_predictions)
            future_predictions = np.maximum(future_predictions, 0)  # Ensure non-negative predictions

            future_predictions = self.add_noise_to_predictions(future_predictions, data)
            future_predictions_dict[stock] = future_predictions

            if update_progress:
                update_progress(progress_step * (model_index + 0.5))

        aggregated_predictions = self.aggregate_predictions(predictions_dict)
        aggregated_future_predictions = self.aggregate_predictions(future_predictions_dict)

        if update_progress:
            update_progress(progress_step * (model_index + 1))

        return aggregated_predictions, aggregated_future_predictions, all_metrics

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
        self.model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    def predict_future(self, last_sequence, n_steps):
        predicted = []
        current_seq = last_sequence
        for _ in range(n_steps):
            pred = self.model.predict(current_seq.reshape(1, -1))
            predicted.append(pred[0])
            current_seq = np.append(current_seq[1:], pred)
        return predicted

    def get_prediction_horizon(self, time_to_predict):
        horizons = {'1 day': 1, '1 week': 7, '1 month': 30, '3 months': 90}
        return horizons.get(time_to_predict, 1)

    def predict(self, historical_data, time_to_predict, update_progress=None, progress_step=0, model_index=0):
        predictions_dict = {}
        future_predictions_dict = {}
        all_metrics = {}

        for stock, data in historical_data.items():
            serialized_data = self.serialize_data(data)
            normalized_data = self.normalize_data(serialized_data)

            train_data, test_data = self.split_data(normalized_data)
            X_train, y_train = self.create_dataset(train_data)
            X_test, y_test = self.create_dataset(test_data)

            self.train_model(X_train.reshape(X_train.shape[0], -1), y_train)

            test_predictions = self.model.predict(X_test.reshape(X_test.shape[0], -1))
            test_predictions = self.scaler.inverse_transform(test_predictions.reshape(-1, 1))
            test_predictions = np.maximum(test_predictions, 0)
            y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

            metrics = self.calculate_metrics(y_test_actual, test_predictions)
            all_metrics[stock] = metrics
            test_dates = data.index[-len(y_test):]
            test_pred_df = pd.DataFrame(test_predictions, index=test_dates, columns=['Close'])
            predictions_dict[stock] = test_pred_df

            prediction_horizon = self.get_prediction_horizon(time_to_predict)
            last_sequence = normalized_data[-60:]
            future_predictions = self.predict_future(last_sequence, prediction_horizon)
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = self.scaler.inverse_transform(future_predictions)
            future_predictions = np.maximum(future_predictions, 0)

            future_predictions = self.add_noise_to_predictions(future_predictions, data)
            future_predictions_dict[stock] = future_predictions

            if update_progress:
                update_progress(progress_step * (model_index + 0.5))

        aggregated_predictions = self.aggregate_predictions(predictions_dict)
        aggregated_future_predictions = self.aggregate_predictions(future_predictions_dict)

        if update_progress:
            update_progress(progress_step * (model_index + 1))

        return aggregated_predictions, aggregated_future_predictions, all_metrics