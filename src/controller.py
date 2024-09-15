import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
from view import View
from models import *
import threading
import numpy as np

class Controller:
    def __init__(self, root):
        self.models = {
            'LSTM': LSTM_Model(),
            'RandomForest': RandomForest_Model(),
            'GRU': GRU_Model(),
            'SVM': SVM_Model(),
            'ADABoost': ADABoost_Model()
        }
        self.view = View(root, self.models)
        self.view.set_submit_callback(self.handle_submit)
        self.name_to_symbol = {}

        self.view.models = self.models

    def fetch_sp500_tickers(self):
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_df = table[0]
        tickers = sp500_df['Symbol'].tolist()
        return tickers

    def fetch_nasdaq_tickers(self):
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        table = pd.read_html(url)
        nasdaq_df = table[4]  # Usually, NASDAQ-100 is the 5th table
        tickers = nasdaq_df['Ticker'].tolist()
        return tickers

    def fetch_djia_tickers(self):
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        table = pd.read_html(url)
        djia_df = table[1]  # The DJIA companies are in the 2nd table
        tickers = djia_df['Symbol'].tolist()
        return tickers

    def fetch_stock_names(self, tickers):
        stock_dict = {}
        for i, ticker in enumerate(tickers):
            try:
                stock_info = yf.Ticker(ticker).info
                short_name = stock_info.get('shortName', 'N/A')
                stock_dict[ticker] = short_name
            except Exception as e:
                print(f"Could not fetch data for ticker {ticker}: {e}")

            progress_value = (i + 1) / len(tickers) * 100
            self.view.update_progress(progress_value)

        return stock_dict

    def populate_stock_names(self, index):
        threading.Thread(target=self._populate_stock_names_thread, args=(index,)).start()

    def _populate_stock_names_thread(self, index):
        if index == 'S&P 500':
            tickers = self.fetch_sp500_tickers()
        elif index == 'NASDAQ':
            tickers = self.fetch_nasdaq_tickers()
        elif index == 'DJIA':
            tickers = self.fetch_djia_tickers()

        stock_dict = self.fetch_stock_names(tickers)
        self.name_to_symbol = {v: k for k, v in stock_dict.items()}
        self.view.stock_name['values'] = list(stock_dict.values())
        self.view.all_stock_names = list(stock_dict.values())
        self.view.update_progress(100)
        self.view.update_progress(0)

    def handle_submit(self, data):
        if data.get('action') == 'update_stock_list':
            self.populate_stock_names(data['index'])
        else:
            stock_name = data['stock_name']
            if data.get('use_individual_stocks', False) and not stock_name:
                messagebox.showerror("Error", "Please select a stock if using individual stocks option.")
                return

            threading.Thread(target=self._handle_submit_thread, args=(data,)).start()

    def _handle_submit_thread(self, data):
        index = data['index']
        stock_name = self.name_to_symbol.get(data['stock_name']) if data['stock_name'] else None
        start_date = data['start_date']
        time_to_predict = data['time_to_predict']
        model_names = data['selected_models']

        historical_data = self.fetch_historical_data(stock_name, index, start_date)
        
        if not historical_data:
            messagebox.showerror("Error", "No historical data found for the selected parameters.")
            return

        predictions = {model_name: {} for model_name in model_names}
        future_predictions = {model_name: {} for model_name in model_names}
        metrics = {model_name: {} for model_name in model_names}

        progress_step = 100 / len(model_names) if model_names else 100
        for i, model_name in enumerate(model_names):
            model = self.models[model_name]
            for stock, data in historical_data.items():
                try:
                    test_pred_df, future_pred, model_metrics = model.predict({stock: data}, time_to_predict, self.view.update_progress, progress_step, i)
                    predictions[model_name][stock] = test_pred_df
                    future_predictions[model_name][stock] = future_pred
                    metrics[model_name][stock] = model_metrics[stock]  # Extract metrics for each stock
                except Exception as e:
                    print(f"Error processing stock {stock} with model {model_name}: {e}")

        self.view.update_plot(historical_data, predictions, future_predictions)
        self.view.update_metrics(metrics)
        self.view.update_progress(100)
        self.view.update_progress(0)

    def fetch_historical_data(self, stock_name, index, start_date):
        if stock_name:
            stock = yf.Ticker(stock_name)
            historical_data = stock.history(start=start_date, end=pd.Timestamp.now().strftime('%Y-%m-%d'))
            return {stock_name: historical_data}
        else:
            if index == 'S&P 500':
                tickers = self.fetch_sp500_tickers()
            elif index == 'NASDAQ':
                tickers = self.fetch_nasdaq_tickers()
            elif index == 'DJIA':
                tickers = self.fetch_djia_tickers()
            else:
                return None

            historical_data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=pd.Timestamp.now().strftime('%Y-%m-%d'))
                if 'Close' in data.columns:  # Ensure the data has the 'Close' column
                    historical_data[ticker] = data

            return historical_data

    def predict(self, historical_data, model_names, time_to_predict):
        predictions = {}
        metrics = {}
        for model_name in model_names:
            model = self.models[model_name]
            prediction, model_metrics = model.predict(historical_data, time_to_predict)
            predictions[model_name] = pd.DataFrame(prediction, columns=['Close'])
            metrics[model_name] = model_metrics
        return predictions, metrics
