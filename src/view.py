import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import matplotlib.dates as mdates

class View:
    def __init__(self, root, models):
        self.root = root
        self.models = models
        self.root.title("Stock Market Prediction")

        self.input_frame = ttk.Frame(self.root)
        self.input_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor=tk.N)

        self.index_label = ttk.Label(self.input_frame, text="Select Index:")
        self.index_label.grid(row=0, column=0, padx=5, pady=5)
        self.index = ttk.Combobox(self.input_frame, values=["S&P 500", "NASDAQ", "DJIA"])
        self.index.grid(row=0, column=1, padx=5, pady=5)
        self.index.bind('<<ComboboxSelected>>', self.update_stock_list)

        self.stock_checkbox_var = tk.BooleanVar()
        self.stock_checkbox = ttk.Checkbutton(self.input_frame, text="Select Individual Stock", variable=self.stock_checkbox_var, command=self.toggle_stock_selection)
        self.stock_checkbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.stock_name_label = ttk.Label(self.input_frame, text="Stock Name:")
        self.stock_name_label.grid(row=2, column=0, padx=5, pady=5)
        self.stock_name = ttk.Combobox(self.input_frame)
        self.stock_name.grid(row=2, column=1, padx=5, pady=5)
        self.stock_name.bind('<KeyRelease>', self.update_combobox)
        self.stock_name.config(state='disabled')

        self.start_date_label = ttk.Label(self.input_frame, text="Start Date:")
        self.start_date_label.grid(row=3, column=0, padx=5, pady=5)
        self.start_date = DateEntry(self.input_frame)
        self.start_date.grid(row=3, column=1, padx=5, pady=5)

        self.time_to_predict_label = ttk.Label(self.input_frame, text="Time to Predict:")
        self.time_to_predict_label.grid(row=4, column=0, padx=5, pady=5)
        self.time_to_predict = ttk.Combobox(self.input_frame, values=["1 day", "1 week", "1 month", "3 months"])
        self.time_to_predict.grid(row=4, column=1, padx=5, pady=5)

        self.accuracy_label = ttk.Label(self.input_frame, text="Accuracy:")
        self.accuracy_label.grid(row=5, column=0, padx=5, pady=5, columnspan=2)

        self.models_frame = ttk.Frame(self.root)
        self.models_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor=tk.N)

        self.models_label = ttk.Label(self.models_frame, text="Select Models:")
        self.models_label.grid(row=0, column=0, padx=5, pady=5, columnspan=2)

        self.model_vars = {}
        i = 1
        for key, value in self.models.items():
            self.model_vars[key] = tk.BooleanVar()
            ttk.Checkbutton(self.models_frame, text=value.name, variable=self.model_vars[key]).grid(row=i, column=0, padx=5, pady=5)
            i += 1

        self.submit_button = ttk.Button(self.models_frame, text="Submit", command=self.on_submit)
        self.submit_button.grid(row=i, column=0, padx=5, pady=5, columnspan=2)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(padx=10, pady=10)

        self.figure = plt.Figure(figsize=(14, 6), dpi=100)  # Adjusted the figure size
        self.figure.subplots_adjust(left=0.045, right=0.684)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack()

        self.metrics_frame = ttk.Frame(self.root)
        self.metrics_frame.pack(padx=10, pady=10)

        self.metrics_table = ttk.Treeview(self.metrics_frame, columns=("Algorithm", "MAE", "MSE", "RMSE", "MAPE", "R2"), show='headings')
        self.metrics_table.heading("Algorithm", text="Algorithm")
        self.metrics_table.heading("MAE", text="MAE")
        self.metrics_table.heading("MSE", text="MSE")
        self.metrics_table.heading("RMSE", text="RMSE")
        self.metrics_table.heading("MAPE", text="MAPE")
        self.metrics_table.heading("R2", text="R2")
        self.metrics_table.pack()

        self.progress_frame = ttk.Frame(self.root)
        self.progress_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.progress = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(fill=tk.X)

        self.all_stock_names = []

    def on_submit(self):
        index = self.index.get()
        stock_name = self.stock_name.get() if self.stock_checkbox_var.get() else None
        start_date = self.start_date.get_date()
        time_to_predict = self.time_to_predict.get()

        selected_models = [model for model, var in self.model_vars.items() if var.get()]

        submission_data = {
            'index': index,
            'stock_name': stock_name,
            'start_date': start_date,
            'time_to_predict': time_to_predict,
            'selected_models': selected_models
        }

        self.submit_callback(submission_data)

    def set_submit_callback(self, callback):
        self.submit_callback = callback

    def ensure_datetime_index(self, data):
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize(None)
        return data

    def update_plot(self, historical_data, predictions, future_predictions):
        self.ax.clear()

        # Handling multiple stocks in historical_data
        if isinstance(historical_data, dict):
            for stock, data in historical_data.items():
                data = self.ensure_datetime_index(data)
                self.ax.plot(data.index, data['Close'], label=f'Historical Data ({stock})')
        else:
            historical_data = self.ensure_datetime_index(historical_data)
            self.ax.plot(historical_data.index, historical_data['Close'], label='Historical Data')

        for model, prediction in predictions.items():
            for stock, pred in prediction.items():
                if 'Close' in pred.columns:
                    pred = self.ensure_datetime_index(pred)
                    self.ax.plot(pred.index, pred['Close'], label=f'{model} (Test Predictions) ({stock})')

                    future_pred = future_predictions[model][stock]
                    future_dates = pd.date_range(start=pred.index[-1], periods=len(future_pred))
                    future_df = pd.DataFrame(future_pred, index=future_dates, columns=['Predicted Price'])
                    self.ax.plot(future_df.index, future_df['Predicted Price'], label=f'{model} (Future Predictions) ({stock})')

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        self.canvas.draw()

    def update_metrics(self, metrics):
        self.metrics_table.delete(*self.metrics_table.get_children())
        accuracy_text = ""

        for model, model_metrics in metrics.items():
            for stock, stock_metrics in model_metrics.items():
                self.metrics_table.insert('', 'end', text=f'{model} ({stock})', values=(
                    model,  
                    stock_metrics['MAE'], stock_metrics['MSE'], stock_metrics['RMSE'],
                    stock_metrics['MAPE'], stock_metrics['R2']
                ))

                accuracy = 100 - stock_metrics['MAPE']
                accuracy_text += f"{model} ({stock}): {accuracy:.2f}%\n"

        self.accuracy_label.config(text=f"Accuracy:\n{accuracy_text}")

    def update_combobox(self, event):
        input_text = self.stock_name.get().lower()
        if input_text == "":
            self.stock_name['values'] = self.all_stock_names
        else:
            filtered_list = [name for name in self.all_stock_names if input_text in name.lower()]
            self.stock_name['values'] = filtered_list
        self.stock_name.event_generate('<Down>')

    def update_stock_list(self, event):
        selected_index = self.index.get()
        self.submit_callback({'action': 'update_stock_list', 'index': selected_index})

    def toggle_stock_selection(self):
        state = 'normal' if self.stock_checkbox_var.get() else 'disabled'
        self.stock_name.config(state=state)

    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()
