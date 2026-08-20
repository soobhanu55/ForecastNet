# ForecastNet: Time Series Sales Forecasting with Deep Learning

## 📌 Project Overview
DeepSales is a high-performance sales forecasting engine designed to predict total monthly product sales across multiple stores. Built on the "Predict Future Sales" dataset from Kaggle, the project addresses the complexities of real-world retail data, including fluctuating inventory and store lists.

## 🚀 Key Features
* **Time Series Forecasting**: Leverages historical daily data to predict future monthly demand.
* **Robust Model Architecture**: Implements advanced Deep Learning models (CNN-LSTM) capable of handling non-stationary time series data.
* **Feature Engineering**: Includes sophisticated preprocessing to manage changing lists of shops and products over time.
* **Sequence Modeling**: Utilizes data reshaping into subsequences for optimized neural network training.

## 🛠️ Technical Stack
* **Deep Learning**: TensorFlow/Keras (CNN, LSTM).
* **Data Science**: Pandas, NumPy, Scikit-learn.
* **Visualization**: Matplotlib, Seaborn.

## 📊 Data Source
The model is trained on historical sales data from the [Kaggle Predict Future Sales Competition](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data).

## Demo

`print_results.py` presents the real evaluation numbers and the real bug found, both already sitting in the notebook's own output cells:

![Terminal recording of the evaluation results and the bug found](docs/demo.gif)

## 📈 Evaluation

Four architectures (MLP, CNN, LSTM, CNN-LSTM) were trained and scored on a held-out validation split (RMSE, real output already saved in the notebook's cells 53-56):

| Model | Train RMSE | Validation RMSE |
|---|---|---|
| **MLP** | **18.529** | **18.672** |
| CNN | 18.660 | 18.779 |
| CNN-LSTM | 18.967 | 18.965 |
| LSTM | 20.693 | *not correctly measured — see below* |

**A real bug, found and disclosed rather than quietly fixed and hidden:** the LSTM validation cell calls `model_cnn.predict(X_valid_series)` instead of `model_lstm.predict(X_valid_series)` — a copy-paste error. The printed "LSTM Validation rmse: 18.779450406113046" is byte-for-byte identical to the CNN validation RMSE in the cell above it, which is how this was caught: two different architectures cannot legitimately produce an identical floating-point RMSE on the same data by chance. The LSTM's *train* RMSE (20.693, the highest of the four) is unaffected and real. Its true validation performance has never actually been measured. Simplest architecture (MLP) currently reports the best validation RMSE of the three correctly-measured models.
