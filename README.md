# Stock Price Prediction of Apple Inc and Tesla Inc. Using Recurrent Neural Network
OHLC Average Prediction of Apple Inc. Using LSTM Recurrent Neural Network

# Dataset:
The dataset is taken from yahoo finace's website in CSV format. The dataset consists of Open, High, Low and Closing Prices of Apple and Tesla stocks from 3rd January 2017 to 28th June 2024

# Price Indicator:
Stock traders mainly use three indicators for prediction: OHLC average (average of Open, High, Low and Closing Prices), HLC average (average of High, Low and Closing Prices) and Closing price, In this project, OHLC average has been used.

# Data Pre-processing:
After converting the dataset into OHLC average, it becomes one column data. This has been converted into two column time series data, 1st column consisting stock price of time t, and second column of time t+1. All values have been normalized between 0 and 1.

# Model: 
Two sequential LSTM layers have been stacked together and one dense layer is used to build the RNN model using Keras deep learning library. Since this is a regression task, 'linear' activation has been used in final layer.

# Test:
Test accuracy metric is root mean square error (RMSE).
# Results: (Only for Apple)
The comparison of OHLC, HLC and Closing price:

![ttt1](https://github.com/dkconnect/Stock-Prediction/blob/main/apple/apple_comparision_17-24.png)

After the training the fitted curve with original stock price:

![tt2](https://github.com/dkconnect/Stock-Prediction/blob/main/apple/apple_prediction_17-24.png)

# Observation and Conclusion:
Since difference among OHLC average, HLC average and closing value is not significat, so only OHLC average is used to build the model and prediction.
This work can greatly help the quantitative traders to take decisions.

