# Forecast Exchange Rate

![GitHub top language](https://img.shields.io/github/languages/top/raulincadet/forex?style=plastic)
![GitHub repo size](https://img.shields.io/github/repo-size/raulincadet/forex?color=green)
![GitHub language count](https://img.shields.io/github/languages/count/raulincadet/forex?style=flat-square)
![Lines of code](https://img.shields.io/tokei/lines/github/raulincadet/forex?color=orange&style=plastic)
![forex](dashboard.png)

===========

## Description
This is a machine learning project that forecast the daily exchange rate between the most traded currencies. 
The dashboard produced by this project has these features:
* The user can select the two currencies for which he wants to forecast the daily exchange rate.
* A currency converter is provided, allowing the user to add the amount of the first chosen currency he wants to convert to the other currency.
* The values (Open, Low, High, Close) of the exchange rate for the currencies selected by the user
are provided in a table, for the last 10 days.
* The forecasted values of the exchange rate are provided in a table.
* A graphic present the historical values of the exchange rate, for the last 90 days, and the forecasted values for 10 future days.

To navigate through the dashboard, click [HERE](https://raulincadet-forex-forex-forecast-gk6cpq.streamlitapp.com/).

## Modules required to run the code
* pandas
* numpy
* matplotlib
* scipy
* scikit-learn
* requests
* seaborn
* statsmodels
* streamlit

## Method

As the user interacts with the dashboard, real time data are automatically collected and future values are forecasted.

### API required to retreive data
To realize this project, data are collected, using two APIs. One from Alpha Vantage to get daily exchange rate, and another One
from the FRED API of the Federal Reserve Bank of St. Louis. The FRED API is used to retreive daily time series to be used as features
to train and test the machine learning algorithm chosen. These APIs are used in the project without any dedicated python module. Data imported
have been converted to a structured data frame, using the modules json, requests, and pandas.

### Machine learning
* Data are splitted to train and test sets
* The GradientBoostingRegressor algorithm of scikit-learn is used to train the model
* A range of 10 values, for the learning rate is used in a loop, and the root squared... is calculated.
* The learning rate with the lower rmse is kept as the best model
* To forecast future values of the exchange rate, future values of the features are forecasted using the ExponentialSmoothing
algorithm.


