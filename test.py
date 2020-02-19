# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:02:19 2019

@author: Ankit
"""

'''testing prophet on a new dataset of delhi air pollution to check its forecasting capabiliyu'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from numpy import mean
from scipy import stats
import statsmodels.api as sm
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import time

from statsmodels.tools.eval_measures import rmse,mse
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



#Sub0indices calcuations have been done using excel on testset1.csv and now reading the updated file with subindices 
testset = pd.read_csv("testset2.csv",parse_dates=True) 
#converting the date column in the proper format
testset['date'] = pd.to_datetime(testset['date'], format="%d-%m-%Y")
#taking aggregate valuees to daily
def positive_average(num):
    return num[num > 0].mean()
#aggregating hourly values into daily by applying positive average
testset = testset.drop('time', axis=1).groupby('date').apply(positive_average)

  #reading the new dataset of delhi air pollution
testset.info()

#removing old columns and keeping calculated sub-indices columns
columnstodrop = ['SO2','CO','Ozone','PM2.5','NO2']
testset = testset.drop(columnstodrop, axis=1)
#making the AQI columns
testset["AQI"] = testset.max(axis=1) 

#plotting shows the value of PM2.5 is the AQI
testset['pm2.5'].plot(figsize=(12,8),legend=True)
testset['AQI'].plot(legend=True)
#dropping columns apart from AQI
columnstodrop = ['so2','co','ozone','pm2.5','no2']
testset = testset.drop(columnstodrop, axis=1)
testset.index.freq='D'
##################################################
testdata = testset.iloc[391:]
traindata = testset.iloc[:391]
###################################################


#testing the same prophet model on this dataset 

--------------------------------------------------------------------#
#prophet
from fbprophet import Prophet
import matplotlib.pyplot as plt

testset = testset.reset_index()
testset.columns = ['ds', 'y']
testset.head()

testset['ds'] = pd.to_datetime(testset['ds'])
testset.info()

#evaluation


train_data_prophet = testset.iloc[:414]
test_data_prophet = testset.iloc[414:]

start_time = time.time()
n = Prophet()
n.fit(train_data_prophet)

future_prophet = n.make_future_dataframe(periods=7,freq='D')
forecast_prophet = n.predict(future_prophet)
print("--- %s seconds ---" % (time.time() - start_time))

#to remove plotting error
pd.plotting.register_matplotlib_converters()
#plottinh
ax = forecast_prophet.plot(x='ds',y='yhat',label='Predictions',legend=True,figsize=(12,8))
test_data_prophet.plot(x='ds',y='y',label='True test data',legend=True,ax=ax,xlim=('2018-02-19','2018-02-25'))


prediction_prophet = forecast_prophet.iloc[-7:]
prediction_prophet = prediction_prophet.filter(['ds','yhat'],axis=1)
prediction_prophet=prediction_prophet.set_index('ds')



#general evaluaiton
mse_prophet = mean_squared_error(test_data_prophet['y'],prediction_prophet['yhat'])
mae_prophet = mean_absolute_error(test_data_prophet['y'],prediction_prophet['yhat'])
rmse_prophet = rmse(test_data_prophet['y'],prediction_prophet['yhat'])
mape_prophet = mean_absolute_percentage_error(test_data_prophet['y'],prediction_prophet['yhat'])

#rmse = 34 mse=1169 mae=29 mape=7 for 7 day forecast
#so the    



