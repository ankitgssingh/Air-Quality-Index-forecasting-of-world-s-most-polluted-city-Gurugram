# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 05:58:47 2019

@author: Ankit
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import mean
sns.set()
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from tbats import BATS, TBATS
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#first preprocessing 
dataset = pd.read_csv("cleaned_using_interpolation.csv")   #reading the csv
dataset.head(10)
dataset.info()        #checking the column names and row names
dataset.shape            #checking the number of rows and columns

#parsing date into datetime format
dataset['date'] = pd.to_datetime(dataset['date'], format="%d-%m-%Y")

#turning all into float
for col in dataset.iloc[:,2:].columns:
    if dataset[col].dtypes == object:
        dataset[col] = dataset[col].str.replace(',', '.').astype('float')
        

#taking aggregate valuees to daily
def positive_average(num):
    return num[num > 0].mean()

daily_data = dataset.drop('time', axis=1).groupby('date').apply(positive_average)
daily_data.info()

daily_data.to_csv('ready.csv')

#Sub0indices calcuations have been done using excel on ready.csv and now reading the updated file with subindices 
dataset = pd.read_csv("ready_new.csv",parse_dates=True)   #reading the csv
dataset['date'] = pd.to_datetime(dataset['date'], format="%d-%m-%Y")
dataset= dataset.set_index('date')
dataset.info()

#removing old columns and keeping sub-indices columns
columnstodrop = ['SO2','CO','Ozone','PM2.5','NO2']
dataset = dataset.drop(columnstodrop, axis=1)
#making the AQI columns
dataset["AQI"] = dataset.max(axis=1) 

#plotting shows the value of PM2.5 is the AQI
dataset['pm2.5'].plot(figsize=(12,8),legend=True)
dataset['AQI'].plot(legend=True)

##################################################
''' Checking correlation'''
corr = dataset.corr()
ax = sns.heatmap(corr, annot=True)
#the correlation between AQI and pm2.5 is .99
###################################################
columnstodrop = ['so2','co','o3','pm2.5','no2']
dataset = dataset.drop(columnstodrop, axis=1)
dataset.index.freq='D'

#acf and pacf visual plot
from pandas.plotting import lag_plot
lag_plot(dataset['AQI'])

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(dataset['AQI'],lags=40) 
#only one lag is outside the CI so maybe stationary

plot_pacf(dataset['AQI'],lags=40,title='AQI')
#only one lag is outside the CI

#statistical summary test for stationarity (not so trustworthy)
X = dataset.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%.2f, mean2=%.2f' % (mean1, mean2))
print('variance1=%.2f, variance2=%.2f' % (var1, var2))
#mean and variance are different so maybe non stationarity but not sure

#does not look like a gaussian distribution 
dataset.hist()


# ADF testing for stationarity
from statsmodels.tsa.stattools import adfuller
#dickey fuller test
adfuller(dataset['AQI'])
dftest = adfuller(dataset['AQI'])
dfout = pd.Series(dftest[0:4],index=['ADF Test Statistics','p-value','# lags used','# observations'])
for key,val in dftest[4].items():
    dfout[f'critical value ({key})'] = val

dfout    
#data is coming to be stationary as p value is less than .05 and test stats is even lower than the crit value at 1 percent CI

#KPSS test for stationary
#define function for kpss test
from statsmodels.tsa.stattools import kpss
#define KPSS
def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

#p vlue is less than 0.05 and test stats is greater than the crit values so we reject the null hypothesis and it is not stationary
#But 
kpss_test(dataset['AQI'], regression='ct')
#this shows series is trend stationary 
#so by adf and kpss test we can say that it is stationary

#checking seasonality by ETS decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result_seasonality = seasonal_decompose(dataset['AQI'],model='mul')
result_seasonality.plot();
#no clear trend or seasonality is noticed

#train test split
train_data = dataset.iloc[:943]
test_data = dataset.iloc[943:]

#exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
start_time = time.time()
model_exp = ExponentialSmoothing(train_data['AQI'], trend='mul',seasonal='mul',seasonal_periods=4)
fitted_model_exp = model_exp.fit()
test_predictions_exp = fitted_model_exp.forecast(30)
print("--- %s seconds ---" % (time.time() - start_time))

train_data['AQI'].plot(legend=True,label='Train',figsize=(12,8))
test_data['AQI'].plot(legend=True,label='Test')
test_predictions_exp.plot(legend=True,label='Prediction')

#calling all evalaution metrics and defining mape
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tools.eval_measures import rmse,mse
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mse_exp = mean_squared_error(test_data,test_predictions_exp)
mae_exp = mean_absolute_error(test_data,test_predictions_exp)
rmse_exp = rmse(test_data['AQI'],test_predictions_exp)
mape_exp = mean_absolute_percentage_error(test_data, test_predictions_exp)
#comparing the standard deviation of the prediction with the real data
dataset.describe()


#Making future forecast
exp_forecast = ExponentialSmoothing(dataset['AQI'],trend='mul',seasonal='mul',seasonal_periods=4).fit()
exp_forecast_prediction = exp_forecast.forecast(30)
 
dataset['AQI'].plot(figsize=(12,8))
exp_forecast_prediction.plot(figsize=(12,8))
test_data['AQI'].plot(legend=True,label='Test')

#---------------------------------------------------

#auto regression model
from statsmodels.tsa.ar_model import AR,ARResults

#AR model order 1
model_ar = AR(train_data['AQI'])

AR1fit = model_ar.fit(maxlag=1)
AR1fit.params

start = len(train_data)
end = len(train_data) + len(test_data) - 1

prediction_ar1 = AR1fit.predict(start=start,end=end)
prediction_ar1 = prediction_ar1.rename('AR (1) Predictions')


test_data.plot(figsize =(12,8),legend=True)
prediction_ar1.plot(legend=True)


mse_ar1 = mean_squared_error(test_data,prediction_ar1)
mae_ar1 = mean_absolute_error(test_data,prediction_ar1)
rmse_ar1 = rmse(test_data['AQI'],prediction_ar1)
mape_ar1 = mean_absolute_percentage_error(test_data, prediction_ar1)

#order 2 AR
AR2fit = model_ar.fit(maxlag=2)
AR2fit.params

prediction_ar2 = AR2fit.predict(start,end)
prediction_ar2 = prediction_ar2.rename('AR (2) Predictions')

test_data.plot(figsize =(12,8),legend=True)
prediction_ar1.plot(legend=True)
prediction_ar2.plot(legend=True)


mse_ar2 = mean_squared_error(test_data,prediction_ar2)
mae_ar2 = mean_absolute_error(test_data,prediction_ar2)
rmse_ar2 = rmse(test_data['AQI'],prediction_ar2)
mape_ar2 = mean_absolute_percentage_error(test_data, prediction_ar2)

#finding the best order value of P
start_time = time.time()
ARfit = model_ar.fit(ic='t-stat')
ARfit.params

#order 19 is the best
prediction_ar19 = ARfit.predict(start,end)
prediction_ar19 = prediction_ar19.rename('AR(19) Predictions')
print("--- %s seconds ---" % (time.time() - start_time))

labels = ['AR1','AR2','AR19']

preds = [prediction_ar1,prediction_ar2,prediction_ar19]

for i in range(3):
    error = mean_absolute_error(test_data['AQI'],preds[i])
    print(f'{labels[i]} MAE was :{error}')
    
test_data.plot(figsize =(12,8),legend=True)
prediction_ar1.plot(legend=True)
prediction_ar2.plot(legend=True)
prediction_ar19.plot(legend=True)    

dataset.AQI.mean()

mse_ar19 = mean_squared_error(test_data,prediction_ar19)
mae_ar19 = mean_absolute_error(test_data,prediction_ar19)
rmse_ar19 = rmse(test_data['AQI'],prediction_ar19)
mape_ar19 = mean_absolute_percentage_error(test_data, prediction_ar19)

#order 19 performed the best

#forecasting the future
#retrainig the model on full dataser
model_AR19 = AR(dataset['AQI'])

AR19fit = model_AR19.fit(ic='t-stat')


forecasted_values_AR19 = AR19fit.predict(start=len(dataset),end=len(dataset)+7).rename('forecast')

#plotting
test_data['AQI'].plot(figsize=(12,8),legend=True)
forecasted_values_AR19.plot(legend=True)

#-----------------------------------
#using auto_arima as lag plots can be misguiding. auto_arima will give the best orders for ARIMA
'''pip install pmdarima'''#installing pmdarima for auto_arima
from pmdarima import auto_arima

stepwise_fit = auto_arima(dataset['AQI'],start_p=0,start_q=0,max_p=6,max_q=3,seasonal=False,trace=True)

stepwise_fit.summary() #orders(2,1,2) were the best observations

#ARMA
from statsmodels.tsa.arima_model import ARMA,ARIMA,ARMAResults,ARIMAResults
train_data = dataset.iloc[:943]
test_data = dataset.iloc[943:]

#checking for all values. got 212 to be the best
auto_arima(dataset['AQI'],seasonal=False).summary()

start = len(train_data)
end = len(train_data) + len(test_data) - 1
#applying ARMA first
start_time = time.time()
model_arma = ARMA(train_data['AQI'],order=(2,2))
result_arma = model_arma.fit()
result_arma.summary()

#making the length frame

predictions_arma = result_arma.predict(start,end).rename('ARMA (2,2) Predictions')
print("--- %s seconds ---" % (time.time() - start_time))

test_data['AQI'].plot(figsize=(12,8),legend=True)
predictions_arma.plot(legend=True)

mse_arma = mean_squared_error(test_data,predictions_arma)
mae_arma = mean_absolute_error(test_data,predictions_arma)
rmse_arma = rmse(test_data['AQI'],predictions_arma)
mape_arma = mean_absolute_percentage_error(test_data, predictions_arma)

#not good predicitons so we use ARIMA by adding I
dataset.mean()
predictions_arma.mean()

'''ARIMA'''

#ARIMA with order p=2, d=1 and q=2 should be used

#from statsmodels.tsa.statespace.tools import diff
#dataset['Diff_1'] = diff(dataset['AQI'],k_diff=1)

stepwise_fit = auto_arima(dataset['AQI'],start_p=0,start_q=0,max_p=3,max_q=3,seasonal=False,trace=True)
stepwise_fit.summary()

start = len(train_data)
end = len(train_data) + len(test_data) - 1

#ARIMA(2,1,2) has the lowest AIC so its the best
start_time = time.time()
model_arima = ARIMA(train_data['AQI'],order=(2,1,2))
#fitting the model
result_arima = model_arima.fit()

#maing the predicitons 
prediction_arima = result_arima.predict(start=start,end=end,typ='levels').rename('ARIMA(2,1,2) Prediction')
print("--- %s seconds ---" % (time.time() - start_time))

result_arima.summary()

#plotting
test_data['AQI'].plot(legend=True,figsize=(12,8))
prediction_arima.plot(legend=True)


#checking the errrors
mse_arima = mean_squared_error(test_data,prediction_arima)
mae_arima = mean_absolute_error(test_data,prediction_arima)
rmse_arima = rmse(test_data['AQI'],prediction_arima)
mape_arima = mean_absolute_percentage_error(test_data, prediction_arima)

#forecasting with arima
model_arima = ARIMA(dataset['AQI'],order=(2,1,2))
results_arima = model_arima.fit()
forecast_arima = results_arima.predict(start=len(dataset),end=len(dataset)+7,typ='levels').rename('ARIMA (2,1,2) Forecast')

dataset['AQI'].plot(legend=True,figsize=(12,8))
test_data['AQI'].plot(legend=True,figsize=(12,8))
forecast_arima.plot(legend=True)

'''SARIMA'''
#there maybe a bit of seasonality so SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#checking the best value for sarimax
auto_arima(train_data,seasonal=True,m=7).summary()

#giving the length of the start and end
start = len(train_data)
end = len(train_data) + len(test_data) - 1
#SARIMAX (1,1,1) is the best as per auto_arima
start_time = time.time()

model_sarima = SARIMAX(train_data['AQI'],order = (1,1,1))
result_sarima = model_sarima.fit()
result_sarima.summary()

prediction_sarima = result_sarima.predict(start,end,typ='levels').rename('SARIMA Prediction')
print("--- %s seconds ---" % (time.time() - start_time))

test_data['AQI'].plot(legend=True,figsize=(12,8))
prediction_sarima.plot(legend=True)

#evaluaiton
mse_sarima = mean_squared_error(test_data,prediction_sarima)
mae_sarima = mean_absolute_error(test_data,prediction_sarima)
rmse_sarima = rmse(test_data['AQI'],prediction_sarima)
mape_sarima = mean_absolute_percentage_error(test_data, prediction_sarima)

#forecasting
model_sarima_forecast = SARIMAX(dataset['AQI'],order = (1,1,2))
result_sarima_forecast = model_sarima_forecast.fit()

forecast_sarima = result_sarima_forecast.predict(len(dataset),len(dataset)+7,typ='levels').rename('SARIMA FORECAST')
test_data['AQI'].plot(legend=True,figsize=(12,8))
forecast_sarima.plot(legend=True)

#---------------------------------------------------------------------#
#prophet
from fbprophet import Prophet #need to import matplotlib.pyplot and again import prophet
import matplotlib.pyplot as plt

#prphet needs spedific names for date and data column 
dataset_prophet = dataset.reset_index()
dataset_prophet.columns = ['ds', 'y']
dataset_prophet.head()
#changing the type of date column to datetime
dataset_prophet['ds'] = pd.to_datetime(dataset_prophet['ds'])

#prediction
dataset_prophet.info()
#splitting
train_data_prophet = dataset_prophet.iloc[:966]
test_data_prophet = dataset_prophet.iloc[966:]

#fitting the model
start_time = time.time()

n = Prophet()
n.fit(train_data_prophet)
#making a future empty dataframe to store the predicitons in 
future_prophet = n.make_future_dataframe(periods=7,freq='D')
#predicitng the dates in the future dataframe we made
forecast_prophet = n.predict(future_prophet)
print("--- %s seconds ---" % (time.time() - start_time))

#to remove plotting error
pd.plotting.register_matplotlib_converters()
#plottinh
ax = forecast_prophet.plot(x='ds',y='yhat',label='Predictions',legend=True,figsize=(12,8))
test_data_prophet.plot(x='ds',y='y',label='True test data',legend=True,ax=ax,xlim=('2019-08-25','2019-08-31'))

#removing the predicted rows in a seperate variable for visualisaiton and evaluation 
prediction_prophet = forecast_prophet.iloc[-7:]
prediction_prophet = prediction_prophet.filter(['ds','yhat'],axis=1)
prediction_prophet=prediction_prophet.set_index('ds')


#general evaluaiton
mse_prophet = mean_squared_error(test_data_prophet['y'],prediction_prophet['yhat'])
mae_prophet = mean_absolute_error(test_data_prophet['y'],prediction_prophet['yhat'])
rmse_prophet = rmse(test_data_prophet['y'],prediction_prophet['yhat'])
mape_prophet = mean_absolute_percentage_error(test_data_prophet['y'],prediction_prophet['yhat'])

'''Novel Prophet did the best forecasting in comparison with all models'''

#forecasting the future by fitting the model on whole dataset
model_prophet = Prophet()
model_prophet.fit(dataset_prophet)

#placeholder to hold the future data
future = model_prophet.make_future_dataframe(periods=60,freq='D')
future

#prediction
forecast_prophet = model_prophet.predict(future)
forecast_prophet.head()

forecast_prophet.columns
#yhat represents the future AQI values
forecast_prophet[['ds','yhat_lower','yhat_upper','yhat']].tail(12)


model_prophet.plot(forecast_prophet);
model_prophet.plot_components(forecast_prophet);


#now lets check whether prophet is good in forecasting particular months or all time
#evaluating using propht inbuilt metric
'''this evaluation would tell us whether prophet is good in predicting particular months or time or it is good for predicting any month'''

from fbprophet.diagnostics import cross_validation,performance_metrics
from fbprophet.plot import plot_cross_validation_metric

#initial
initial = 2 * 365
initial = str(initial) + 'days'
initial

#period
period = 2 * 365
period = str(period) + 'days'


#horizon
horizon = 60
horizon = str(horizon) + 'days'

df_cv = cross_validation(n,initial=initial,period=period,horizon=horizon)
df_cv.head()


len(df_cv)
performance_metrics(df_cv)
plot_cross_validation_metric(df_cv,metric='rmse');

plot_cross_validation_metric(df_cv,metric='mape');
'''it can be seen from the plot that prophet has more or less constant mape all round so it is good in predicting any months'''
##############################################################################
'''lstm'''


len(dataset)
train_data = dataset.iloc[:966]
test_data = dataset.iloc[966:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)

#making the time sereis for keras
from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 30
n_features = 1

start_time = time.time()

train_generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=30)

X,y = train_generator[0]

model_lstm = Sequential()
model_lstm.add(LSTM(150,activation='relu',input_shape=(n_input,n_features)))
model_lstm.add(Dropout(0.15))

model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam',loss='mse')

model_lstm.summary()

######################################################
'''tried by adding more layers but no significang change seen 
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 70, activation='relu', return_sequences = True, input_shape = (n_input,n_features)))
model.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))



# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 70))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')'''

#####################################################


model_lstm.fit_generator(train_generator,epochs=30)

model_lstm.history.history.keys()

plt.plot(range(len(model_lstm.history.history['loss'])),model_lstm.history.history['loss'])

first_eval_batch = scaled_train[-30:]
first_eval_batch = first_eval_batch.reshape((1,n_input,n_features))

model_lstm.predict(first_eval_batch)

#forecasting(predicting)
test_prediction = []
first_eval_batch = scaled_train[-n_input:]
#reshaping back
current_batch = first_eval_batch.reshape((1,n_input,n_features))

#how far can we forecast
for i in range(len(test_data)):
    current_pred = model_lstm.predict(current_batch)[0]
    test_prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

print("--- %s seconds ---" % (time.time() - start_time))

test_prediction
true_prediction = scaler.inverse_transform(test_prediction)

true_prediction = pd.DataFrame(true_prediction)
true_prediction.index = test_data.index

#plotting
test_data['AQI'].plot(legend=True,figsize=(12,8))
true_prediction[0].plot(legend=True,label='LSTM')

#model.save('lstm_model.h5')
#pwd

#evaluaiton
#
mse_lstm = mean_squared_error(test_data,true_prediction)
mae_lstm = mean_absolute_error(test_data,true_prediction)
rmse_lstm = rmse(test_data,true_prediction)
mape_lstm = mean_absolute_percentage_error(test_data, true_prediction)



#forecasting the future
train_data_new = dataset
scaler.fit(train_data_new)
train_data_new = scaler.transform(train_data_new)


n_input = 30
n_features = 1
generator = TimeseriesGenerator(train_data_new, train_data_new, length=n_input, batch_size=6)

model_lstm.fit_generator(generator,epochs=30)


pred_list = []

batch = train_data_new[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model_lstm.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


from pandas.tseries.offsets import DateOffset
add_dates = [dataset.index[-1] + DateOffset(days=x) for x in range(0,32) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=dataset.columns)



df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pd.concat([dataset,df_predict], axis=1)



plt.figure(figsize=(20, 5))
plt.plot(df_proj.index, df_proj['AQI'])
plt.plot(df_proj.index, df_proj['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()
#--------------------------------------#
#forward rolling forecast
history = [x for x in train]
prediction_fwdrolling = list()
for i in range(len(test)):
    #making prediction
    yhat = history[-7]
    prediction_fwdrolling.append(yhat)
    #observatioln
    history.append(test[i])
#plot prediction vs observation
plt.plot(test)
plt.plot(prediction_fwdrolling)
plt.show()    


#rolling window forecast
dataset_rolling = pd.read_csv("ready.csv",index_col='date',header=0)   #reading the csv
columnstodrop = ['SO2','CO','Ozone','PM2.5','NO2']
dataset_rolling = dataset_rolling.drop(columnstodrop, axis=1)

X = dataset_rolling.values
train, test = X[0:-7], X[-7:]

window_size = range(1,60)
scores = list()
for w in window_size:
    #walk-forward validation
    history = [x for x in train]
    prediction = list()
    for i in range(len(test)):
        #make prediction
        yhat = mean(history[-w:])
        prediction.append(yhat)
        #observation
        history.append(test[i])
    #performance
    rmse = sqrt(mean_squared_error(test,prediction))
    scores.append(rmse)
    print('w=%d RMSE:%.3f' % (w, rmse))
# plot scores
plt.plot(window_size, scores)
plt.show()    

#window size 41 shows least rmse so predicting with it
history = [x for x in train]
prediction = list()
for i in range(len(test)):
     #make prediction
     yhat = mean(history[-41:])
     prediction.append(yhat)
     #observation
     history.append(test[i])
    
# plot scores
plt.plot(test)
plt.plot(prediction)
plt.show()  




#--------------------------------------------------------#
#######################TBATS####################################
train_data = dataset.iloc[:943]
test_data = dataset.iloc[943:]

# Fit the model
start_time = time.time()

estimator_tbat = TBATS(seasonal_periods=(1, 365.25))
model_tbat = estimator_tbat.fit(train_data)
# Forecast 7 days ahead
tbat_forecast = model_tbat.forecast(steps=30)
print("--- %s seconds ---" % (time.time() - start_time))

tbat_forecast = pd.DataFrame(tbat_forecast) 
tbat_forecast.index = test_data.index

test_data['AQI'].plot(legend=True,label='Test', figsize=(12,8))
tbat_forecast[0].plot(legend=True,label='TBAT')    

#evaluaiton
mse_tbat = mean_squared_error(test_data,tbat_forecast)
mae_tbat = mean_absolute_error(test_data,tbat_forecast)
rmse_tbat = rmse(test_data,tbat_forecast)
mape_tbat = mean_absolute_percentage_error(test_data, tbat_forecast)

model_tbat.summary()

#forecasting
training again on the full dataset
model_tbat = estimator_tbat.fit(dataset)

# Forecast 7 days ahead
tbat_forecast = model_tbat.forecast(steps=30)

tbat_forecast = pd.DataFrame(tbat_forecast) 
tbat_forecast.index = exp_forecast_prediction.index

test_data['AQI'].plot(legend=True,label='Test', figsize=(12,8))
tbat_forecast[0].plot(legend=True,label='TBAT')    

#---------------------------------------------------------------------#
#--------------------------------------------------------------#
#plotting all

test_data['AQI'].plot(legend=True,figsize=(12,8))
test_predictions_exp.plot(legend=True,label='Prediction exponential smoothing')
prediction_ar1.plot(legend=True,label='AR1')
prediction_ar2.plot(legend=True,label='AR2')
prediction_ar19.plot(legend=True,label='AR19')
predictions_arma.plot(legend=True,label='ARMA')
prediction_arima.plot(legend=True,label='ARIMA')
prediction_sarima.plot(legend=True,label='SARIMA')
true_prediction[0].plot(legend=True,label='LSTM')
prediction_prophet['yhat'].plot(legend=True,label='Prophet')
tbat_forecast[0].plot(legend=True,label='TBAT')    




###################################################################
##########################################################


from statsmodels.tsa.arima_model import ARIMA

X = dataset.values
size = int(len(X) * 0.90)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,1,2))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
#evaluaiton
mse_rolling = mean_squared_error(test,predictions)
mae_rolling = mean_absolute_error(test,predictions)
rmse_rolling = rmse(test,predictions)
mape_rolling = mean_absolute_percentage_error(test, predictions)

print('Test MAE: %.3f' % mae_rolling)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


test= pd.DataFrame(test) 
predictions= pd.DataFrame(predictions) 


test.plot(legend=True,label='Test', figsize=(12,8))
predictions.plot(legend=True,label='ARIMA rolling Prediction')
plt.show()   

########################################################################







