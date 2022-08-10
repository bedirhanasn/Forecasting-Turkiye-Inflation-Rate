#!/usr/bin/env python
# coding: utf-8

# In[1]:


# simple model approach for inflation forecasting in Turkey


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

# ignore harmless warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# import data from xlsx file
dataset = pd.read_excel('pivot.xls')

#print head of the dataset
dataset.head()


# In[4]:


# print tail of the dataset
dataset.tail()


# In[5]:


# revise the first row of the dataset to be the months of the year
new_header = dataset.iloc[0] 
dataset = dataset[1:] 
dataset.columns = new_header


# In[6]:


# print head of the dataset
dataset.head()


# In[7]:


# unpivot from wide to long format
dataset = dataset.melt(id_vars=['Year'], var_name='Month', value_name='Rate')
dataset.head()


# In[8]:


# add new date column and assign last day of month
dataset['Date'] = pd.to_datetime(dataset[['Year', 'Month']].assign(DAY=1)) + MonthEnd(1)

# order ascending data values 
dataset = dataset.sort_values(by=['Date'])
dataset.head()


# In[9]:


dataset.tail()


# In[10]:


new_dataset = dataset[:-5]


# In[11]:


new_dataset.tail()


# In[12]:


# select needed columns
df = new_dataset[['Date','Rate']]

# set date column as index
df.set_index('Date', inplace=True)
df.tail()


# In[13]:


# detail of final DataFrame
df.describe().transpose()


# In[14]:


# time series of inflation percentage
dataset.plot(x='Date', y='Rate', figsize=(10,6))


# In[15]:


# more time series of inflation percentage
time_series = df['Rate']
time_series.rolling(window=12).mean().plot(label='Rolling Mean')
time_series.rolling(window=12).std().plot(label='Rolling Std')
time_series.plot(figsize=(10,6))
plt.legend()
plt.show()


# observed: the actual value in the series;
# trend: the increasing or decreasing value in the series;
# seasonality: the repeating short-term cycle or pattern in the series;
# residual/noise: the random variation in the series.

# In[16]:


# plot decomposition components
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(time_series)
fig = decomp.plot()


# In[17]:


# check if inflation series is stationary
from statsmodels.tsa.stattools import adfuller

# ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('1. ADF: ', result[0])
    print('2. P-value: ', result[1])
    print('3. Num of Lags: ', result[2])
    print('4. Num of Observations: ', result[3])
    print('5. Critial Values:')
    for key, value in result[4].items():
        print('\t', key, ': ', value)
        
    if result[1] <= 0.05:
        print('\nStrong evidence against the null hypothesis (H0), reject the null hypothesis. Data has no unit root and is stationary.')
    else:
        print('\nWeak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary.')

# run function
adf_test(df['Rate'])


# In[18]:


# finding differencing value
from pmdarima.arima.utils import ndiffs
print(ndiffs(df['Rate'], test='adf'))
print(ndiffs(df['Rate'], test='kpss'))
print(ndiffs(df['Rate'], test='pp'))


# In[19]:


# plotting ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig = plt.figure(figsize=(8,7))
ax1 = fig.add_subplot(2,1,1)
fig = plot_pacf(df, ax=ax1)
ax2 = fig.add_subplot(2,1,2)
fig = plot_acf(df, ax=ax2)
plt.show()


# In[20]:


# auto ARIMA function
from pmdarima import auto_arima
stepwise_fit = auto_arima(df['Rate'], trace=True, suppress_warnings=True)
stepwise_fit.summary()


# In[21]:


# p=2, d=0, q=1
from statsmodels.tsa.arima.model import ARIMA

# fitting the model
model = ARIMA(df['Rate'], order=(2,0,1), freq='M')
model_fit = model.fit()
model_fit.summary()


# In[22]:


# predict values
pred = model_fit.predict(start=0, end=len(df) - 1, typ='levels', dynamic=False)


# In[23]:


# display last rows
pred.tail()


# In[24]:


# plot results
df['Rate'].plot(legend=True, label='Actual', figsize=(14,5))
pred.plot(legend=True, label='Forecast')
plt.title('Forecast vs Actual Results')
plt.show()


# In[25]:


# diagnostic plots for standardized residuals of one endogenous variable
model_fit.plot_diagnostics(figsize=(9,9))
plt.show()


# In[26]:


# root mean squared error
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(pred, df['Rate'], squared=False)
rmse


# In[27]:


# mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(pred, df['Rate'])
mae


# In[28]:


# mean absolute percentage error
mape = np.mean(np.abs(df['Rate'] - pred) / df['Rate']) * 100
mape


# In[29]:


# correlation
corr = np.corrcoef(pred, df['Rate'])[0,1]
corr


# In[30]:


# predict values
forecast = model_fit.predict(start=0, end=len(df) + 4, typ='levels', dynamic=False)


# In[31]:


# display forecasted values
forecast.tail(5)


# In[32]:


# final plot
forecast.iloc[50:].plot(legend=True, label='Forecast', figsize=(14,5), color='red')
df['Rate'].iloc[50:].plot(legend=True, label='Actual')
plt.title('Forecast Inflation')
plt.show()

