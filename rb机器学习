import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import talib as ta
import numpy as np
import glob
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


origion = pd.read_csv(f'data/{future}.csv')
origion['datetime'] = pd.to_datetime(origion['datetime'])
origion.set_index('datetime',inplace=True)
origion = origion.dropna(axis=0)
origon = origion[25000:80000]
origion.head()

data = origion.resample('1min').agg(
    {
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
    }
).dropna()


p = 60

data['max_diff'] = data['high'].rolling(p).max().shift(-p) - data['close']
data['min_diff'] = data['low'].rolling(p).min().shift(-p) - data['close']
sma = ta.SMA(data['close'],5)
data['slope'] = ta.LINEARREG_SLOPE(sma,3)
data['g'] = np.gradient(data['close'])
data['grad'] = data['g'].shift()
data['atr'] = ta.ATR(data['high'],data['low'],data['close'],30)
data['volume_per'] = (data['volume'] - data['volume'].shift() +1)/(data['volume'].shift()+1)
data['hc'] = data['high'] - data['close']
data['lc'] = data['low'] - data['close']
data['diff'] = abs(data['max_diff']) - abs(data['min_diff'])

data['label'] = 0


h1 = data[(data['diff']>=12)].index
l1 = data[(data['diff']<=-12)].index
data.loc[h1,'label'] = 1
data.loc[l1,'label'] = 2
data.head()

data['label'].value_counts()

new_data = data.iloc[:,[7,9,10,11,12,13,15]]
print(new_data.head())
feature = new_data.iloc[:,:-1]
label = new_data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(feature,label,test_size=0.3)
