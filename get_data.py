import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import talib as ta
import numpy as np
from datetime import datetime,timedelta,time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

T_START = datetime(2024,3,1)
T_END = datetime(2024,4,30)
B_START = T_START - timedelta(days=5)
B_END= T_END + timedelta(days=2)


def get_new_date(x):
    x = datetime.fromtimestamp(x.timestamp()) - timedelta(hours = 8)
    x = x.replace(second=0)
    x = x.replace(microsecond=0)
    return x

def get_tick_data(tick_path,start = T_START, end = T_END):
    tick_data = pd.read_csv(tick_path)
    tick_data['date'] = pd.to_datetime(tick_data['date'])
    tick_data['datetime'] = tick_data['date'].apply(get_new_date)
    tick_data['max'] = np.nan
    tick_data['min'] = np.nan 
    tick_data['slope'] = np.nan
    tick_data['grad'] = np.nan
    tick_data['atr'] = np.nan
    if start and end:
        tick_data = tick_data[(tick_data['datetime']>=start)&(tick_data['datetime']<=end)]
    return tick_data

def get_bar_data(bar_path,start=B_START,end=B_END):
    origion = pd.read_csv(bar_path)
    origion['datetime'] = pd.to_datetime(origion['datetime'])
    # origion.set_index('datetime',inplace=True)
    origion = origion.iloc[:,[1,3,4,8,9]]
    for i,v in origion.iterrows():
        t0 = datetime.fromtimestamp(origion.iloc[i,0].timestamp()) - timedelta(hours = 8)
        if t0.time() == time(23,0) or t0.time() == time(15,0):
            t1 = datetime.fromtimestamp(origion.iloc[i+1,0].timestamp()) - timedelta(hours = 8) - timedelta(minutes =1)
            origion.iloc[i,0] = t1
    data = origion
    if start and end:
        data = data[(data['datetime']>=start)&(data['datetime']<=end)]
    data.set_index('datetime',inplace=True)
    data = get_param(data)
    return data

#计算指标在这里
def get_param(data):
    data['sma'] = ta.SMA(data['close'],5)
    data['slope'] = ta.LINEARREG_SLOPE(data['sma'],3)
    data['max'] = data['high'].rolling(60).max()
    data['min'] = data['low'].rolling(60).min()
    data['g'] = np.gradient(data['close'])
    data['grad'] = data['g'].shift()
    data['atr'] = ta.ATR(data['high'],data['low'],data['close'],14)
    return data

def transform_data(bar_path,tick_path,new_path):
    bar_data = get_bar_data(bar_path)
    tick_data = get_tick_data(tick_path)
    columns = list(bar_data.columns)
    slope_id = columns.index('slope')
    max_id = columns.index('max')
    min_id = columns.index('min')
    grad_id = columns.index('grad')
    atr_id = columns.index('atr')

    for i,v in tqdm(bar_data.iterrows()):
        tick_ids = tick_data[tick_data['datetime'] == i].index
        tick_data.loc[tick_ids,'slope'] = v[slope_id]
        tick_data.loc[tick_ids,'max'] = v[max_id]
        tick_data.loc[tick_ids,'min'] = v[min_id]
        tick_data.loc[tick_ids,'grad'] = v[grad_id]
        tick_data.loc[tick_ids,'atr'] = v[atr_id]
    tick_data.to_csv(new_path,index=None)
    print(tick_data.head())

def view_data(path):
    pd_data = pd.read_csv(path)
    print(pd_data.tail(5))

if __name__ == "__main__":
    bar_path = 'D:/Code/jupyter_project/data_analysis/bar_analysis/bar_backtest/eg数据分析/eg2405.csv'
    tick_path = 'D:/Code/jupyter_project/data_analysis/bar_analysis/bar_backtest/eg数据分析/eg2405_tick.csv'
    new_path = os.path.dirname(__file__)+'/rb2405_new_tick.csv'
    # transform_data(bar_path, tick_path, new_path)
    view_data(new_path)
