import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import akshare as ak
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import argparse
import time
import re

from index_weight import calculate_index_weight

start_date = pd.to_datetime("20230101")
open_time = pd.to_datetime(dt.date.today().strftime("%Y-%m-%d ") + "09:00:00")
close_time = pd.to_datetime(dt.date.today().strftime("%Y-%m-%d ") + "15:00:00")

def get_minute(symbol: str):

    data = ak.futures_zh_minute_sina(symbol)
    data = data.set_index("datetime", drop=True)
    data.index = pd.to_datetime(data.index)
    data = data[data.index.map(lambda x: x.date()) == dt.date.today()]
    minute_price = data["close"]
    minute_price = minute_price[minute_price.index >= open_time]
    minute_price.loc[open_time] = data["open"].iloc[0]
    minute_price = minute_price.sort_index()
    minute_price.name = ''.join(re.findall(r'[A-Z]', symbol))
    
    return minute_price

def calculate_index():

    all_data = pd.DataFrame()
    for symbol in symbol_list:
        if ''.join(re.findall(r'[A-Z]', symbol)) in other_corr.index:
            data = get_minute(symbol)
            data = (data.diff() / data.shift()).fillna(0)
            all_data = pd.concat([all_data, data], axis=1).fillna(0)
    
    all_data = all_data[si_corr.index]
    index = (all_data.apply(lambda x: np.dot(result.x, x), axis=1) + 1).cumprod()
    index = index[(index.index >= open_time) & (index.index <= close_time)].asfreq("1min")

    return index

si_dom = None
for symbol in symbol_list:
    if symbol[:2] == "SI":
        si_dom = symbol
        break

def get_minute_si():

    if si_dom == None:
        raise ValueError("No dominant contract loaded!")

    data = ak.futures_zh_minute_sina(si_dom)
    data = data.set_index("datetime", drop=True)
    data.index = pd.to_datetime(data.index)
    data = data[data.index.map(lambda x: x.date()) == dt.date.today()]

    minute_price = data["close"].asfreq("1min")
    vwap = (data["close"] * data["volume"]).cumsum() / data["volume"].cumsum().asfreq("1min")
    volume = data["volume"].asfreq("1min")

    minute_price.loc[open_time] = data["open"].iloc[0]
    minute_price = minute_price.sort_index()
    vwap.loc[open_time] = data["open"].iloc[0]
    vwap = vwap.sort_index()
    
    return minute_price, vwap, volume

minute_price, vwap, volume = get_minute_si()
index = calculate_index() * minute_price.iloc[0]

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(minute_price, label="SI2511")
ax1.plot(vwap, label="VWAP")
ax1.plot(index, label="Index")
ax2 = ax1.twinx()
ax2.bar(volume.index - pd.Timedelta(seconds=30), volume, color="grey", width=1e-4)
ax1.legend()
plt.gca().xaxis.set_major_formatter(mdt.DateFormatter('%H:%M'))
plt.grid(True)
plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Minute-by-minute market monitoring for the dominant industrial silicon future contract.",
        epilog="Example: python si_monitor.py --new_index"
    )

    parser.add_argument("--new_index", action="store_true", help="Launch a new calculation of index weight.")

    args = parser.parse_args()

    if args.new_index:
        calculate_index_weight()