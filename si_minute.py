"""
泰山崩于前而色不变
麋鹿兴于左而目不瞬
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import akshare as ak
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import matplotlib.gridspec as gs
import argparse
import re

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

spread = False
rq_ready = True
try:
    import rqdatac as rq
    rq.init()
except:
    rq_ready = False
    print("WARNING: Ricequant api not available!")

from IPython.utils import io
from index import calculate_index_weight

date_list = pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"])

index_info = pd.read_json("index_info.json")

with io.capture_output() as _:
    gfex_text = ak.match_main_contract(symbol="gfex")
si_dom = gfex_text[:6]

def get_minute(symbol: str, date: dt.datetime):

    underlying = ''.join(re.findall(r'[A-Z]', symbol))

    if date != pd.to_datetime(dt.date.today()):
        data = rq.futures.get_dominant_price(underlying, frequency="1m").loc[underlying]
        data = data[data["trading_date"] == date]
    else:
        data = ak.futures_zh_minute_sina(symbol)
        data = data.set_index("datetime", drop=True)
        data.index = pd.to_datetime(data.index)
        data = data[data.index.map(lambda x: x.date()) == dt.date.today()]

    open_time = pd.to_datetime(date.strftime("%Y-%m-%d ") + "09:00:00")
    minute_price = data["close"]
    minute_price = minute_price[minute_price.index >= open_time]
    minute_price.loc[open_time] = data["open"].iloc[0]
    minute_price = minute_price.sort_index()
    minute_price.name = underlying
    
    return minute_price


def calculate_index(date: dt.datetime):

    all_data = pd.DataFrame()
    for symbol in index_info["symbol"]:
        data = get_minute(symbol, date)
        data = (data.diff() / data.shift()).fillna(0)
        all_data = pd.concat([all_data, data], axis=1).fillna(0)
    
    open_time = pd.to_datetime(date.strftime("%Y-%m-%d ") + "09:00:00")
    close_time = pd.to_datetime(date.strftime("%Y-%m-%d ") + "15:00:00")
    all_data = all_data[index_info.index]
    index = (all_data.apply(lambda x: np.dot(index_info["weight"], x), axis=1) + 1).cumprod()
    index = index[(index.index >= open_time) & (index.index <= close_time)].asfreq("1min")

    return index


def get_minute_si(date: dt.datetime):

    if date != pd.to_datetime(dt.date.today()):
        data = rq.futures.get_dominant_price("SI", frequency="1m").loc["SI"]
        data = data[data["trading_date"] == date]
    else:
        data = ak.futures_zh_minute_sina(si_dom)
        data = data.set_index("datetime", drop=True)
        data.index = pd.to_datetime(data.index)
        data = data[data.index.map(lambda x: x.date()) == dt.date.today()]

    minute_price = data["close"]
    vwap = (data["close"] * data["volume"]).cumsum() / data["volume"].cumsum()
    volume = data["volume"]

    open_time = pd.to_datetime(date.strftime("%Y-%m-%d ") + "09:00:00")
    minute_price.loc[open_time] = data["open"].iloc[0]
    minute_price = minute_price.sort_index()
    vwap.loc[open_time] = data["open"].iloc[0]
    vwap = vwap.sort_index()
    
    return data["open"].iloc[0], minute_price, vwap, volume


def get_last_price(date: dt.datetime):

    date_idx = date_list[date_list == date].index[0]
    last_date = date_list[date_idx-1]
    data = ak.futures_zh_daily_sina("SI2511").set_index("date")
    data.index = pd.to_datetime(data.index)
    data = data.loc[last_date, :]
    return data["close"], data["settle"]


def recommend_spread(date: dt.datetime):
    
    date_idx = date_list[date_list == date].index[0]
    last_dates = date_list[date_idx-4: date_idx]
    cum_spread = 0
    for i, date in enumerate(last_dates):
        _, minute_price, _, volume = get_minute_si(date)
        index = (calculate_index(date) * minute_price.iloc[0]).reindex(minute_price.index)
        spread = (minute_price - index).fillna(0) * volume
        spread = spread.groupby(spread.index.date).apply(lambda x: x.cumsum()) / volume.groupby(volume.index.date).apply(lambda x: x.cumsum())
        cum_spread += spread.iloc[-1] * np.exp(0.5*(i-4))
    return -cum_spread - 5


def calculate_beta(si_val, index_val):
    
    all_X = index_val.diff(3) / index_val.shift(3)
    all_y = si_val.diff(3) / si_val.shift(3)

    def regression(si_series: pd.Series):
        y = si_series
        X = all_X.reindex(si_series.index)
        X = sm.add_constant(X)
        return sm.OLS(y, X).fit().params.iloc[1]
    
    beta = all_y.reindex(index_val.index).rolling(5).apply(regression).reindex(si_val.index)

    pos_X = all_X[all_X >= 0]
    neg_X = all_X[all_X < 0]
    pos_y = all_y[pos_X.index]
    neg_y = all_y[neg_X.index]
    pos_beta = sm.OLS(pos_y, sm.add_constant(pos_X)).fit().params.iloc[1]
    neg_beta = sm.OLS(neg_y, sm.add_constant(neg_X)).fit().params.iloc[1]

    return beta - 1, pos_beta / neg_beta


def main(price: float=None, date: dt.datetime=None):

    today_open, minute_price, vwap, volume = get_minute_si(date)
    index = calculate_index(date).reindex(minute_price.index)
    try:
        beta, pn_ratio = calculate_beta(index, minute_price / minute_price.iloc[0])
    except:
        beta, pn_ratio = None, None
    index = index * minute_price.iloc[0]
    numeric_index = np.arange(len(minute_price))
    xticks = numeric_index[::15]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})

    ax1, ax2 = axes
    ax1.plot(numeric_index, minute_price, label="SI2511")
    ax1.plot(numeric_index, vwap, label="VWAP")
    ax1.plot(numeric_index, index, label="Index")
    if price:
        ax1.axhline(price, linestyle="--", color="red")
    ax1.axvline(75, color="black", linewidth=1)
    ax1.axvline(135, color="black", linewidth=1)
    ax1_ = ax1.twinx()
    ax1_.bar(numeric_index[:-1] + 0.5, volume, color="grey", width=0.5)
    ax1.legend()
    ax1.set_xlim(0, 225)
    ax1.set_xticklabels([])
    ax1.grid(True)

    ax2.plot(numeric_index, minute_price - index)
    if spread:
        ax2.axhline(recommend_spread(date), linestyle="--", color="orange")
    if beta is not None:
        ax2_ = ax2.twinx()
        ax2_.bar(numeric_index, beta, color="grey", width=0.5)
    ax2_.axhline(0, color="red")
    ax2.set_xlim(0, 225)
    ax2.set_xticks(xticks, [minute_price.index[i].strftime('%H:%M') for i in xticks])
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    if pn_ratio is not None:
        fig.text(0.92, 0.25, 
            "%.2f%%" % (pn_ratio * 100), fontsize=12, 
            bbox=dict(boxstyle='round', facecolor=("green" if pn_ratio < 1 else "red"), alpha=0.5))
    
    if (last_data := get_last_price(date)):
        last_close, last_settle = last_data
        fig.text(0.92, 0.9, 
                "昨收 %d\n昨结 %d\n今开 %d" % (last_close, last_settle, today_open), 
                va='center', ha='left', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        chg_close = (today_open / last_close - 1) * 100
        fig.text(0.92, 0.80, "%.2f%%" % chg_close, fontsize=12,
                bbox=dict(boxstyle="round", facecolor=("green" if chg_close < 0 else "red"), alpha=0.5))
        chg_settle = (today_open / last_settle - 1) * 100
        fig.text(0.92, 0.75, "%.2f%%" % chg_settle, fontsize=12,
                bbox=dict(boxstyle="round", facecolor=("green" if chg_settle < 0 else "red"), alpha=0.5))

    ax1.set_title(date.strftime("%Y-%m-%d"))
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Minute-by-minute market monitoring for the dominant industrial silicon future contract.",
        epilog="Example: python si_monitor.py --price [8725] --date [2025-09-15] --index --spread"
    )

    parser.add_argument("--price", type=float, help="set the open price of current position")
    parser.add_argument("--date", type=str, help="set the date to observe")
    parser.add_argument("--index", action="store_true", help="launch a new calculation of index weight")
    parser.add_argument("--spread", action="store_true", help="show recommended spread")

    args = parser.parse_args()

    if args.index:
        calculate_index_weight()
    if args.spread:
        spread = True

    date = pd.to_datetime(dt.date.today())
    if rq_ready and args.date:
        date = pd.to_datetime(args.date)
    
    main(args.price, date)

