import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

import rqdatac as rq
rq.init()

index_weight = pd.read_json("index_weight.json")[0]

si_data = rq.futures.get_dominant_price("SI", frequency="1m").loc["SI"]

def preprocess(data: tuple):

    minute_price = data["close"]
    minute_price.loc[data["trading_date"].iloc[0] + pd.Timedelta(hours=9)] = data["open"].iloc[0]
    minute_price = minute_price.sort_index()
    trading_date = minute_price.index.strftime("%Y-%m-%d")
    mask = (minute_price.index >= pd.to_datetime(trading_date + " 09:00:00")) & \
           (minute_price.index <= pd.to_datetime(trading_date + " 15:00:00"))
    minute_price = minute_price[mask]

    return minute_price

si_minute = si_data.groupby("trading_date").apply(preprocess)
si_minute.name = "SI"
si_val = si_minute.groupby(level=0).apply(lambda x: (((x.diff() / x.shift()).fillna(0) + 1).cumprod())).droplevel(0)

all_minute = pd.DataFrame()
for underlying in index_weight.index:
    data = rq.futures.get_dominant_price(underlying, frequency="1m").loc[underlying]
    minute = data.groupby("trading_date").apply(preprocess)
    minute.name = underlying
    all_minute = pd.concat([all_minute, minute], axis=1)

index_val = all_minute.groupby(level=0).apply(lambda x: (((x.diff() / x.shift()).apply(lambda y: y.dot(index_weight), axis=1)).fillna(0) + 1).cumprod()).droplevel(0)

val_diff = index_val - si_val
counts, bin_edges = np.histogram(val_diff, bins=100)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

def normal_func(x, mu, sigma, amplitude):
    return amplitude * norm.pdf(x, mu, sigma)

params, params_covariance = optimize.curve_fit(
    normal_func, bin_centers, counts, p0=[val_diff.mean(), val_diff.std(), counts.max()]
)

print(f"拟合参数: μ={params[0]:.6f}, σ={params[1]:.6f}, 幅度={params[2]:.6f}")

confidence_levels = [0.10, 0.20, 0.30]

print("标准正态分布的单边分位数:")
for level in confidence_levels:
    z_score = norm.ppf(1 - level)
    print(f"{int(level*100)}%水平: μ + {z_score:.4f}σ")