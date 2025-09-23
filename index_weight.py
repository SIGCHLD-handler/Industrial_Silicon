import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import akshare as ak
import datetime as dt
from scipy.optimize import minimize
from IPython.utils import io
import re

start_date = pd.to_datetime("20230101")
open_time = pd.to_datetime(dt.date.today().strftime("%Y-%m-%d ") + "09:00:00")
close_time = pd.to_datetime(dt.date.today().strftime("%Y-%m-%d ") + "15:00:00")

def calculate_index_weight():

    with io.capture_output() as _:
        dce_text = ak.match_main_contract(symbol="dce")
        czce_text = ak.match_main_contract(symbol="czce")
        shfe_text = ak.match_main_contract(symbol="shfe")
        gfex_text = ak.match_main_contract(symbol="gfex")

    symbol_list = ",".join([dce_text, czce_text, shfe_text, gfex_text]).split(',')
    underlying_list = [''.join(re.findall(r'[A-Z]', symbol)) for symbol in symbol_list]

    all_data = pd.DataFrame()
    for underlying in underlying_list:
        data = ak.futures_zh_daily_sina(underlying+'0')
        if len(data.index) < 126:
            continue
        data = data.set_index("date", drop=True)
        data.index = pd.to_datetime(data.index)
        intra_ret = (data["close"] - data["open"]) / data["open"]
        intra_ret = intra_ret[intra_ret.index[intra_ret.index >= start_date]]
        intra_ret.name = underlying
        all_data = pd.concat([all_data, intra_ret], axis=1)

    corr = all_data.corr()
    si_corr = corr["SI"]
    si_corr = si_corr[si_corr.index != "SI"].fillna(0).sort_values(ascending=False).iloc[:20]
    other_corr = corr[si_corr.index]
    other_corr = other_corr.loc[si_corr.index, :].fillna(0)

    threshold = 0.7
    upper_triangle = other_corr.corr().where(np.triu(np.ones(other_corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    si_corr = si_corr.drop(to_drop)
    other_corr = other_corr.drop(to_drop)
    other_corr = other_corr[other_corr.index]

    def objective(w: pd.Series):
        numerator = np.dot(w, si_corr)
        denominator = np.sqrt(np.dot(w, np.dot(other_corr, w)))
        return -numerator / denominator

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    w0 = np.ones(len(si_corr)) / len(si_corr)

    result = minimize(objective, w0, method='SLSQP', constraints=constraints)

    pd.DataFrame(result.x, index=si_corr.index).to_json("index_weight.json")