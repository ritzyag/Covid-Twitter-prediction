import pandas as pd
import numpy as np
import json


def crosscorr(datax, datay, window_size,method, lag=0, start = 0):
        x = datax.iloc[start : start + window_size]
        y = (datay.shift(lag).iloc[start : start + window_size])
        return x.corr(y, method = method)

final = pd.read_csv('/path/to/Indian_series_rolling_data')
final["%primary"] = (final["primary"]/final["total"])*100
final["%secondary"] = (final["secondary"]/final["total"])*100
final["%thirdparty"] = (final["thirdparty"]/final["total"])*100
final["reporting"] = final["primary"] + final["secondary"] + final["thirdparty"]
final["%reporting"] = (final["reporting"]/final["total"])*100

final = final[final["Date"] >= '2021-01-01']

window = final.shape[0]

correlations = {}
for method in ["pearson", "spearman"]:
    for type1 in ["cases", "deaths"]:
        for type2 in ["primary", "secondary", "thirdparty","reporting", "%primary","%secondary","%thirdparty", "%reporting", "total"]:
            d1 = final[type1]
            d2 = final[type2]
            rs = [crosscorr(d1,d2, window,method, lag = lag, start = 0 ) for lag in [0,7,14,21,28,35]]
            # print(rs)
            correlations[method+ "_" + type1 + "_" + type2] = rs

a = pd.DataFrame(correlations).T
a.columns = ["Week Lag 0", "Week Lag 1","Week Lag 2","Week Lag 3","Week Lag 4","Week Lag 5"]
a.to_csv('/path/to/output/file/directory')          
