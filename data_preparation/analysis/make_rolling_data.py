import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gzip
import jsonlines

def date(row):
    row1 = ["20" + str(row.split("/")[2])]
    row2 = row.split("/")[0:2]
    row3 = row2+ row1
    return "/".join(row3)

#===============indian data===============
indian = pd.read_csv('/path/to/case_time_series.csv')
indian = indian[["Date_YMD", "Total Confirmed", "Total Deceased", "Daily Confirmed", "Daily Deceased"]]
indian['Date'] =  pd.to_datetime(indian['Date_YMD'], format='%Y-%m-%d')
indian = indian[["Date", "Daily Confirmed", "Daily Deceased"]]
indian.columns = ["Date", "cases1", "deaths1"]
indian["cases_rolling"] = indian["cases1"].rolling(7, min_periods = 1).sum()
indian["deaths_rolling"] = indian["deaths1"].rolling(7, min_periods = 1).sum()
indian[["Date", "cases_rolling", "deaths_rolling", "cases1", "deaths1"]].to_csv('/path/to/output/folder/Indian_series_rolling_daily.csv', index = 0)

#===================inference files===========
location_jsonl = '/path/to/tweets/json/file'
location_preds = '/path/to/predictions/csv/file'
location_or_objects = '/path/to/tweets_location/json/file'

dates = []
ids = []
username = []
country = []
with gzip.open(location_jsonl) as fp:
    rdr = jsonlines.Reader(fp).iter()
    for row in rdr:
            for r in row:
                dates.append(r["datestamp"])
                ids.append(r["id"])
                username.append(r["user_id"])

tweets_preds = pd.read_csv(location_preds)
tweets_preds["datestamp"] = dates
tweets_preds["id"] = ids
tweets_preds["user_id"] = username

ids = []
country = []
with gzip.open(location_or_objects) as fp:
    rdr = jsonlines.Reader(fp).iter()
    for row in rdr:
        if row["place"] is not None:
            ids.append(row["id"])
            country.append(row["place"]["country_code"])

id_country = pd.DataFrame({"id" : ids, "country" : country})
tweets_preds = tweets_preds.merge(id_country, on = "id")

tweets_preds2 = tweets_preds.groupby(["datestamp", "user_id", "pred_multi"]).agg({"id" : "max"}).reset_index()
tweets_preds3 = tweets_preds2[["id"]].merge(tweets_preds, on = "id", how = "inner")
tweets_preds4 = tweets_preds3.drop_duplicates(subset = "tweet", keep = "last")

tweets_preds4['month'] = (np.where(tweets_preds4['datestamp'].str.contains('-'),tweets_preds4['datestamp'].str.split('-').str[1],tweets_preds4['datestamp']))
tweets_preds4['year'] = (np.where(tweets_preds4['datestamp'].str.contains('-'),tweets_preds4['datestamp'].str.split('-').str[0],tweets_preds4['datestamp']))
tweets_preds4['Date'] =  pd.to_datetime(tweets_preds4['datestamp'], format='%Y-%m-%d')
tweets_preds4["Day_of_Week"] = tweets_preds4.Date.dt.weekday
tweets_preds4["First_day_of_the_week"] = tweets_preds4.Date - tweets_preds4.Day_of_Week * timedelta(days=1)
tweets_preds4 = tweets_preds4[tweets_preds4.loc[:,'First_day_of_the_week'] >= '2021-01-01']
#======================country filter============================
tweets_preds4 = tweets_preds4[tweets_preds4["country"].isin(["IN"])]

#======================rolling predictions=======================
a = tweets_preds4.groupby(["datestamp"]).count()[["tweet"]]
b = (tweets_preds4[tweets_preds4.loc[:,"pred_multi"] == 0]).groupby(["datestamp"]).count()[["tweet"]]
c = (tweets_preds4[tweets_preds4.loc[:,"pred_multi"] == 1]).groupby(["datestamp"]).count()[["tweet"]]
d = (tweets_preds4[tweets_preds4.loc[:,"pred_multi"] == 2]).groupby(["datestamp"]).count()[["tweet"]]

final_2021 = a.merge(b, on = ["datestamp"], how = "left").merge(c, on = ["datestamp"], how = "left").merge(d, on = ["datestamp"], how = "left").reset_index().fillna(0)
final_2021.columns = ["Date", "total", "primary", "secondary", "thirdparty"]

for col in final_2021.columns[1:]:
    new_col = col + "_rolling"
    final_2021[new_col] = final_2021[col].rolling(7, min_periods = 1).sum()

data_rolling = pd.read_csv('/path/to/output/folder/Indian_series_rolling_daily.csv')

cols = ["Date", "cases_rolling", "deaths_rolling", "total_rolling", "primary_rolling", "secondary_rolling", "thirdparty_rolling"]
finallll = data_rolling.merge(final_2021, on = "Date")[cols]

finallll.columns = ["Date", "cases", "deaths", "total", "primary", "secondary", "thirdparty"]

finallll = finallll[finallll["Date"] >= '2020-02-05']
finallll.to_csv('/path/to/output/folder/Indian_series_rolling_data.csv', index = 0)

final = pd.read_csv('/path/to/output/folder/Indian_series_rolling_data.csv')
final.columns = ["Date", "cases_rolling", "deaths_rolling", "total", "primary", "secondary", "thirdparty"]

indian = pd.read_csv('/path/to/output/folder/Indian_series_rolling_daily.csv')
indian = indian[["Date", "cases1", "deaths1"]]
indian.columns = ["Date", "cases", "deaths"]

final.merge(indian, on = "Date").to_csv('/path/to/output/folder/Indian_series_rolling_data2.csv', index = 0)

