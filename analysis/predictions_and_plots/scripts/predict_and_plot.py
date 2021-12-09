import pandas as pd
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def make_train_test_data(df, input_var, y_var, lag):
    x_cols = []
    for var in input_var:
        df1 = df[["Date",var]]
        lag1_df1 = pd.concat([df1.Date, df1.shift(lag)], axis = 1).iloc[:, [0,2]].dropna()
        lag1_df1.columns = ["Date", "lag" + str(lag) + "_" + str(var)]
        df = df.merge(lag1_df1, on = "Date")
        x_cols.append("lag" + str(lag) + "_" + str(var))
    df_x = df[x_cols]
    df_y = df[[y_var]]
    dt_col = df[["Date"]]
    # print(df)
    return df_x, df_y, dt_col

def mean_absolute_percentage_error(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def performance_metrics(df_actual, df_pred, type1):
    rmse = np.sqrt(metrics.mean_squared_error(df_actual[type1], df_pred[:,0]))
    mean = np.mean(df_actual[type1])
    df = pd.DataFrame({type1 : df_actual[type1],"pred" : df_pred[:,0]})
    corr_pearson = df[type1].corr(df["pred"], method = "pearson")
    corr_spearman = df[type1].corr(df["pred"], method = "spearman")
    mae = metrics.mean_absolute_error(df_actual[type1], df_pred[:,0])
    mape = mean_absolute_percentage_error(df_actual[type1], df_pred[:,0])
    p_val_Spearman = scipy.stats.spearmanr(df[type1], df["pred"], nan_policy = "omit").pvalue
    p_val_pearson = scipy.stats.pearsonr(df[type1], df["pred"])[1]
    print("RMSE : " + str(rmse) )
    print("mean : " + str(mean) )
    print("corr_pearson : " + str(corr_pearson) )
    print("p_val_pearson : " + str(p_val_pearson) )
    print("corr_spearman : "  + str(corr_spearman) )
    print("p_val_Spearman : "  + str(p_val_Spearman) )
    print("mae : " + str(mae) )
    print("mape : " + str(mape) )
    return df,rmse, mean, corr_pearson, corr_spearman, mae, mape

df = pd.read_csv("path/to/csv_rolling/data")

df["%primary"] = (df["primary"]/df["total"])*100
df["%secondary"] = (df["secondary"]/df["total"])*100
df["%thirdparty"] = (df["thirdparty"]/df["total"])*100
df["reporting"] = df["primary"] + df["secondary"] + df["thirdparty"]
df["%reporting"] = (df["reporting"]/df["total"])*100

df_final2 = df[(df["Date"] >= '2021-01-01' )].reset_index()

to_pred = "cases"
# to_pred = "deaths"

# input_feature = "%primary"
# input_feature = "%secondary"
# input_feature = "%thirdparty"
# input_feature = "%reporting"
# input_feature = "primary"
# input_feature = "secondary"
# input_feature = "thirdparty"
input_feature = "reporting"

df_final_x, df_final_y, dt  = make_train_test_data(df_final2, [input_feature], to_pred, 7)
train_x, test_x, train_y, test_y= train_test_split(df_final_x,df_final_y,test_size=0.30,random_state=42, shuffle = False)
print(dt.iloc[0,0], dt.iloc[train_y.shape[0]-1, 0], dt.iloc[train_y.shape[0], 0], dt.iloc[-1, 0])
print(train_x.shape, test_x.shape)
dt_train, dt_test= train_test_split(dt,test_size=0.30,random_state=42, shuffle = False)

scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

#======================linear regression===================
lr = linear_model.Lasso()
lr.fit(train_x, train_y)

lr_pred_y_test = lr.predict(test_x)
lr_pred_y_train = lr.predict(train_x)

print("============Test Results============")
df_test_lasso,rmse, mean, corr_pearson, corr_spearman, mae, mape = performance_metrics(test_y,lr_pred_y_test, to_pred )
print("============Train Results============")
df_train_lasso,rmse, mean, corr_pearson, corr_spearman, mae, mape = performance_metrics(train_y,lr_pred_y_train, to_pred )

#======================polynomial regression===================
plr = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
plr.fit(train_x, train_y)

plr_pred_y_test = plr.predict(test_x)
plr_pred_y_train = plr.predict(train_x)

print("============Test Results============")
df_test,rmse, mean, corr_pearson, corr_spearman, mae, mape = performance_metrics(test_y,plr_pred_y_test, to_pred )
print("============Train Results============")
df_test_poly,rmse, mean, corr_pearson, corr_spearman, mae, mape = performance_metrics(train_y,plr_pred_y_train, to_pred )

df_test_lasso.columns = [to_pred, "lasso"]
df_test_poly.columns = [to_pred, "polynomial"]
df_all = df_test_lasso.merge(df_test_poly, on = to_pred)
df_all.columns = ["actual_" + to_pred, "pred_lasso", "pred_polynomial"]

def generate_plot(df, to_save, name):
    df["Date"] = dt_test.reset_index().Date
#     print(df)
    colors = ["black", "green", "red", "blue"]
    df.plot(x = "Date", color = colors)
#     df.plot(x = "Date")
    plt.xticks(rotation=45)
    plt.ylabel("#" + to_pred)
    plt.tight_layout()
    plt.grid()
    if to_save:
        plt.savefig("/path/to/output/directory/" + name + ".pdf")
    plt.show()

df_all1 = df_all.div(7)
generate_plot(df_all1,1, "Indian_lasso_poly" )