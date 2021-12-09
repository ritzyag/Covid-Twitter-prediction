
from sklearn.model_selection import KFold
import pandas as pd

import pandas as pd
import math
def label_fn(label):
    if label in [0,1,2]:
        return 0
    else:
        return 1

data_path = "/path/to/training/data/with/multiclass/labels"
data_df = pd.read_csv(data_path)
kfold = KFold(5, False, 1)
fold = 1
for train, test in kfold.split(data_df):
    #for stage 1
    train_data1 = data_df.iloc[train, :]
    train_data1["label_stage1"] = train_data1["label"].apply(label_fn)
    #for stage2
    train_data2 = train_data1[train_data1["label"].isin([0,1,2])]
    #common for stage 1 and stage 2
    test_data = data_df.iloc[test, :]
    print(train_data1.shape, test_data.shape)
    print(train_data2.shape, test_data.shape)
    #write data stage 1
    train_data1.loc[:,["tweet", "label_stage1"]].to_csv("/path/to/stage1/folder" + "/train_fold" + str(fold) + ".csv", index = 0)
    #write data stage 2
    train_data2.loc[:,["tweet", "label"]].to_csv("/path/to/stage2/folder" + "/train_fold" + str(fold) + ".csv", index = 0)
    #write test data
    test_data.loc[:,["tweet", "label"]].to_csv("/path/to/stage2/folder" + "/test_fold" + str(fold) + ".csv", index = 0)
    fold = fold+1