import pandas as pd
import math
def label_fn(label):
    if label == "primary_reporting_tweet":
        return 0
    elif label == "secondary_reporting_tweet":
        return 1
    elif label == "third_party_reporting_tweet":
        return 2
    elif label == "general_reporting_tweet" or (math.isnan(float(label))):
        return 3

#============wave 2===============
data = pd.read_csv("/path/to/crowdsourced_annotated_data_file_2")
data = data.drop_duplicates(subset = "id", keep = "last")
data["label"] = data["choose_the_type_of_reporting_tweet"].apply(label_fn)
data2= data[(data["is_this_a_symptom_reporting_tweet:confidence"] > 0.7 )]

data_reporting = data2[(data2['label'].isin([0,1,2])) & (data2["choose_the_type_of_reporting_tweet:confidence"] > 0.6 )].loc[:, ["id","tweet","label"]]
data_non_reporting = data2[(data2["label"] == 3) | (data2["choose_the_type_of_reporting_tweet:confidence"].isna())].loc[:, ["id","tweet","label"]]
data_high_w2 = pd.concat([data_reporting,data_non_reporting])

#============wave 1===============
data = pd.read_csv("/path/to/crowdsourced_annotated_data_file_1")
data = data.drop_duplicates(subset = "id", keep = "last")
data["label"] = data["choose_the_type_of_reporting_tweet"].apply(label_fn)
data2= data[(data["is_this_a_symptom_reporting_tweet:confidence"] > 0.7 )]

data_reporting = data2[(data2['label'].isin([0,1,2])) & (data2["choose_the_type_of_reporting_tweet:confidence"] > 0.6 )].loc[:, ["id","tweet","label"]]
data_non_reporting = data2[(data2["label"] == 3) | (data2["choose_the_type_of_reporting_tweet:confidence"].isna())].loc[:, ["id","tweet","label"]]
data_high_w1 = pd.concat([data_reporting,data_non_reporting])

#============wave 1 and wave 2 combined===============
train_w1_w2 = pd.concat([data_high_w1, data_high_w2])

train_w1_w2_class_3 = train_w1_w2[train_w1_w2["label"] == 3].sample(n = 500, replace = False, random_state = 42)
train_w1_w2_class_not_3 = train_w1_w2[train_w1_w2['label'].isin([0,1,2])]

data_train_shuffled = pd.concat([train_w1_w2_class_3, train_w1_w2_class_not_3])
data_train_shuffled_final = data_train_shuffled.sample(frac=1).reset_index(drop=True)

data_train_shuffled_final2 = data_train_shuffled_final.drop_duplicates(subset = "id", keep = "last")
data_train_shuffled_final2.groupby("label").count()

data_train_shuffled_final2.loc[:,["tweet", "label"]].to_csv("/path/to/final_multiclass_training_data", index = 0)

train_w1_w2.drop_duplicates(subset = "id", keep = "last").groupby("label").count()