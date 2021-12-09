import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

stage1 = "/path/to/stage1/folder"
stage2 = "/path/to/stage2/folder"
for fold in range(1,6):
    stage1_op = pd.read_csv(stage1 + "fold" + str(fold) + "_predictions.csv")
    stage2_op = pd.read_csv(stage2 + "fold" + str(fold) + "_predictions.csv")
    y_true1 = np.array(stage1_op.loc[:, "label"])
    y_pred1 = np.array(stage1_op.loc[:, "pred"])
    y_pred2 = np.array(stage2_op.loc[:, "pred"])
    y_pred_final = np.where(np.isin(y_pred1, np.array([1])),np.full(y_pred1.shape, 3), y_pred2)
    tmp = classification_report(y_true1, y_pred_final, digits=4, output_dict= True)
    print(json.dumps(tmp))
    print(confusion_matrix(y_true1, y_pred_final))