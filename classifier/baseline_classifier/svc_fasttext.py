from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import classification_report
import fasttext
import random
import csv
import numpy as np
import json

PATH = '/path/to/multiclass_training/data'
OUT1 = '/path/to/output_folder/'

with open(PATH) as fp:
    next(fp)
    rdr = list(csv.reader(fp.readlines()))
    random.shuffle(rdr)
    with open(OUT1 + 'data.txt', 'w') as fout:
            for data in rdr:
                text = data[0]
                print(text, file = fout)

model = fasttext.train_unsupervised(OUT1 + 'data.txt', model='skipgram')
model.save_model(OUT1 + "fasttext_emb.bin")

ft_model = fasttext.load_model(OUT1 + 'fasttext_emb.bin')

# READ DATA
with open(PATH) as fp:
        x, y = [], []
        rdr = csv.reader(fp)
        header = next(rdr)
        for tmp in rdr:
                x.append(np.mean([ft_model[w] for w in tmp[0].split()], axis = 0))
                y.append(int(tmp[1]))

X = np.array(x)
y = np.array(y)

OUTFILE = '/path/to/jsonl/output/file'
with open(OUTFILE, 'w') as fout:
        folds = KFold(5, shuffle=True, random_state=42)
        for train_id, test_id in folds.split(X):
                X_tr, X_te = X[train_id], X[test_id]
                y_tr, y_te = y[train_id], y[test_id]
                model = SVC()
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                tmp = classification_report(y_te, preds,output_dict=True)
                print(tmp)
                print(json.dumps(tmp), file = fout)