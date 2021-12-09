from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import classification_report
import numpy as np
import csv
import json

PATH = '/path/to/multiclass_training/data'

# READ DATA
with open(PATH) as fp:
        x, y = [], []
        rdr = csv.reader(fp)
        header = next(rdr)
        for tmp in rdr:
                x.append(tmp[0])
                y.append(int(tmp[1]))

vectorizer = TfidfVectorizer(min_df = 0.01, max_df = 0.8, ngram_range=(1,1))
X = vectorizer.fit_transform(x)
y = np.array(y)

OUTFILE = '/path/to/jsonl/output/file'
with open(OUTFILE, 'w') as fout:
        folds = KFold(5, shuffle=True, random_state=42)
        for train_id, test_id in folds.split(X):
                X_tr, X_te = X[train_id], X[test_id]
                y_tr, y_te = y[train_id], y[test_id]
                model = RandomForestClassifier()
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                tmp = classification_report(y_te, preds,output_dict=True)
                print(tmp)
                print(json.dumps(tmp), file = fout)