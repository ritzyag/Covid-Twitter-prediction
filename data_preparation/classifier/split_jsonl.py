
import csv
import json
import string
import os
import random
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
import gzip
import jsonlines
import json


PATH = "/path/to/data/"
OUTLOCATION = "/path/to/tweet_batches/"
#===15 files===== (1.5 MN)
files = []
for root, dirs, files_all in os.walk(PATH):
    for file1 in files_all:
        if file1.endswith(".gz"):
             files.append(file1)

n_tweets = 100001
itr = 1
n = 0
for f in files:
    with gzip.open(os.path.join(PATH, f)) as fp:
    # with jsonlines.open(os.path.join(PATH, f)) as fp:
        rdr = jsonlines.Reader(fp).iter()
        # rdr = fp.iter()
        cond = True
        while cond:
            if (n > (100000*itr)):
                itr = itr+ 1
            n = n + 1
            file_name = "file_" + str(itr) + ".jsonl"
            with open(OUTLOCATION + str(file_name), 'a+') as outfile:
                for row in rdr:
                        if (n % n_tweets) != 0:
                            if (row["lang"] == "en"):
                                json.dump(row, outfile)
                                outfile.write('\n')
                                n = n + 1
                        else:
                            break
                else:
                    cond = False          
        else:
            continue