import json
import string
import os
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
import jsonlines
import time

def check(key, text):
        text = ' ' + text.translate(str.maketrans("","", string.punctuation)).lower() + ' '
        key = key.translate(str.maketrans("","", string.punctuation)).lower()
        key_list = [' ' + key +  ' ' for key in key.split()]
        for word in key_list:
                if word not in text: 
                        return False    
        return True

def check_file(fn):
        samples = defaultdict(list)
        with jsonlines.open(os.path.join(PATH, fn)) as fp:
                rdr = fp.iter()
                for row in rdr:
                        for tp in topics:
                                for pr in prons:
                                        if check(tp + pr, row["tweet"]):
                                                samples[tp].append(row)
        return samples

#==========reading all batches of fetched tweets========
PATH = "/path/to/tweet_batches/"

files = []
for root, dirs, files_all in os.walk(PATH):
    for file1 in files_all:
        if file1.endswith(".jsonl"):
             files.append(file1)

#==========code run parallely on different nodes with 10 files on each node=========
files2 = files[0:10]

with open(PATH + '../../files/expanded_covid_symptom_exps.txt') as fp2:
        prons = ['']
        topics = fp2.read().splitlines()

len(topics)
#6435

N_PROC = 12
with Pool(N_PROC) as p:
        t1 = time.time()
        results = list(tqdm(p.imap(check_file, files2), total = len(files2), ncols = 50))
        t2 = time.time()

time_taken = (t2-t1)/60
print(time_taken)
181.435

# counts = Counter()

samples = defaultdict(list)
for i,cnt in enumerate(results):
        for key in cnt:
                samples[key].extend(cnt[key])

cntlist = sorted([(key, len(samples[key])) for key in samples], reverse = True, key = lambda x: x[1])
with open('/home/mtech0/18CS60R01/ritika/final_data_for_inference/global/all_symp_matched/pron_counts_part2.txt', 'w') as fw:
        for key, val in cntlist:
                print(key + ":", "{:,}".format(val), file= fw)

printlist = []
idset = set()
i = 0
for key in samples:  
        for row in samples[key]:
                i = i+ 1
                if not row["id"] in idset:
                        idset.add(row["id"])      
                        printlist.append(row)

print(len(printlist))

with open('/path/to/jsonl/file_with_matched_symptoms', 'w') as outfile:
        for row in printlist:
                json.dump(row, outfile)               
                outfile.write('\n')

#======example========
files = ['jsonl_files_with_matched_symptom_node1', 'jsonl_files_with_matched_symptom_node2']

#===============combining all symptom tweets from different nodes===============
n = 0
i = 0
id_list = []
for f in files:
        rdr = jsonlines.open('/path/to/jsonl/'+str(f)).iter()
        with open('/path/to/jsonl/symp_matches_full.jsonl', 'a+') as outfile:
                for row in rdr:
                        i = i+1
                        id_1 = row["id"]
                        if id_1 not in id_list:
                                id_list.append(id_1)
                                json.dump(row, outfile)
                                outfile.write('\n')
                                n = n+1

ids = []
rdr = jsonlines.open('/path/to/jsonl/symp_matches_full.jsonl').iter()
with open('/path/to/jsonl/symp_matches_full_ids.jsonl', "a+") as out:
        for tweet in rdr:
                out.write(str(tweet["id"]))
                out.write("\n")
                ids.append(tweet["id"])