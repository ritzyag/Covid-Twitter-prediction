import csv
import re
import string
import random
import html
from tqdm import tqdm
import jsonlines

trans =  str.maketrans(string.punctuation, ' ' * len(string.punctuation), '')

def process(txt):
        txt = html.unescape(txt)
        txt = re.sub('#\s+', ' ', txt)
        txt = re.sub('@\s+', ' ', txt)
        txt = re.sub('https?:\s+', ' ', txt)
        txt = re.sub("'s ", 's ', txt)
        txt = txt.translate(trans)
        txt = re.sub('\s+', ' ', txt)
        # print(txt)
        return txt
        
def jaccard(s1, s2):
        a1 = set(s1.split())
        a2 = set(s2.split())
        
        return len(a1 & a2) / min(len(a1), len(a2))

removed = []

def remove_similar(tweets, length = 0, thresh = 0.8):
        cleaned = [process(row[txtcol]) for row in tweets]
        for row in tweets:
                row[txtcol] = html.unescape(row[txtcol])
        i = 0
        while i < len(tweets):
                if len(cleaned[i]) < length:
                        cleaned.pop(i)
                        tweets.pop(i)               
                else: i+= 1
        i = 0
        with tqdm() as pbar:
                while i < len(tweets):
                        if i and i % 100 == 0:
                                pbar.set_description("Total: %d"%len(tweets))
                                pbar.update(100)
                        j = i + 1
                        while j < len(tweets):
                                if jaccard(cleaned[i], cleaned[j]) > thresh:
                                        removed.append([cleaned[i], cleaned[j]])
                                        if len(cleaned[i]) < len(cleaned[j]):
                                                cleaned.pop(i)
                                                tweets.pop(i)
                                                break
                                        else:
                                                cleaned.pop(j)
                                                tweets.pop(j)
                                else:
                                        j += 1
                        else:
                                i += 1                               
        return tweets
                            
rdr = jsonlines.open('/path/to/jsonl/symp_matches_full.jsonl').iter()
txtcol = "tweet"
L = list(rdr)
L = remove_similar(L)
  
n=0
import json
with open('/path/to/jsonl/symp_matches_full_unique.jsonl', 'w') as fo:
        for row in L:
                json.dump(row, fo)
                fo.write('\n')
                n = n+1

ids = []
rdr = jsonlines.open('/path/to/jsonl/symp_matches_full_unique.jsonl').iter()
with open('/path/to/jsonl/symp_matches_full_unique_ids.csv', "w+") as out:
        for tweet in rdr:
                out.write(str(tweet["id"]))
                out.write("\n")
                ids.append(tweet["id"])