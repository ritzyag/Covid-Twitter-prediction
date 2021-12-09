
import pandas as pd
import torch
# Preliminaries
from torchtext.data import Dataset,Field, TabularDataset, Iterator
# Models
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

import json
import os
import csv
import warnings
import pandas as pd

torch.manual_seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

BERT_MODEL = 'digitalepidemiologylab/covid-twitter-bert-v2'

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, num_labels = 3)

MAX_SEQ_LEN = 100
BATCH_SIZE = 100
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

PRINT = True

if not PRINT:
    warnings.filterwarnings("ignore")

#Fields
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('text', text_field)]

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.encoder = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 3)
    def forward(self, text, label= None):
        output = self.encoder(text, labels=label)[:2]
        if not label is None:
            loss, text_fea = output
        else:
            text_fea = output[0]
            loss = None
        return loss, text_fea

def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    raw_scores = []
    model.eval()
    i = 0
    with torch.no_grad():
        for (text), _ in test_loader:
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                output = model(text)
                _, output = output
                raw_scores.extend(F.softmax(output, dim = 1).tolist())
                y_pred.extend(torch.argmax(output, 1).tolist())
                i = i+1
                print(i)
    return y_pred, raw_scores

def load_checkpoint(load_path, model):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    if PRINT: print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def predict_score(data_path, load_path, model = None):
    data = TabularDataset(path=data_path, format='CSV', fields = fields, skip_header = True)
    test_iter = Iterator(data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
    if not model:
        best_model = BERT().to(device)
        load_checkpoint(load_path + '/model.pt', best_model)
    preds, raw = evaluate(best_model, test_iter)
    return preds, raw

model_location = '/path/to/trained/model'
preds, raw = predict_score('/path/to/data', model_location)

score_list = []
for scores in raw:
    score_list.append((scores[1]))

tweet = pd.read_csv('/path/to/data')
tweet["score"] = score_list
tweet.to_csv("/path/to/output/file", index = 0)