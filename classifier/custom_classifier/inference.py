
import pandas as pd
import torch
# Preliminaries
from torchtext.data import Dataset,Field, TabularDataset, Iterator
# Models
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel
import numpy as np

import json
import os
import csv
import warnings

torch.manual_seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

BERT_MODEL = 'digitalepidemiologylab/covid-twitter-bert-v2'

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, num_labels = 4)

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
        self.encoder = AutoModel.from_pretrained(BERT_MODEL)
        # self.CLayer = nn.Linear(1048,4)
        self.CLayer = nn.Linear(1034,4)
    def forward(self, text, label= None, tokens = None):
        #using tokens to handcraft features
        # print(self.encoder(text))
        embed = self.encoder(text)[0][:,0,:]
        embed_handcraft = torch.cat([embed, tokens], dim = 1)
        text_fea = self.CLayer(embed_handcraft)
        if not label is None:
            loss = F.cross_entropy(text_fea, label)
        else:
            loss = None
        return loss, text_fea

ft_list_1 = ["list", "1"]
ft_list_2 = ["list", "2"]
ft_list_3 = ["list", "3"]
ft_list_4 = ["list", "4"]
ft_list_5 = ["list", "5"]
ft_list_6 = ["list", "6"]
ft_list_7 = ["list", "7"]
ft_list_8 = ["list", "8"]
ft_list_9 = ["list", "9"]
ft_list_10 = ["list", "10"]

feature_words_dict = {"f1" : ft_list_1,"f2" : ft_list_2,"f3" : ft_list_3,"f4" : ft_list_4,"f5": ft_list_5,"f6" : ft_list_6,"f7" : ft_list_7,"f8" : ft_list_8, "f9" : ft_list_9, "f10" : ft_list_10}
feature_list = np.array(["f1","f2","f3","f4","f5", "f6","f7","f8", "f9", "f10"])

def handcraft(token_list):
    token_list = np.array(token_list)
    for key, val in feature_words_dict.items():
        token_list = np.where(np.isin(token_list, val), np.full(token_list.shape,key), token_list)
    vector = np.where(np.isin(feature_list, token_list), np.full(feature_list.shape, 1), np.full(feature_list.shape, 0))
    return vector

def evaluate(model, test_loader):
    y_pred = []
    # y_true = []
    raw_scores = []
    model.eval()
    k = 0
    with torch.no_grad():
        for (text), _ in test_loader:
                vectors_to_append = [None] * BATCH_SIZE
                for i,t in enumerate(text):
                    tokens = tokenizer.convert_ids_to_tokens(t)
                    index = [tokens.index('[SEP]') if '[SEP]' in tokens else 100][0]
                    tokens = tokens[1:index]
                    vector_to_append = handcraft(tokens)
                    vectors_to_append[i] = vector_to_append
                vectors_to_append = vectors_to_append[0:i+1]
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                vectors_to_append = torch.from_numpy(np.array(vectors_to_append)).type(torch.FloatTensor)
                vectors_to_append = vectors_to_append.to(device)
                output = model(text = text,label = None,tokens= vectors_to_append)
                _, output = output
                raw_scores.extend(F.softmax(output, dim = 1).tolist())
                y_pred.extend(torch.argmax(output, 1).tolist())
                k = k+ 1
                print(k)
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

import pandas as pd
tweet = pd.read_csv('/path/to/data')
tweet["pred_multi"] = preds
tweet.to_csv('/path/to/output/file', index = 0)